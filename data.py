from PIL import Image, ImageFilter
from numpy.core.fromnumeric import resize
import cv2
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import transforms
from utils import read_and_parse_file


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self):
        self.base_transform = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply(
                                                      [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.7),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlur(3),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                                       [0.229, 0.224, 0.225])
                                                  ])

    def __call__(self, raw_img):
        view_1 = self.base_transform(raw_img)
        view_2 = self.base_transform(raw_img)
        return [view_1, view_2]


class ImgDataset(torch.utils.data.Dataset):
    """Some Information about ImgDataset"""
    def __init__(self, data=None, targets=None, transform=None, target_transform=None, img_root=None):
        super(ImgDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.img_root = img_root

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.img_root:
            img = Image.open(os.path.join(self.img_root, img)).convert('RGB')
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10:
    def __init__(self, 
                 root, 
                 protocal='I', 
                 download=False, 
                 batch_size=128, 
                 num_workers=4):
        data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Fetch data
        self._train_dataset = dsets.CIFAR10(root=root,
                                            train=True,
                                            transform=TwoCropsTransform(),
                                            download=download)
        self._test_dataset = dsets.CIFAR10(root=root,
                                           train=False,
                                           transform=data_transforms)
        self._database_dataset = ImgDataset(transform=data_transforms)

        # Reconstruct datasets
        all_data = list(np.concatenate([self._train_dataset.data, self._test_dataset.data]))
        all_targets = np.concatenate([self._train_dataset.targets, self._test_dataset.targets])

        all_pairs = pd.DataFrame({'data': all_data, 'targets': all_targets})
        all_pair_grps = all_pairs.groupby('targets')

        if protocal == 'I':
            self._protocal_I(all_pair_grps)
        elif protocal == 'II':
            self._protocal_II(all_pair_grps)
        else:
            raise ValueError("Protocal %s is not implemented." % protocal)

        # Setup data loaders
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=num_workers)
        self._test_loader = torch.utils.data.DataLoader(dataset=self._test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)
        self._database_loader = torch.utils.data.DataLoader(dataset=self._database_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers)

    def _protocal_I(self, all_pair_grps):
        train_list, test_list = [], []
        for label, pair_group in all_pair_grps:
            perm = np.random.permutation(len(pair_group))
            pair_group = pair_group.reset_index(drop=True)
            train_list.append(pair_group.take(perm[1000:]))
            test_list.append(pair_group.take(perm[:1000]))

        train_pairs = pd.concat(train_list)
        test_pairs = pd.concat(test_list)

        self._train_dataset.data = train_pairs.data.to_numpy()
        self._train_dataset.targets = train_pairs.targets.to_numpy(dtype=np.int8)
        self._database_dataset.data = train_pairs.data.to_numpy()
        self._database_dataset.targets = train_pairs.targets.to_numpy(dtype=np.int8)
        self._test_dataset.data = test_pairs.data.to_numpy()
        self._test_dataset.targets = test_pairs.targets.to_numpy(dtype=np.int8)

    def _protocal_II(self, all_pair_grps):
        train_list, database_list, test_list = [], [], []
        for label, pair_group in all_pair_grps:
            perm = np.random.permutation(len(pair_group))
            pair_group = pair_group.reset_index(drop=True)
            train_list.append(pair_group.take(perm[1000:1500]))
            database_list.append(pair_group.take(perm[1000:]))
            test_list.append(pair_group.take(perm[:1000]))

        train_pairs = pd.concat(train_list)
        database_pairs = pd.concat(database_list)
        test_pairs = pd.concat(test_list)

        self._train_dataset.data = train_pairs.data.to_numpy()
        self._train_dataset.targets = train_pairs.targets.to_numpy(dtype=np.int8)
        self._database_dataset.data = database_pairs.data.to_numpy()
        self._database_dataset.targets = database_pairs.targets.to_numpy(dtype=np.int8)
        self._test_dataset.data = test_pairs.data.to_numpy()
        self._test_dataset.targets = test_pairs.targets.to_numpy(dtype=np.int8)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def database_dataset(self):
        return self._database_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def database_loader(self):
        return self._database_loader

    @property
    def test_loader(self):
        return self._test_loader


class Flickr25K:
    def __init__(self, 
                 root, 
                 img_root, 
                 batch_size=128, 
                 num_workers=4):
        '''
            root: root of image path file
            img_root: root of image file
        '''
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Fetch data
        with open(os.path.join(root, 'img.txt'), 'r') as image_file:
            data = np.array([i.strip() for i in image_file])
        targets = np.loadtxt(os.path.join(root, 'targets.txt'), dtype=np.int8)

        # Split dataset
        perm_index = np.random.permutation(len(data))
        train_index = perm_index[2000: 2000+5000]
        database_index = perm_index[2000:]
        test_index = perm_index[:2000]
        self._train_dataset = ImgDataset(data=data[train_index],
                                         targets=targets[train_index],
                                         transform=TwoCropsTransform(),
                                         img_root=img_root)
        self._database_dataset = ImgDataset(data=data[database_index],
                                            targets=targets[database_index],
                                            transform=test_transforms,
                                            img_root=img_root)
        self._test_dataset = ImgDataset(data=data[test_index],
                                        targets=targets[test_index],
                                        transform=test_transforms,
                                        img_root=img_root)

        # Setup data loaders
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=num_workers)
        self._test_loader = torch.utils.data.DataLoader(dataset=self._test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)
        self._database_loader = torch.utils.data.DataLoader(dataset=self._database_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def database_dataset(self):
        return self._database_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def database_loader(self):
        return self._database_loader

    @property
    def test_loader(self):
        return self._test_loader


class NUSWIDE:
    def __init__(self, 
                 root, 
                 img_root, 
                 batch_size=128, 
                 num_workers=4):
        '''
            root: root of image path file
            img_root: root of image file
        '''
        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Fetch data
        train_data, train_targets = read_and_parse_file(os.path.join(root, 'train.txt'))
        database_data, database_targets = read_and_parse_file(os.path.join(root, 'database.txt'))
        test_data, test_targets = read_and_parse_file(os.path.join(root, 'test.txt'))

        self._train_dataset = ImgDataset(data=train_data,
                                         targets=train_targets,
                                         transform=TwoCropsTransform(),
                                         img_root=img_root)
        self._database_dataset = ImgDataset(data=database_data,
                                            targets=database_targets,
                                            transform=test_transforms,
                                            img_root=img_root)
        self._test_dataset = ImgDataset(data=test_data,
                                        targets=test_targets,
                                        transform=test_transforms,
                                        img_root=img_root)

        # Setup data loaders
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=num_workers)
        self._test_loader = torch.utils.data.DataLoader(dataset=self._test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers)
        self._database_loader = torch.utils.data.DataLoader(dataset=self._database_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def database_dataset(self):
        return self._database_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def database_loader(self):
        return self._database_loader

    @property
    def test_loader(self):
        return self._test_loader
