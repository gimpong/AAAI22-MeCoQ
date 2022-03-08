import os
import random
import logging
from argparse import ArgumentParser

import numpy as np
import torch

from utils import set_logger, Monitor, Evaluator, WarmUpAndCosineDecayScheduler
from data import CIFAR10, Flickr25K, NUSWIDE
from network import MeCoQ
from loss import MeCoQLoss
from engine import train, test

def parse_args():
    parser = ArgumentParser(description="Run MeCoQ")
    # dataset configurations
    parser.add_argument('--dataset', 
                        type=str, default='CIFAR10',
                        help="Choose a dataset from 'CIFAR10', 'Flickr25K' or 'NUSWIDE'.")
    parser.add_argument('--protocal', 
                        type=str, default='I',
                        help="Select evaluation protocal on CIFAR10. Options: 'I' or 'II'.")
    parser.add_argument('--download_cifar10', 
                        dest='download_cifar10', action='store_true',
                        help='Download CIFAR-10 via torchvision or not.')
    parser.set_defaults(download_cifar10=False)
    parser.add_argument('--num_workers', 
                        type=int, default=10,
                        help='Number of threads for data fetching.')

    # optimizer configurations
    parser.add_argument('--batch_size', 
                        type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epoch_num', 
                        type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--optimizer', 
                        type=str, default='SGD',
                        help="The name of optimizer in 'torch.optim'.")
    parser.add_argument('--lr', 
                        type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--lr_scaling',
                        type=float, default=1e-3,
                        help='Learning rate scaling for CNN layers.')
    parser.add_argument('--momentum', 
                        type=float, default=0.9,
                        help='Learning rate.')
    parser.add_argument('--hp_beta', 
                        type=float, default=5e-6,
                        help='Weight decay factor.')
    parser.add_argument('--disable_scheduler', 
                        dest='use_scheduler', action='store_false',
                        help='Disabling the learning rate scheduler.')
    parser.set_defaults(use_scheduler=True)
    parser.add_argument('--warmup_epoch_num',
                        type=int, default=1,
                        help='Number of warmup epochs for lr scheduler.')
    parser.add_argument('--start_lr',
                        type=float, default=1e-5,
                        help='Learning rate at the start of warmup.')
    parser.add_argument('--final_lr',
                        type=float, default=1e-5,
                        help='Final learning rate of cosine decaying schedule.')

    # quantization configurations
    parser.add_argument('--feat_dim', 
                        type=int, default=64,
                        help='Dimension of image features.')
    parser.add_argument('--M', 
                        type=int, default=4,
                        help='Number of codebooks.')
    parser.add_argument('--K', 
                        type=int, default=256,
                        help='Number of sub-codewords per sub-codebook.')
    parser.add_argument('--alpha', 
                        type=float, default=10,
                        help='Alpha scaling parameter for soft codeword assignment.')
    parser.add_argument('--trainable_layer_num', 
                        type=int, default=0,
                        help='The number of trainable layers for VGG-16 backbone. Options: 0, 1 or 2.')
    parser.add_argument('--vgg_model_path', 
                        type=str, default=None,
                        help='The path of pretrained VGG-16 model weights. If not declared, it will download the weights from TorchVision')

    # contrastive learning configurations
    parser.add_argument('--T', 
                        type=float, default=0.1,
                        help='Temperature parameter for nce loss.')
    parser.add_argument('--mode',
                        type=str, default='simple',
                        help="Loss mode of contrastive learning. Options: 'simple', 'debias'.")
    parser.add_argument('--pos_prior',
                        type=float, default=0,
                        help='Class prior of positive samples among the dataset.')
    parser.add_argument('--hp_lambda', 
                        type=float, default=1,
                        help='Weight for entropy regularization loss.')
    parser.add_argument('--hp_gamma', 
                        type=float, default=0.5,
                        help='Weight for codebook regularization loss.')

    # memory queue configurations
    parser.add_argument('--queue_begin_epoch', 
                        type=int, default=np.inf,
                        help='The epoch for starting using memory queue.')

    # evaluation configurations
    parser.add_argument('--symmetric_distance', 
                        dest='is_asym_dist', action='store_false',
                        help='Declare this option to use symmetric quantization distance, otherwise to use asymmetric quantization distance.')
    parser.set_defaults(is_asym_dist=True)
    parser.add_argument('--topK', 
                        type=int, default=None,
                        help='TopK for metric evaluation')
    parser.add_argument('--eval_interval', 
                        type=int, default=1,
                        help='Interval for evaluation (in epoch).')
    parser.add_argument('--monitor_counter', 
                        type=int, default=10,
                        help='The maximum patience for metric monitor.')

    # other configurations
    parser.add_argument('--device', 
                        type=str, default='cpu',
                        help="Device: 'cpu', 'cuda:X'")
    parser.add_argument('--seed', 
                        type=int, default=2021,
                        help='Random seed.')
    parser.add_argument('--notes', 
                        type=str, default="",
                        help="Notes and remarks for current experiment.")
    parser.add_argument('--disable_writer',
                        dest='use_writer', action='store_false',
                        help='Disabling tensorboard summary writer.')
    parser.set_defaults(use_writer=True)

    return parser.parse_args()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    writer = set_logger(config)

    logging.info("config: " + str(config))
    logging.info("prepare %s datatset" % config.dataset)
    if config.dataset == 'CIFAR10':
        datahub = CIFAR10(root='./datasets/CIFAR-10/',
                          protocal=config.protocal,
                          download=config.download_cifar10,
                          batch_size=config.batch_size,
                          num_workers=config.num_workers)
    elif config.dataset == 'Flickr25K':
        datahub = Flickr25K(root="./data/Flickr25k/",
                            img_root="./datasets/Flickr25K/mirflickr/",
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
    elif config.dataset == 'NUSWIDE':
        datahub = NUSWIDE(root="./data/Nuswide/",
                          img_root="./datasets/NUS-WIDE/Flickr/",
                          batch_size=config.batch_size,
                          num_workers=config.num_workers)
    else:
        raise ValueError("Unknown dataset '%s'." % config.dataset)

    logging.info("setup model")
    model = MeCoQ(feat_dim=config.feat_dim, 
                  M=config.M, K=config.K, alpha=config.alpha, 
                  trainable_layer_num=config.trainable_layer_num, 
                  CNN_model_path=config.vgg_model_path)
    model = model.to(config.device)

    logging.info("define loss function")
    loss_fn = MeCoQLoss(T=config.T,
                        mode=config.mode,
                        pos_prior=config.pos_prior,
                        hp_lambda=config.hp_lambda,
                        hp_gamma=config.hp_gamma,
                        device=config.device)

    logging.info("setup %s optimizer" % config.optimizer)
    if config.optimizer == 'SGD':
        params = [
            {'params': model.vgg.parameters(), 'lr': config.lr * config.lr_scaling},
            {'params': model.projection.parameters(), 'lr': config.lr},
            {'params': model.pq_layer.parameters(), 'lr': config.lr}
        ]
        optimizer = torch.optim.SGD(params,
                                    lr=config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.hp_beta)
    else:
        params = [
            {'params': model.vgg.parameters(), 'lr': config.lr * config.lr_scaling},
            {'params': model.projection.parameters(), 'lr': config.lr},
            {'params': model.pq_layer.parameters(), 'lr': config.lr}
        ]
        optimizer = getattr(torch.optim, config.optimizer)(params,
                                                           lr=config.lr,
                                                           weight_decay=config.hp_beta)

    logging.info("prepare monitor and evaluator")
    monitor = Monitor(max_patience=config.monitor_counter)
    evaluator = Evaluator(feat_dim=config.feat_dim,
                          M=config.M, K=config.K,
                          is_asym_dist=config.is_asym_dist,
                          device=config.device)

    lr_scheduler = WarmUpAndCosineDecayScheduler(optimizer=optimizer, 
                                                 start_lr=config.start_lr, 
                                                 base_lr=config.lr, 
                                                 final_lr=config.final_lr,
                                                 epoch_num=config.epoch_num, 
                                                 batch_num_per_epoch=len(datahub.train_loader), 
                                                 warmup_epoch_num=config.warmup_epoch_num) if config.use_scheduler else None

    logging.info("begin to train model")
    train(datahub=datahub,
          model=model,
          loss_fn=loss_fn,
          optimizer=optimizer,
          lr_scheduler=lr_scheduler,
          config=config,
          evaluator=evaluator,
          monitor=monitor,
          writer=writer)

    # Load best checkpoint
    logging.info("finish training, now load the best model and codes")
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_root, 'model.cpt')))
    evaluator.set_codebooks(codebooks=model.codebooks)
    evaluator.set_db_codes(db_code_file=os.path.join(config.checkpoint_root, 'db_codes.npy'))
    evaluator.set_db_targets(db_target_file=os.path.join(config.checkpoint_root, 'db_targets.npy'))

    logging.info("begin to test model")
    test(datahub=datahub,
         model=model,
         config=config,
         evaluator=evaluator,
         writer=writer)
    logging.info("finish all procedures")
