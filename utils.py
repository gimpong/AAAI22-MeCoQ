import os
import shutil
import logging
from tqdm import tqdm
from datetime import datetime

import math
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np


class WarmUpAndCosineDecayScheduler:
    def __init__(self, optimizer, start_lr, base_lr, final_lr,
                 epoch_num, batch_num_per_epoch, warmup_epoch_num):
        self.optimizer = optimizer
        self.step_counter = 0
        warmup_step_num = batch_num_per_epoch * warmup_epoch_num
        decay_step_num = batch_num_per_epoch * (epoch_num - warmup_epoch_num)
        warmup_lr_schedule = np.linspace(start_lr, base_lr, warmup_step_num)
        cosine_lr_schedule = final_lr + 0.5 * \
            (base_lr - final_lr) * (1 + np.cos(np.pi *
                                               np.arange(decay_step_num) / decay_step_num))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # step at each mini-batch
    def step(self):
        curr_lr = self.lr_schedule[self.step_counter]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = curr_lr
        self.step_counter += 1
        return curr_lr

@torch.no_grad()
def get_db_codes_and_targets(database_loader, model, device='cpu'):
    model.eval()
    code_list, target_list = [], []
    for data, targets in database_loader:
        data, targets = data.to(device), targets.to(device)
        target_list.append(targets)
        _, _, codes = model(data, hard_quant=True)
        code_list.append(codes)

    db_codes = torch.cat(code_list)
    db_targets = torch.cat(target_list)
    model.train()
    return db_codes, db_targets


def save_tensor(tensor, to_path):
    with open(to_path, 'wb') as f:
        np.save(f, tensor.cpu().numpy())


def read_tensor(from_path, device='cpu'):
    with open(from_path, 'rb') as f:
        data = torch.from_numpy(np.load(f)).to(device)
    return data


def read_and_parse_file(file_path):
    data_tbl = np.loadtxt(file_path, dtype=np.str)
    data, targets = data_tbl[:, 0], data_tbl[:, 1:].astype(np.int8)
    return data, targets


def my_add_scalar(writer, dic, global_step):
    for key, value in dic.items():
        writer.add_scalar(key, value, global_step)


def set_logger(config):
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./checkpoints/", exist_ok=True)
    if config.notes:
        prefix = config.notes
    else:
        prefix = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_file = os.path.join('./logs/', prefix + '.log')
    config.__dict__['checkpoint_root'] = os.path.join('./checkpoints/', prefix)
    os.makedirs(config.checkpoint_root, exist_ok=True)

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s.',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

    writer_root = os.path.join('./logs/', prefix + '.writer')
    if os.path.exists(writer_root):
        shutil.rmtree(writer_root)
    writer = SummaryWriter(writer_root) if config.use_writer else None
    return writer


class Monitor:
    def __init__(self, max_patience=5, delta=1e-6):
        self.counter_ = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta

    def update(self, cur_value):
        self.counter_ += 1
        is_break = False
        is_lose_patience = False
        if cur_value < self.best_value + self.delta:
            cur_value = 0
            self.patience -= 1
            logging.info("the monitor loses its patience to %d!" %
                         self.patience)
            is_lose_patience = True
            if self.patience == 0:
                self.patience = self.max_patience
                is_break = True
        else:
            self.patience = self.max_patience
            self.best_value = cur_value
            cur_value = 0
        return (is_break, is_lose_patience)

    @property
    def counter(self):
        return self.counter_


class Evaluator:
    def __init__(self, feat_dim, M, K, codebooks=None, db_codes=None, db_targets=None, is_asym_dist=True, 
                 codebook_file=None, db_code_file=None, db_target_file=None, device='cpu'):
        self.feat_dim, self.M, self.K, self.D, self.device = feat_dim, M, K, feat_dim//M, device
        self.is_asym_dist = is_asym_dist
        self.set_codebooks(codebooks, codebook_file)
        self.set_db_codes(db_codes, db_code_file)
        self.set_db_targets(db_targets, db_target_file)

    def set_codebooks(self, codebooks=None, codebook_file=None):
        if codebook_file:  # Higher priority
            self.C = read_tensor(codebook_file, device=self.device)
        else:
            self.C = codebooks
        # Compute the lookup tables after updating the codebooks
        if (not self.is_asym_dist) and (self.C is not None):
            with torch.no_grad():
                # C:[MxKxD], intra_dist_tbls:[MxKxK]
                self.intra_dist_tbls = torch.einsum('mkd,mjd->mkj', self.C, self.C)

    def set_db_codes(self, db_codes=None, db_code_file=None):
        # db_codes:[db_sizexM]
        if db_code_file:  # Higher priority
            self.db_codes = read_tensor(db_code_file, device=self.device)
        else:
            self.db_codes = db_codes

    def set_db_targets(self, db_targets=None, db_target_file=None):
        # db_targets:[db_size](single target version) OR [db_sizextgt_size](multi-target version)
        if db_target_file: # Higher priority
            self.db_targets = read_tensor(db_target_file, device=self.device)
        else:
            self.db_targets = db_targets

    def _symmetric_distance(self, query_codes):
        # query_codes:[bxM]
        dists = self.intra_dist_tbls[0][query_codes[:,0]][:, self.db_codes[:,0]]
        for i in range(1, self.M):
            # intra_dist_tbls[i]:[KxK].index(query_codes[:,i]:[b])=>[bxK]
            # intra_dist_tbls[i][query_codes[:,i]]:[bxK].column_index(db_codes[:,i]:[db_size])=>[bxdb_size]
            sub_dists = self.intra_dist_tbls[i][query_codes[:,i]][:, self.db_codes[:,i]]
            dists += sub_dists
        return dists

    def _asymmetric_distance(self, query_feats):
        # query_feats:[bxfeat_dim]=>[bxMxD]
        query_feats_ = query_feats.view(query_feats.shape[0], self.M, self.D)
        # x_:[bxMxD], C:[MxKxD] => qry_asym_dist_tbl:[MxbxK]
        qry_asym_dist_tbl = torch.einsum('bmd,mkd->mbk', query_feats_, self.C)
        # qry_asym_dist_tbl[i]:[bxK].column_index(db_codes[:,i]:[db_size])=>[bxdb_size]
        dists = qry_asym_dist_tbl[0][:, self.db_codes[:,0]]
        for i in range(1, self.M):
            sub_dists = qry_asym_dist_tbl[i][:, self.db_codes[:,i]]
            dists += sub_dists
        return dists

    @torch.no_grad()
    def distance(self, query_inputs):
        if self.is_asym_dist:
            return self._asymmetric_distance(query_inputs)
        else:
            return self._symmetric_distance(query_inputs)

    @torch.no_grad()
    def MAP(self, test_loader, model, topK=None, test_batch_num=np.inf):
        model.eval()
        AP_list = []
        for i, (query_data, query_targets) in enumerate(tqdm(test_loader, desc="Test batch")):
            query_data, query_targets = query_data.to(self.device), query_targets.to(self.device)
            if self.is_asym_dist:
                feats = model(query_data, only_feats=True, norm_feats=False)
                dists = self.distance(feats)
            else:
                _, _, codes = model(query_data, hard_quant=True)
                dists = self.distance(codes)
            top_indices = torch.argsort(dists, descending=True)
            if topK:
                top_indices = top_indices[:, :topK]
            else: # topK is None
                topK = top_indices.shape[-1]
            
            # db_targets:[db_size] OR [db_sizexlabel_size].index(top_indices:[bxtopK])=>[bxtopK] OR [bxtopKxlabel_size]
            top_targets = self.db_targets[top_indices]

            # query_targets:[bxlabel_size] or [b]
            # single target version
            if len(query_targets.shape) == 1 and len(self.db_targets.shape) == 1:
                # top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=-1) == top_targets).float()
            # multi-target version
            elif len(query_targets.shape) == 2 and len(self.db_targets.shape) == 2:
                # query_targets:[bxlabel_size].matmul(top_targets:[bxtopKxlabel_size])=>top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=1) * top_targets).sum(dim=-1).bool().float()
            else:
                raise RuntimeError("Invalid target shape: dimension of query target is %d, and dimension of database target is %d" % 
                                    (len(query_targets.shape), len(self.db_targets.shape)))

            # hit_counts:[b]
            hit_counts = top_hit_list.sum(dim=-1)
            hit_counts[hit_counts <= 10e-6] = 1.0 # avoid zero division
            # hit_cumsum_list:[bxtopK]
            hit_cumsum_list = top_hit_list.cumsum(dim=-1)
            # position_list:[topK]
            position_list = torch.arange(1, topK+1, dtype=torch.float, device=self.device)
            # precision_list:[bxtopK]
            precision_list = hit_cumsum_list / position_list
            # recall_list:[bxtopK]
            recall_list = hit_cumsum_list / hit_counts.unsqueeze(dim=-1)
            # AP:[b]
            AP = (precision_list * top_hit_list).sum(dim=-1) / hit_counts
            AP_list.append(AP)

            if i + 1 >= test_batch_num:
                break
    
        mAP = torch.cat(AP_list).mean().item()
        model.train()
        return mAP

    @torch.no_grad()
    def PR_curve(self, test_loader, model):
        model.eval()
        precisions, recalls = [], []
        for query_data, query_targets in test_loader:
            query_data, query_targets = query_data.to(self.device), query_targets.to(self.device)
            if self.is_asym_dist:
                feats = model(query_data, only_feats=True, norm_feats=False)
                dists = self.distance(feats)
            else:
                _, _, codes = model(query_data, hard_quant=True)
                dists = self.distance(codes)
            top_indices = torch.argsort(dists, descending=True)
            
            # db_targets:[db_size] OR [db_sizexlabel_size].index(top_indices:[bxdb_size])=>[bxdb_size] OR [bxdb_sizexlabel_size]
            top_targets = self.db_targets[top_indices]

            # query_targets:[bxlabel_size] or [b]
            # single target version
            if len(query_targets.shape) == 1 and len(self.db_targets.shape) == 1:
                # top_hit_list:[bxdb_size]
                top_hit_list = (query_targets.unsqueeze(dim=-1) == top_targets).float()
            # multi-target version
            elif len(query_targets.shape) == 2 and len(self.db_targets.shape) == 2:
                # query_targets:[bxlabel_size].matmul(top_targets:[bxdb_sizexlabel_size])=>top_hit_list:[bxdb_size]
                top_hit_list = (query_targets.unsqueeze(dim=1) * top_targets).sum(dim=-1).bool().float()
            else:
                raise RuntimeError("Invalid target shape: dimension of query target is %d, and dimension of database target is %d" % 
                                    (len(query_targets.shape), len(self.db_targets.shape)))

            # hit_counts:[b]
            hit_counts = top_hit_list.sum(dim=-1)
            hit_counts[hit_counts <= 10e-6] = 1.0 # avoid zero division
            # hit_cumsum_list:[bxdb_size]
            hit_cumsum_list = top_hit_list.cumsum(dim=-1)
            # position_list:[db_size]
            position_list = torch.arange(1, len(self.db_targets)+1, dtype=torch.float, device=self.device)
            # precision_list:[bxdb_size]
            precision_list = hit_cumsum_list / position_list
            # recall_list:[bxdb_size]
            recall_list = hit_cumsum_list / hit_counts.unsqueeze(dim=-1)
            precisions.append(precision_list)
            recalls.append(recall_list)
    
        precision_axis = torch.cat(precisions).mean(dim=0)
        recall_axis = torch.cat(recalls).mean(dim=0)
        model.train()
        return np.stack([recall_axis.cpu().numpy(), precision_axis.cpu().numpy()])
    
    @torch.no_grad()
    def P_at_topK_curve(self, test_loader, model, topK):
        model.eval()
        precisions = []
        for query_data, query_targets in test_loader:
            query_data, query_targets = query_data.to(self.device), query_targets.to(self.device)
            if self.is_asym_dist:
                feats = model(query_data, only_feats=True, norm_feats=False)
                dists = self.distance(feats)
            else:
                _, _, codes = model(query_data, hard_quant=True)
                dists = self.distance(codes)
            top_indices = torch.argsort(dists, descending=True)
            if topK:
                top_indices = top_indices[:, :topK]
            else: # topK is None
                topK = top_indices.shape[-1]
            
            # db_targets:[db_size] OR [db_sizexlabel_size].index(top_indices:[bxtopK])=>[bxtopK] OR [bxtopKxlabel_size]
            top_targets = self.db_targets[top_indices]

            # query_targets:[bxlabel_size] or [b]
            # single target version
            if len(query_targets.shape) == 1 and len(self.db_targets.shape) == 1:
                # top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=-1) == top_targets).float()
            # multi-target version
            elif len(query_targets.shape) == 2 and len(self.db_targets.shape) == 2:
                # query_targets:[bxlabel_size].matmul(top_targets:[bxtopKxlabel_size])=>top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=1) * top_targets).sum(dim=-1).bool().float()
            else:
                raise RuntimeError("Invalid target shape: dimension of query target is %d, and dimension of database target is %d" % 
                                    (len(query_targets.shape), len(self.db_targets.shape)))

            # hit_cumsum_list:[bxtopK]
            hit_cumsum_list = top_hit_list.cumsum(dim=-1)
            # position_list:[topK]
            position_list = torch.arange(1, topK+1, dtype=torch.float, device=self.device)
            # precision_list:[bxtopK]
            precision_list = hit_cumsum_list / position_list
            precisions.append(precision_list)
            
        precision_axis = torch.cat(precisions).mean(dim=0)
        model.train()
        return np.stack([np.arange(1, topK+1, dtype=np.float), precision_axis.cpu().numpy()])



@torch.no_grad()
def get_db_feats_and_targets(database_loader, model, device='cpu'):
    model.eval()
    feat_list, target_list = [], []
    for data, targets in database_loader:
        data, targets = data.to(device), targets.to(device)
        target_list.append(targets)
        feats = model(data, only_feats=True, norm_feats=False)
        feat_list.append(feats)

    db_feats = torch.cat(feat_list)
    db_targets = torch.cat(target_list)
    model.train()
    return db_feats, db_targets



class FeatureEvaluator:
    def __init__(self, feat_dim, db_feats, db_targets, device='cpu'):
        self.feat_dim, self.device = feat_dim, device
        # db_feats:[db_sizexfeat_dim]
        self.db_feats = db_feats
        # db_targets:[db_size](single target version) OR [db_sizextgt_size](multi-target version)
        self.db_targets = db_targets

    @torch.no_grad()
    def distance(self, query_feats):
        # query_feats:[bxfeat_dim].matmul(self.db_feats.T:[feat_dimxdb_size])=>[bxdb_size]
        return query_feats.matmul(self.db_feats.T)

    @torch.no_grad()
    def MAP(self, test_loader, model, topK=None, test_batch_num=np.inf):
        model.eval()
        AP_list = []
        for i, (query_data, query_targets) in enumerate(tqdm(test_loader, desc="Test batch")):
            query_data, query_targets = query_data.to(self.device), query_targets.to(self.device)
            feats = model(query_data, only_feats=True, norm_feats=False)
            dists = self.distance(feats)
            top_indices = torch.argsort(dists, descending=True)
            if topK:
                top_indices = top_indices[:, :topK]
            else: # topK is None
                topK = top_indices.shape[-1]

            # db_targets:[db_size] OR [db_sizexlabel_size].index(top_indices:[bxtopK])=>[bxtopK] OR [bxtopKxlabel_size]
            top_targets = self.db_targets[top_indices]

            # query_targets:[bxlabel_size] or [b]
            # single target version
            if len(query_targets.shape) == 1 and len(self.db_targets.shape) == 1:
                # top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=-1) == top_targets).float()
            # multi-target version
            elif len(query_targets.shape) == 2 and len(self.db_targets.shape) == 2:
                # query_targets:[bxlabel_size].matmul(top_targets:[bxtopKxlabel_size])=>top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=1) * top_targets).sum(dim=-1).bool().float()
            else:
                raise RuntimeError("Invalid target shape: dimension of query target is %d, and dimension of database target is %d" % 
                                    (len(query_targets.shape), len(self.db_targets.shape)))

            # hit_counts:[b]
            hit_counts = top_hit_list.sum(dim=-1)
            hit_counts[hit_counts <= 10e-6] = 1.0 # avoid zero division
            # hit_cumsum_list:[bxtopK]
            hit_cumsum_list = top_hit_list.cumsum(dim=-1)
            # position_list:[topK]
            position_list = torch.arange(1, topK+1, dtype=torch.float, device=self.device)
            # precision_list:[bxtopK]
            precision_list = hit_cumsum_list / position_list
            # recall_list:[bxtopK]
            recall_list = hit_cumsum_list / hit_counts.unsqueeze(dim=-1)
            # AP:[b]
            AP = (precision_list * top_hit_list).sum(dim=-1) / hit_counts
            AP_list.append(AP)

            if i + 1 >= test_batch_num:
                break
    
        mAP = torch.cat(AP_list).mean().item()
        model.train()
        return mAP