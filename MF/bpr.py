# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from cvpods.utils.xklib import stack_data_from_batch

class Bpr(nn.Module):
    """
        the simplest recommendation model, used to test the successful of 
        transfer cvpack2 to recommendation area
    """
    def __init__(self, cfg):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Bpr, self).__init__()
        
        self.user_embedding = torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_USER, cfg.MODEL.BPR.DIM)
        self.item_embedding = torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_ITEM, cfg.MODEL.BPR.DIM)
        self.user_bias= torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_USER, 1)
        self.item_bias= torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_ITEM, 1)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.l2_norm_alpha = cfg.MODEL.BPR.L2_NORM
        self.sum = 0.0
        self.steps = 0.0
        self.init_embedding()
        self.to(self.device)
        
    def init_embedding(self, init = 0): 
        nn.init.kaiming_normal_(self.item_embedding.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_embedding.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        l2_loss = (torch.sum(self.user_embedding(users)**2) + torch.sum(self.item_embedding(items)**2)) / 2
        return l2_loss

    def forward(self, batched_inputs):
        """
        Input:
        Output:
        """
        #import pdb
        #pdb.set_trace() 
        #print ('output:', batched_inputs)
        users = stack_data_from_batch(batched_inputs, 'user', torch.int64)
        items = stack_data_from_batch(batched_inputs, 'item', torch.int64)
        scores = stack_data_from_batch(batched_inputs, 'score', torch.float32)
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        users_bias= self.user_bias(users).reshape([-1])
        items_bias= self.item_bias(items).reshape([-1])
        
        preds = (users_emb * items_emb).sum(dim=-1) + users_bias + items_bias
        #loss  = (torch.abs((preds - scores)) * sample_weight).mean()
        loss  = (torch.abs((preds - scores) ** 2)).mean()
        preds = (users_emb * items_emb).sum(dim=-1)
        loss  = ((preds - scores)**2).sum() + self.l2_norm(users, items) * self.l2_norm_alpha
        if self.training:
            return {
                "loss_mse": loss,
            }
        else : 
            self.steps += 1
            self.sum += preds.reshape([1]).cpu().item()
            #print (self.sum / self.steps)
            return {
                "score": preds.reshape([-1]).cpu().detach().tolist()
            }
