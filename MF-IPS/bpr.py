# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from cvpods.utils.xklib import stack_data_from_batch
import numpy as np
from compute_ips import bayes_method

class MF_IPS(nn.Module):
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
        super(MF_IPS, self).__init__()
        
        self.user_embedding = torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_USER, cfg.MODEL.BPR.DIM)
        self.item_embedding = torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_ITEM, cfg.MODEL.BPR.DIM)
        self.p_ips = bayes_method() 
        self.with_ips = cfg.MODEL.BPR.WITH_IPS
        self.p_ips_inv = [ 1.0 / _ for _ in self.p_ips ]
        #self.p_ips_inv = [ _ / sum(self.p_ips_inv) for _ in self.p_ips_inv ]
        """ for more robust training. """
        print ("Inv average IPS:", self.p_ips_inv)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Input:
        Output:
        """
        #import pdb
        #pdb.set_trace() 
        #print ('output:', batched_inputs)
        if not self.with_ips: 
            ips_inv_tensor = torch.as_tensor([ self.p_ips_inv[int(b['score'] - 1)] for b in batched_inputs ]).cuda().float() # (B,) 
        else : 
            ips_inv_tensor = 1.0 / stack_data_from_batch(batched_inputs, 'propensity', torch.float32)  # ground-truth IPS
            #ips_inv_tensor = ips_inv_tensor / ips_inv_tensor.sum()
        users = stack_data_from_batch(batched_inputs, 'user', torch.int64)
        items = stack_data_from_batch(batched_inputs, 'item', torch.int64)
        scores = stack_data_from_batch(batched_inputs, 'score', torch.float32)
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        preds = (users_emb * items_emb).sum(dim=-1)
        loss  = (((preds - scores)**2) * ips_inv_tensor).mean()

        if self.training:
            return {
                "loss_mse": loss,
            }
        else : 
            #print ("preds:", preds)
            return {
                "score": preds.reshape([-1]).cpu().detach().tolist()
            }
