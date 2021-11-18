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
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.sum = 0.0
        self.steps = 0.0
        self.to(self.device)


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
        preds = (users_emb * items_emb).sum(dim=-1)
        loss  = ((preds - scores)**2).mean()
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
