# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from cvpods.utils.xklib import stack_data_from_batch
import numpy as np
from compute_ips import bayes_method

class MF_Weights(nn.Module):
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
        super(MF_Weights, self).__init__()
        
        self.user_embedding = torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_USER, cfg.MODEL.BPR.DIM)
        self.item_embedding = torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_ITEM, cfg.MODEL.BPR.DIM)

        weights_path = cfg.MODEL.SAMPLE_WEIGHTS
        assert weights_path, "invalid weights path"
        self.weights = np.loadtxt(weights_path)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)


    def forward(self, batched_inputs):
        """
        Input:
        Output:
        """
        #print ('output:', batched_inputs)
        sample_weight = torch.as_tensor([ self.weights[b['user'], b['item']] for b in batched_inputs ]).float().cuda()
        users = stack_data_from_batch(batched_inputs, 'user', torch.int64)
        items = stack_data_from_batch(batched_inputs, 'item', torch.int64)
        scores = stack_data_from_batch(batched_inputs, 'score', torch.float32)
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        preds = (users_emb * items_emb).sum(dim=-1)
        loss  = (((preds - scores)**2) * sample_weight).mean()
        if self.training:
            assert not ((sample_weight == 0).any()), "Dataset is not the same"
            return {
                "loss_mse": loss,
            }
        else : 
            #print ("preds:", preds)
            return {
                "score": preds.reshape([-1]).cpu().detach().tolist()
            }
