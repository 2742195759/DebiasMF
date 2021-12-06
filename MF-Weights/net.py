import torch
import torch.nn as nn
from cvpods.utils.xklib import stack_data_from_batch, Once
import numpy as np
import logging
from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone import build_resnet_backbone


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
        self.user_bias= torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_USER, 1)
        self.item_bias= torch.nn.Embedding(
                    cfg.MODEL.BPR.MAX_NUM_ITEM, 1)

        weights_path = cfg.MODEL.SAMPLE_WEIGHTS
        self.rate = cfg.MODEL.BPR.RATE
        self.once = Once(lambda x : print (x))
        assert weights_path, "invalid weights path"
        print("Loading from %s" % weights_path)
        self.weights = self.weight_process(np.loadtxt(weights_path))
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.init_embedding()
        self.to(self.device)

    def weight_process(self, weights):
        # do nothing. 
        return weights

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
        #print ('output:', batched_inputs)
        sample_weight = torch.as_tensor([ self.weights[b['user'], b['item']] for b in batched_inputs ]).float().cuda()
        users = stack_data_from_batch(batched_inputs, 'user', torch.int64)
        items = stack_data_from_batch(batched_inputs, 'item', torch.int64)
        scores = stack_data_from_batch(batched_inputs, 'score', torch.float32)
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        users_bias= self.user_bias(users).reshape([-1])
        items_bias= self.item_bias(items).reshape([-1])
        
        preds = (users_emb * items_emb).sum(dim=-1) + users_bias + items_bias
        loss  = (((preds - scores) ** 2) * sample_weight).mean()
        #loss  = (torch.abs((preds - scores) ** 2)).mean()
        if self.training:
            self.once(sample_weight)
            #assert ((sample_weight > 1e-5).all()), "Dataset is not the same"
            # TODO
            #sample_weight[(sample_weight < 1e-5)] = 1
            return {
                "loss_mse": loss,
                "weight_m": sample_weight.mean().detach().cpu(),
                "weight_v": sample_weight.var().detach().cpu(),
            }
        else : 
            #print ("preds:", preds)
            return {
                "score": preds.reshape([-1]).cpu().detach().tolist()
            }

def build_model(cfg):
    model = MF_Weights(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
