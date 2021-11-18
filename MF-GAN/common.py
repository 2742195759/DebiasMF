import logging
import torch
import torch.nn as nn
from cvpods.modeling.xkmodel import MLP, SoftmaxPredictor
from cvpods.utils.xklib import stack_data_from_batch
import torchvision.models as models
from cvpods import model_zoo
import cvpods

user_num = 2000
item_num = 3000
dim = 10

class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.output = {}
        self.emb_s = nn.Embedding(6, dim)
        self.emb_u = nn.Embedding(user_num, dim)
        self.emb_i = nn.Embedding(item_num, dim)
        self.log_sigmoid = nn.LogSigmoid() 
        self.log_softmax = nn.LogSoftmax(dim=-1) 
        self.sigmoid = nn.Sigmoid() 
        self.with_feature = True
        self.softmax = nn.Softmax(dim=-1) 
        if self.with_feature : 
            self.mlp = cvpods.modeling.xkmodel.MLP([3 * dim + 10 * 2, 3, 1], nn.ReLU, False)
        else :
            self.mlp = cvpods.modeling.xkmodel.MLP([3 * dim, 3, 1], nn.ReLU, False)
        self.loss_name = "ModelBase"
        self.loss_crit = nn.BCELoss()
        self.user_feat_weight = nn.Linear(14, 10)
        self.item_feat_weight = nn.Linear(33, 10)
        self.device = "cuda"
        self.to(self.device)

    def loss(self, logits, batched_inputs):
        """ 
            Hookable 
        """
        raise RuntimeError("Not Implemented ModelBase.loss function")

    def cal_logits(self, batched_inputs, user, item, score):
        """
            output = (B, )
        """
        #__import__('pdb').set_trace()
        bs = len(batched_inputs)
        x = torch.cat([self.emb_s(score), self.emb_i(item), self.emb_u(user)], dim=-1)
        if self.with_feature: 
            user_feat = stack_data_from_batch(batched_inputs, 'user_feat', torch.float32).reshape([bs, -1])
            item_feat = stack_data_from_batch(batched_inputs, 'item_feat', torch.float32).reshape([bs, -1])
            tmp_x = torch.cat([self.user_feat_weight(user_feat), self.item_feat_weight(item_feat)], dim=-1)
            x = torch.cat([tmp_x, x], dim=-1)
            #print (x.shape)

        y = self.mlp(x)
        return y.reshape([-1])

    def forward(self, batched_inputs):
        """
        Input:
        Output:
        """
        n_batch = len(batched_inputs)
        user = stack_data_from_batch(batched_inputs, 'user', torch.long).reshape([-1])
        item = stack_data_from_batch(batched_inputs, 'item', torch.long).reshape([-1]) # (N, )
        score = stack_data_from_batch(batched_inputs, 'score', torch.long).reshape([-1]) # (N, )
        logits = self.cal_logits(batched_inputs, user, item, score)
        if self.training:
            loss = self.loss(logits, batched_inputs)
            return {
                self.loss_name: loss.mean(),
                **self.output
            }
        else : 
            return self.default_eval(logits)

    def default_eval(self, logits):
        scores = self.sigmoid(logits)
        return scores.detach().cpu().reshape([-1, 1])

class Generator(ModelBase):
    def __init__(self):
        """ Generator
        """
        super(Generator, self).__init__()
        self.loss_name = "generator loss"

    def loss(self, logits, batched_inputs):
        logits_copy = logits.reshape([-1]) / 0.3
        assert 'dis_gt' in batched_inputs[0], "Use Discrimination to generate gt for generator"
        dis_gt = stack_data_from_batch(batched_inputs, 'dis_gt', torch.float32).reshape([-1])
        dis_gt = torch.max(torch.as_tensor(0.0).cuda(), dis_gt - 1e-4)
        reward = - torch.log(1 - dis_gt).reshape([-1]) # monotonic increasing
        prob_x = self.softmax(logits_copy.reshape([-1]))
        mean_reward = prob_x * reward
        tmp = (logits.reshape([-1]).topk(10)[0].float()).max()
        self.output['logits max'] = tmp.detach().cpu()
        reward = reward - mean_reward
        loss = - self.log_softmax(logits_copy.reshape([-1]))
        self.output['probs max'] = prob_x.max().detach().cpu()
        self.output['avg reward'] = reward.mean().detach().cpu()
        proxy_loss = (reward * loss).mean()
        return proxy_loss

    def default_eval(self, logits):
        return logits.detach().cpu().reshape([-1, 1])

class Discriminator(ModelBase):
    def __init__(self):
        """ Generator
        """
        super(Discriminator, self).__init__()
        self.loss_name = "discri loss"

    def loss(self, logits, batched_inputs):
        n_batch = len(batched_inputs)
        assert 'is_clean' in batched_inputs[0], "please insert the ground truth from the generator"
        is_clean = stack_data_from_batch(batched_inputs, 'is_clean', torch.float32).reshape([-1])
        is_clean_bool = is_clean.bool()
        is_clean = is_clean.float()
        logits = logits.reshape([-1])  #(N, )
        probs = self.sigmoid(logits)
        self.output['Clean Losses'] = self.loss_crit(probs[0:n_batch//2], is_clean[0:n_batch//2]).detach().cpu()
        self.output['Noise Losses'] = self.loss_crit(probs[n_batch//2:], is_clean[n_batch//2:]).detach().cpu()
        self.output['accuracy'] = ((probs < 0.5) ^ (is_clean_bool)).float().mean().detach().cpu()
        loss = self.loss_crit(probs, is_clean)
        return loss
