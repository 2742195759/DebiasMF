import logging#{{{
import os
import pickle as pkl
import sys
from collections import OrderedDict
import torch
import tqdm
from colorama import Fore, Style
import cvpods.data.transforms as T
from cvpods.engine.base_runner import RunnerBase
from cvpods.checkpoint import Checkpointer
from cvpods.data import build_test_loader, build_train_loader
from cvpods.evaluation import (
    DatasetEvaluator, inference_on_dataset,
    print_csv_format, verify_results
)
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import (
    CommonMetricPrinter, JSONWriter, PathManager,
    TensorboardXWriter, collect_env_info, comm,
    seed_all_rng, setup_logger, VisdomWriter
)
from cvpods.engine import DefaultRunner, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import ClassificationEvaluator
from cvpods.utils import comm
from cvpods.utils import EventStorage
import cvpods.model_zoo as model_zoo
from config import clean_cfg, noise_cfg, global_cfg, args
from sklearn.metrics import roc_auc_score
import copy
import sys
import numpy as np
#}}}

if True:  # "INIT"/*{{{*/
    """ for reuse, we set the 
    """
    clean_dataloader = build_train_loader(clean_cfg)
    noise_dataloader = build_train_loader(noise_cfg)#/*}}}*/

# main function{{{
def save_weight_logits(users, items, outputs):
    #user_num, item_num = 943, 1682
    dataset2meta = {
        'yahoo': [15400, 1000], 
        'coat': [290, 300],
    }
    user_num, item_num = dataset2meta[args.dataset]
    mat = np.zeros((user_num, item_num))
    for user, item, output in zip(users, items, outputs):
        mat[user, item] = output
    np.savetxt(global_cfg.GAN.SAMPLE_WEIGHT_PATH, mat)

from cvpods.utils import PltHistogram

def main(generator):
    with EventStorage() as storage:
        hist = PltHistogram()
        output = []
        temp = args.temp
        users = []
        items = []
        clean_size = noise_cfg.DATASETS.CLEAN_NUM
        generator.eval()
        for data in tqdm.tqdm(noise_dataloader): 
            for d in data : 
                users.append(d['user'])
                items.append(d['item'])
            output.append(generator(data).reshape([-1,1]))
        output = torch.cat(output, dim=0)
        users = torch.as_tensor(users).long()
        items = torch.as_tensor(items).long()
        #method = "hard-sigmoid"
        method = args.weight_method
        output = output.reshape([-1])
        if method == 'sigmoid':
            output = (output - output.mean()) / output.std()
            output = torch.nn.Sigmoid()(output)
            #import pdb
            #pdb.set_trace() 
            #assert (output > 1e-5).all()

        if method == 'softmax_normal':
            output = torch.nn.Softmax(dim=0)(output / temp)
            minn = output.min()
            maxx = output.max()
            print ("Range is : ", maxx - minn)
            output = (output - minn) / (maxx - minn) * 1
            assert (output <= 1).all()
            assert (output >= 0).all()
            print ("number is : ", (output > 0.6).sum())
            #output = (output) / output.mean()
        if method == 'softmax':
            #import pdb
            #pdb.set_trace() 
            output = torch.nn.Softmax(dim=0)(output / temp)
            mmean = output.mean()
            mstd = output.std()
            print ("mean and var is : ", mmean, mstd)
            output = (output - mmean) / (mstd) * 0.7
            output = output + 0.5  # make the mean to 0.5
            output[output>1] = 1
            output[output<0] = 0
            assert (output <= 1).all()
            assert (output >= 0).all()
            print ("number is : ", (output > 0.5).sum())
            #output = (output) / output.mean()
        if method == 'softmax_raw':
            #import pdb
            #pdb.set_trace() 
            output = torch.nn.Softmax(dim=0)(output / temp)

        if method == 'upbound':
            """ classify by ratio of 
            """
            bin_count = 70
            minn = output.min().item()
            maxx = output.max().item()

            def get_freq(input, bin_count, minn, maxx):
                output_arr = input.numpy()
                hist, _ = np.histogram(output_arr, bins=bin_count, range=(minn, maxx))
                return hist

            output_arr = output.numpy()
            interval = (maxx - minn) / (bin_count - 1)
            output_int = ((output_arr - minn) / interval).astype('int')
            output_weight = hist_clean[output_int] / (hist_noise[output_int] + hist_clean[output_int])
            output = torch.as_tensor(output_weight)

        if method == 'clip':
            ratio = 0.8
            length = len(output)
            top_k = int(length * ratio)
            indices = torch.topk(output.reshape([-1]), top_k)[1]
            output = torch.zeros_like(output)
            output[indices] = 1.0
            clean_ratio = labels_tmp.sum()*1.0 / len(labels_tmp)
            print ("Ratio", clean_ratio)
            print ("Clean / Tot", clean_ratio * ratio)

        if method == 'hard-sigmoid':
            clean_weight = 1.0
            noise_weight = 0.0
            output = (output - output.mean()) / output.std()
            output = torch.nn.Sigmoid()(output)

            clean_boolean = (output > 0.5)

            output[ clean_boolean] = clean_weight
            output[~clean_boolean] = noise_weight
        
        save_weight_logits(users, items, output.numpy())

if __name__ == "__main__":
    assert False, "Error"
#}}}
