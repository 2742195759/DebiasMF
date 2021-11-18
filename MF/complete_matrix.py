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
from sklearn.metrics import roc_auc_score
import copy
import sys
import numpy as np #}}}

from config import config

def main():
    categorys = []
    user_num, item_num = 943, 1682
    train_loader = build_train_loader(config)
    model = model_zoo.get(
        ".",
        playground_path="/home/data/GAN/MF",
        custom_config=dict(
            MODEL=dict(
                WEIGHTS="/home/data/GAN/MF/log/model_save.pth",
            ),
        )
    )
    model.eval()
    matrix = np.zeros((user_num, item_num)).astype('float')
    for i in tqdm.tqdm(range(user_num)):
        for j in range(item_num):
            b = {
                'user': i+1, "item": j+1, "score": 0.0
            }
            try : 
                matrix[i,j] = model([b])['score'][0]
            except: 
                import pdb
                pdb.set_trace() 
                a = 1

    for data in tqdm.tqdm(train_loader): 
        for d in data:
            matrix[d['user']-1, d['item']-1] = d['score']
    np.save('./log/matrix.npy', matrix)
    
    score_list = []
    for data in tqdm.tqdm(train_loader): 
        for d in data:
            score_list.append(int(d['score']))
    import collections

    matrix = np.load('./log/matrix.npy')
    # from https://www.cs.toronto.edu/~zemel/documents/acmrec2009-MarlinZemel.pdf
    prob = [0.52 , 0.24 , 0.13 , 0.06 , 0.05]
    score =[1, 2, 3, 4, 5]
    all_int = user_num * item_num
    matrix = np.reshape(matrix, [-1])
    indices = np.argsort(matrix)
    start_pos = 0
    for s, p in zip(score, prob):
        length = int(p * all_int)
        tmp = indices[start_pos:start_pos + length]
        matrix[tmp] = s 
        start_pos += length 

    matrix = np.reshape(matrix, [user_num, item_num])
    def print_density(mat):
        ret = []
        for i in range(5):
            ret.append((mat == i+1).sum() / (mat!=0).sum())
        print ("density for score", ret)
        return ret

    print_density(matrix)
    np.save('./log/matrix_after.npy', matrix)

    def sample_clean(matrix, clean_num):
        mat = matrix.reshape([-1])
        clean_mat = np.zeros_like(mat)
        import random
        pool= set()
        while(len(pool) != clean_num):
            ind = random.choice(range(mat.size))
            if ind not in pool: 
                clean_mat[ind] = mat[ind]
                pool.add(ind)
        assert (clean_mat != 0).sum() == clean_num, "Error"
        np.save("./log/clean_matrix.npy", clean_mat)
        return clean_mat

    sample_clean(matrix, 10000)

    def sample_by_alpha(matrix, alpha):
        observed_rate = [1, 1, alpha, alpha ** 2, alpha ** 3]
        observed_rate.reverse()
        k = 0.05 / sum([ a * b for a, b in zip(observed_rate, prob)])
        observed_rate = [ k * _ for _ in observed_rate ]
        print ("Observed_rate:", observed_rate)
        print (observed_rate)
        def sample(mat, observed_rate):
            tmp = mat.reshape([-1])
            observed = np.zeros((mat.size,))
            for idx, rate in enumerate(observed_rate):
                ind = np.where(tmp == idx + 1)[0]
                ind = np.random.choice(ind, size=int(rate*(mat==idx+1).sum()), p=None)
                observed[ind] = idx + 1  # choise them
            return observed

        observed_matrix = sample(matrix, observed_rate).reshape([user_num, item_num])
        print ("density:", (observed_matrix != 0).sum() / observed_matrix.size)
        print_density(observed_matrix)

        np.save("./log/observe_matrix_%s" % alpha, observed_matrix)

    print("Ground-Truth: ", [0.06 , 0.11 , 0.27 , 0.35 , 0.21])
    
    for alpha in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        print ("Sample:", alpha)
        sample_by_alpha(matrix, alpha)
        print ("\n\n")



if __name__ == "__main__":
    main()
