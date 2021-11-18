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
import copy
import collections

def bayes_method():
    user_num, item_num = 290, 300

    config_clean = copy.deepcopy(config)
    config_clean.DATASETS.TRAIN=("selectionbias_coat_clean", )
    clean_loader = build_train_loader(config_clean)
    noise_loader = build_train_loader(config)
    
    def statistic_score(loader):
        scores = []
        density = 0.0
        for data in tqdm.tqdm(loader):
            for d in data: 
                scores.append(int(d['score']))
                density += 1
        counter = collections.Counter(scores)
        l = [ counter[i+1] for i in range(5) ]
        l = [ i * 1.0 / sum(l) for i in l ]
        return l, density * 1.0 / (user_num * item_num)

    p_r, _ = statistic_score(clean_loader)
    p_r_observed, p_observed = statistic_score(noise_loader)
    p_ips = [ p_r_observed[i] * p_observed / p_r[i] for i in range(5) ]
    return p_ips

if __name__ == "__main__":
    p_ips = bayes_method()
    print (p_ips)
