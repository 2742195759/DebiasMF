import sys
import os
import os.path as osp
import tqdm
import collections
import cvpods
import copy
import numpy as np
from cvpods.engine.base_runner import RunnerBase
from cvpods.checkpoint import Checkpointer
from cvpods.data import build_test_loader, build_train_loader
from cvpods.utils import (
    CommonMetricPrinter, JSONWriter, PathManager,
    TensorboardXWriter, collect_env_info, comm,
    seed_all_rng, setup_logger, VisdomWriter
)
from cvpods.engine import DefaultRunner, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import ClassificationEvaluator
from cvpods.utils import comm
from config import noise_cfg as config
import cvpods.model_zoo as model_zoo
import logging

def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Support Args:")

    parser.add_argument("--data_path",                 type=str,   default="./data/Amazon_Instant_Video/",  help="data path")
    parser.add_argument("--epoch_number",              type=int,   default=40,                              help="number of training epochs")
    parser.add_argument("--learning_rate",             type=float, default=0.01,                            help="learning rate")
    parser.add_argument("--intervener_soft",           type=bool,  default=False,                           help="the regular item of the MF model")

    return parser.parse_args()

def get_prob_from_dataloader(loader):
    scores = []
    for data in tqdm.tqdm(loader):
        for d in data: 
            scores.append(int(d['score']))
    counter = collections.Counter(scores)
    l = [ counter[i+1] for i in range(5) ]
    l = [ i * 1.0 / sum(l) for i in l ]
    return np.array(l)

def correct_prob_from_dataloader(loader, weight_path, reciprocal=True):
    scores = [0] * 5
    weight = np.loadtxt(weight_path)
    #print (weight.shape)
    for data in tqdm.tqdm(loader):
        for d in data: 
            w = weight[d['user'], d['item']]
            assert w > 0.0, "different dataset"
            if reciprocal: w = 1.0 / w
            scores[int(d['score'])-1] += w
    scores = [ i * 1.0 / sum(scores) for i in scores ]
    return np.array(scores)

def cal_kldiv(arr1, arr2):
    #print (arr1)
    #print (arr2)
    return (np.log(arr1 / arr2) * arr1).sum()

def main(echo=True):
    config_clean = copy.deepcopy(config)
    config_clean.DATASETS.TRAIN=("selectionbias_coat_clean", )

    config_noise= copy.deepcopy(config)
    config_noise.DATASETS.TRAIN=("selectionbias_coat_train", )

    clean_loader = build_train_loader(config_clean)
    noise_loader = build_train_loader(config_noise)

    clean_prob = get_prob_from_dataloader(clean_loader)
    noise_prob = get_prob_from_dataloader(noise_loader)
    gan_correct_prob = correct_prob_from_dataloader(noise_loader, "/home/data/GAN/MF-GAN/cache/weights.ascii")
    ips_correct_prob = correct_prob_from_dataloader(noise_loader, "/home/data/dataset/rec_debias/coat/propensities.ascii")
    if echo : 
        print ("Gan Correct Noise : ", gan_correct_prob)
        print ("Ips Correct Noise : ", ips_correct_prob)
        print ("\n")
        #print ("Self KLDiv : ", cal_kldiv(clean_prob, clean_prob), "\n")
        print ("test-noise KLDiv : ", cal_kldiv(clean_prob, noise_prob), "\n")
        print ("test-gan   KLDiv : ", cal_kldiv(clean_prob, gan_correct_prob), "\n")
        print ("test-ips   KLDiv : ", cal_kldiv(clean_prob, ips_correct_prob), "\n")
        
    print (gan_correct_prob)
    print (clean_prob)
    return cal_kldiv(clean_prob, gan_correct_prob)

if __name__ == "__main__":
    args = parameter_parser()
    main()

