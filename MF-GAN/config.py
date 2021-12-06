import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig
import numpy as np
import torch
import random



def parameter_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Args for MF-GAN:")

    np.random.seed(2021)
    torch.set_rng_state(torch.manual_seed(2021).get_state())
    random.seed(2021)

    parser.add_argument("--dataset",                   type=str,   default="coat",                          help="the dataset name")
    parser.add_argument("--clean_num",                 type=int,   default=350   ,                          help="the dataset name")
    parser.add_argument("--temp",                      type=float, default=0.6   ,                          help="the dataset name")
    parser.add_argument("--with_feature",              type=bool,  default=False,                           help="GAN with feature?")
    parser.add_argument("--dis_epoch",                 type=int,   default=100,                              help="GAN with feature?")
    parser.add_argument("--dis_lr",                    type=float, default=0.01,                             help="GAN with feature?")
    parser.add_argument("--dis_l2",                    type=float, default=1e-4,                            help="GAN with feature?")
    parser.add_argument("--gen_epoch",                 type=int,   default=10,                              help="GAN with feature?")
    parser.add_argument("--gen_lr",                    type=float, default=0.05,                            help="GAN with feature?")
    parser.add_argument("--gen_l2",                    type=float, default=1e-4,                            help="GAN with feature?")
    parser.add_argument("--max_iter",                  type=int,   default=40,                              help="GAN with feature?")
    parser.add_argument("--fake_data_method",          type=str,   default="sample",                        help="GAN with feature?")
    parser.add_argument("--output_file",               type=str,   default="/home/data/DebiasMF/MF-GAN/cache/weights.ascii", help="GAN with feature?")
    parser.add_argument("--trainer",                   type=str,   default="exact",                        help="GAN with feature?")

    parser.add_argument("--weight_method",             type=str,   default="softmax",                       help="GAN with feature?")
    args = parser.parse_args()
    assert (args.weight_method in ['sigmoid', 'softmax', 'hard-sigmoid'])
    print ("The args: ", args)
    dataset2clean = {
        "yahoo": 1000,
        "coat" : 400, 
    }
    args.clean_num = dataset2clean[args.dataset]
    return args

args = parameter_parser() # the args passed by shell

""" 
    A Dataset is effected by the following hyperparameters
    EPOCH / BATCH_SIZE / DATASETS
"""

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
    ),
    DATASETS=dict(
        ROOT="/home/data/dataset/rec_debias",
        TRAIN=("autodebias_%s_clean" % args.dataset, ),
        WITH_FEATURE=args.with_feature,
        CLEAN_NUM=args.clean_num,
    ),
    DATALOADER=dict(
        NUM_WORKERS=0, 
        SAMPLER_TRAIN="DistributedGroupSamplerTimeSeed" if args.trainer == "random" else "DistributedGroupSampler",
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_EPOCH=args.dis_epoch, 
            WARMUP_ITERS=0,
        ),
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=64,
        OPTIMIZER=dict(
            BASE_LR=args.dis_lr,
            MOMENTUM=0.90,
            WEIGHT_DECAY=args.dis_l2,
            WEIGHT_DECAY_NORM=1e-4,
            GAMMA=0.3,
        ),
    ),
    SEED=2021,
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
            ],
            TEST_PIPELINES= [
            ], 
        )
    ),
    OUTPUT_DIR=osp.join(
        '/home/data/Output',
        'mf-gan-dis',
    ),
)

class CleanConfig(BaseClassificationConfig):
    def __init__(self):
        super(CleanConfig, self).__init__()
        self._register_configuration(_config_dict)

class NoiseConfig(CleanConfig):
    def __init__(self):
        super(NoiseConfig, self).__init__()
        self._register_configuration(dict(
            MODEL=dict(
                WEIGHTS="",
            ),
            DATASETS=dict(
                TRAIN=("autodebias_%s_train" % args.dataset, ),
                WITH_FEATURE=args.with_feature,
            ), 
            SOLVER=dict(
                LR_SCHEDULER=dict(
                    MAX_EPOCH=args.gen_epoch,
                ),  
                OPTIMIZER=dict(
                    BASE_LR=args.gen_lr,
                    MOMENTUM=0.90,
                    WEIGHT_DECAY=args.gen_l2,
                    WEIGHT_DECAY_NORM=1e-4,
                    GAMMA=0.3,
                ),
            ),
            OUTPUT_DIR=osp.join(
                '/home/data/Output',
                'mf-gan-gen',
            ),
            SEED=2021,
        ))


#backbone_var_list = [ 'backbone-var-' + str(i) for i in range(10) ]
backbone_var_list = []
class GANConfig(BaseClassificationConfig):
    def __init__(self):
        super(GANConfig, self).__init__()
        self._register_configuration(dict(
            GAN=dict(
                MAX_ITER=args.max_iter, # max iteration time
                START_ITER=0, # set it to zero
                RESUME=False, # load the generator and discriminator
                FAKE_DATA_METHOD=args.fake_data_method,  # 'sample' | 'topk'
                SAMPLE_WEIGHT_PATH=args.output_file,  
            ),
            OUTPUT_DIR=osp.join(
                '/home/data/Output',
                'mf-gan',
            ),
            VISDOM=dict(
                HOST="192.168.1.1", 
                PORT="8082", 
                TURN_ON=False,
                ENV_PREFIX='mf-gan',
                KEY_LIST=['KLDIV']
            ), 
            SEED=2021,
        ))

clean_cfg = CleanConfig()
noise_cfg = NoiseConfig()
global_cfg = GANConfig()

#model = "debug"
#model = "reproduct"
model = "train"

if model == 'debug': 
    print ("Debug Mode")
    global_cfg.GAN.RESUME=True
    global_cfg.VISDOM.TURN_ON=False
elif model == 'reproduct': 
    print ("Reproduct Mode")
    global_cfg.GAN.RESUME=True
    global_cfg.VISDOM.TURN_ON=False
else:
    #global_cfg.GAN.RESUME=True
    print ("Train Mode")
    pass

import logging

class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return False
logging.getLogger("cvpods").addFilter(NoParsingFilter())
#logging.getLogger("cvpods").addFilter(NoParsingFilter())
