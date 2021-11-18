import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

""" A Dataset is effected by the following hyperparameters
    EPOCH / BATCH_SIZE / DATASETS
"""

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
    ),
    DATASETS=dict(
        ROOT="/home/data/dataset/rec_debias",
        TRAIN=("selectionbias_coat_clean", ),
        WITH_FEATURE=True,
        CLEAN_NUM=1000,
    ),
    DATALOADER=dict(
        NUM_WORKERS=0, 
        SAMPLER_TRAIN="DistributedGroupSamplerTimeSeed",
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_EPOCH=7, 
            WARMUP_ITERS=0,
        ),
        IMS_PER_BATCH=32,
        IMS_PER_DEVICE=32,
        OPTIMIZER=dict(
            BASE_LR=0.01,
            MOMENTUM=0.90,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=1e-4,
            GAMMA=0.3,
        ),
    ),
    SEED=16925062,
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
                TRAIN=("selectionbias_coat_train", ),
                WITH_FEATURE=True,
            ), 
            SOLVER=dict(
                LR_SCHEDULER=dict(
                    MAX_EPOCH=4,
                ),  
                OPTIMIZER=dict(
                    BASE_LR=0.02,
                    MOMENTUM=0.90,
                    WEIGHT_DECAY=2e-4,
                    WEIGHT_DECAY_NORM=1e-4,
                    GAMMA=0.3,
                ),
            ),
            OUTPUT_DIR=osp.join(
                '/home/data/Output',
                'mf-gan-gen',
            ),
        ))


#backbone_var_list = [ 'backbone-var-' + str(i) for i in range(10) ]
backbone_var_list = []
class GANConfig(BaseClassificationConfig):
    def __init__(self):
        super(GANConfig, self).__init__()
        self._register_configuration(dict(
            GAN=dict(
                MAX_ITER=5000, # max iteration time
                START_ITER=0, # set it to zero
                EPOCH_DIS=1 , # the training epoch of Discriminator during a iteration
                EPOCH_GEN=1 , # the training epoch of Generator during a iteration
                RESUME=False, # load the generator and discriminator
                HISTOGRAM_INTERVAL=50000, # interval for sending histogram, but will cost a mount of time
                FAKE_DATA_METHOD='sample',  # 'sample' | 'topk'
                SAMPLE_WEIGHT_PATH='/home/data/GAN/gan-based/cache/weights.pkl',  
            ),
            OUTPUT_DIR=osp.join(
                '/home/data/Output',
                'mf-gan',
            ),
            VISDOM=dict(
                HOST="192.168.1.1", 
                PORT="8082", 
                TURN_ON=True,
                ENV_PREFIX='mf-gan',
                KEY_LIST=['KLDIV']
            ), 
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
