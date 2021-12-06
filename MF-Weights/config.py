import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        SAMPLE_WEIGHTS="/home/data/DebiasMF/MF-GAN/cache/weights.ascii", 
        BPR=dict(
            DIM=20,
            MAX_NUM_USER=20000, 
            MAX_NUM_ITEM=20000,
            RATE=0.9
        ),
    ),
    DATASETS=dict(
        ROOT="/home/data/dataset/rec_debias",
        TRAIN=("autodebias_coat_train", ),
        TEST=("autodebias_coat_test", ),
        SEED=2021,
    ),
    TEST=dict(
        SORT_BY="MSE",
        EVAL_PERIOD=5,
    ),
    DATALOADER=dict(NUM_WORKERS=0, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(70,90),
            MAX_EPOCH=20,
            WARMUP_ITERS=5,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.8,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=1e-4,
        ),
        CHECKPOINT_PERIOD=300,
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=128,
    ),
    INPUT=dict(
        AUG=dict(
        )
    ),
    OUTPUT_DIR=osp.join(
        '/cvpods_output/',
        'MF-Weights',
    ),
    VISDOM=dict(
        HOST="10.255.129.13", 
        PORT="8082", 
        TURN_ON=False,
        ENV_PREFIX='MF-Weights',
        KEY_LIST=['AUC'],
    ),
    SEED=2021,
)


class CustomerConfig(BaseClassificationConfig):
    def __init__(self):
        super(CustomerConfig, self).__init__()
        self._register_configuration(_config_dict)

config = CustomerConfig()
