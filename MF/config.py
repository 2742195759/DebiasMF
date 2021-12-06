import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        BPR=dict(
            DIM=40,
            MAX_NUM_USER=16000, 
            MAX_NUM_ITEM=3000,
            L2_NORM=0.03,
        ),
    ),
    DATASETS=dict(
        ROOT="/home/data/dataset/rec_debias",
        TRAIN=("autodebias_coat_test", ),
        TEST=("autodebias_coat_clean", ),
    ),
    TEST=dict(
        SORT_BY="MSE",
        EVAL_PERIOD=20,
    ),
    DATALOADER=dict(NUM_WORKERS=0, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEP=None,
            MAX_EPOCH=500,
            WARMUP_ITERS=30,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.01,
            MOMENTUM=0.9,
            WEIGHT_DECAY=0.000,
            WEIGHT_DECAY_NORM=0e-4,
        ),
        CHECKPOINT_PERIOD=30,
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=64,
    ),
    INPUT=dict(
        AUG=dict(
        )
    ),
    OUTPUT_DIR=osp.join(
        '/cvpods_output/',
        'MF-Raw',
    ),
    VISDOM=dict(
        HOST="192.168.1.1", 
        PORT="8082", 
        TURN_ON=False,
        ENV_PREFIX='demo',
        KEY_LIST=['mse'],
    )
)


class CustomerConfig(BaseClassificationConfig):
    def __init__(self):
        super(CustomerConfig, self).__init__()
        self._register_configuration(_config_dict)

config = CustomerConfig()
