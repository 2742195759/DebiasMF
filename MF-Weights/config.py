import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        SAMPLE_WEIGHTS="/home/data/DebiasMF/MF-GAN/cache/weights.ascii", 
        BPR=dict(
            DIM=60,
            MAX_NUM_USER=3000, 
            MAX_NUM_ITEM=3000,
        ),
    ),
    DATASETS=dict(
        ROOT="/home/data/dataset/rec_debias",
        TRAIN=("selectionbias_coat_train", ),
        TEST=("selectionbias_coat_test", ),
        SEED=2021,
    ),
    TEST=dict(
        SORT_BY="mse",
        EVAL_PERIOD=200,
    ),
    DATALOADER=dict(NUM_WORKERS=0, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEP=None,
            MAX_EPOCH=800,
            WARMUP_ITERS=30,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.1,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-3,
            WEIGHT_DECAY_NORM=1e-4,
        ),
        CHECKPOINT_PERIOD=30,
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
