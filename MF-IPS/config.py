import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        BPR=dict(
            DIM=10,
            MAX_NUM_USER=3000, 
            MAX_NUM_ITEM=3000,
            WITH_IPS=False,
        ),
    ),
    DATASETS=dict(
        ROOT="/home/data/dataset/rec_debias",
        TRAIN=("selectionbias_coat_train", ),
        TEST=("selectionbias_coat_test", ),
    ),
    TEST=dict(
        SORT_BY="mse",
        EVAL_PERIOD=20,
    ),
    DATALOADER=dict(NUM_WORKERS=0, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEP=None,
            MAX_EPOCH=500,
            WARMUP_ITERS=100,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.001,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-2,
            WEIGHT_DECAY_NORM=1e-4,
        ),
        CHECKPOINT_PERIOD=30,
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=128,
    ),
    INPUT=dict(
        AUG=dict(
            #TRAIN_PIPELINES=[
            #    ("RepeatList", dict(transforms=[
            #        ("Torch_RRC", transforms.RandomResizedCrop(224, scale=(0.2, 1.))),
            #        ("Torch_RG", transforms.RandomGrayscale(p=0.2)),
            #        ("Torch_CJ", transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)),
            #        ("Torch_RHF", transforms.RandomHorizontalFlip()),
            #    ], repeat_times=2)),
            #],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/cvpods_output/',
        'MF-IPS',
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
