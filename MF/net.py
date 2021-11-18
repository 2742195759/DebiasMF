import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone import build_resnet_backbone

from bpr import Bpr


def build_model(cfg):

    model = Bpr(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
