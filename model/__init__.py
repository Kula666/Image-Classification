from .alexnet import *


def get_model(config):
    return globals()[config.architecture](config.num_classes).to(config.device)
