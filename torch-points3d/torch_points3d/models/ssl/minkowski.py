from torch_points3d.models.ssl.base import VICRegBase
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet

import torch.nn as nn

class MinkowskiVICReg(VICRegBase):
    def __init__(self, opt, dataset):
        super().__init__(opt)
        self.encoder = initialize_minkowski_unet(opt.encoder_name, dataset.feature_dimension, opt.representation_D, D=opt.D)
        self.expander = nn.Sequential()

    def set_input(self, input, device):
        # unpack data from dataset and apply preprocessing
        self.input = ...
    