from torch_points3d.models.ssl.base import VICRegBase
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet

class MinkowskiVICReg(VICRegBase):
    def __init__(self, opt, dataset):
        super().__init__(opt)
        self.model = initialize_minkowski_unet(opt.model_name, dataset.feature_dimension, )
    
    def forward(self, *args, **kwargs):
        pass