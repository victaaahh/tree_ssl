from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.model_interface import InstanceTrackerInterface


class SSLBase(BaseModel, InstanceTrackerInterface):
    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        
        self.mode = opt.mode
    
    def set_input(self, input, device):
        pass
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class VICRegBase(SSLBase):
    def __init__(self, opt):
        super().__init__(opt)
        if self.mode not in ["finetune", "freeze"]: # What about the built in mode?
            self.loss_names.extend(["vicreg_loss"])
    
    def vicreg_loss():
        pass