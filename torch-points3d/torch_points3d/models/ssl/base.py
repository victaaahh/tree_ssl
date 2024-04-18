from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.model_interface import InstanceTrackerInterface


class VICRegBase(BaseModel, InstanceTrackerInterface):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.loss_scaling = opt.loss_scaling

        self.loss_names = ["vicreg_loss"]
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        
        self.encoder = None
        self.expander = None
    
    def forward(self, *args, **kwargs):
        assert self.encoder is not None and self.expander is not None

        Y1 = self.encoder(self.input1)
        Y2 = self.encoder(self.input2)
        
        Z1 = self.expander(Y1)
        Z2 = self.expander(Y2)
        
        self.loss = self.vicreg_loss(Z1, Z2, self.loss_scaling)
    
    def vicreg_loss(self, Z1, Z2, scaling):
        pass