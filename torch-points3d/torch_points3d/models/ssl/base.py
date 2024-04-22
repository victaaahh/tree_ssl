from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.model_interface import InstanceTrackerInterface

import torch
import torch.nn.functional as F
import torch.nn as nn

class VICRegBase(BaseModel, InstanceTrackerInterface):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.loss_scaling = opt.loss_scaling

        self.loss_names = ["vicreg_loss"]
        
        self.encoder = None
        
        # Define expander in base class?
        self.expander = None
    
    def forward(self, *args, **kwargs):
        assert self.encoder is not None and self.expander is not None

        Z1 = self.expander(self.encoder(self.input1))
        Z2 = self.expander(self.encoder(self.input2))
        
        # Should we split loss into multiple losses and use the inbuilt scaling functionality?
        # Then perhaps we can monitor them individually
        self.vicreg_loss = self.compute_vicreg_loss(Z1, Z2, self.loss_scaling)
    
    def compute_vicreg_loss(self, Z1, Z2, scaling, eps=0.0001, gamma=1):
        # How should I get information on dimension D and batch size N? From config or from measuring?
        
        # Batch size and vec size:
        N, D = Z1.shape

        inv_loss = F.mse_loss(Z1, Z2)
        
        std_Z1 = torch.sqrt(torch.var(Z1, dim=0) + eps)
        std_Z2 = torch.sqrt(torch.var(Z2, dim=0) + eps)
        var_loss = torch.mean(F.relu(gamma - std_Z1)) / 2 + torch.mean(F.relu(gamma - std_Z2)) / 2
        
        #Z1 = Z1 - torch.mean(Z1, dim=0)
        #Z2 = Z2 - torch.mean(Z2, dim=0)
        
        cov_Z1 = (Z1.T @ Z1) / (N - 1)
        cov_Z2 = (Z2.T @ Z2) / (N - 1)
        cov_Z1_no_diag = cov_Z1[~torch.eye(D)]
        cov_Z2_no_diag = cov_Z2[~torch.eye(D)]

        cov_loss = cov_Z1_no_diag.pow(2).sum() / D + \
                   cov_Z2_no_diag.pow(2).sum() / D

        loss = scaling["invariance"] * inv_loss + scaling["variance"] * var_loss + scaling["covariance"] * cov_loss
        return loss