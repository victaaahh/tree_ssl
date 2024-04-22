from torch_points3d.models.ssl.base import VICRegBase
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from torch_geometric.data import Batch

class MinkowskiVICReg(VICRegBase):
    def __init__(self, opt, dataset):
        super().__init__(opt)
        self.encoder = initialize_minkowski_unet(opt.encoder_name, dataset.feature_dimension, opt.representation_D, D=opt.D)
        
        # Add non-linearity and batch normalization
        self.expander = nn.Sequential(nn.Linear(opt.representation_D, opt.expander_dims[0]),
                                      *[nn.Linear(opt.expander_dims[i], opt.expander_dims[i+1]) for i in range(len(opt.expander_dims)-1)])

    def set_input(self, data, device):
        # unpack data from dataset and apply preprocessing

        # self.batch_idx = data.batch.squeeze()
        
        data = data.to_data_list()
        # Maybe theres a faster/better way than slicing
        X1 = Batch.from_data_list(data[::2])
        X2 = Batch.from_data_list(data[1::2])

        coords1 = torch.cat([X1.batch.unsqueeze(-1).int(), X1.coords.int()], -1)
        coords2 = torch.cat([X2.batch.unsqueeze(-1).int(), X2.coords.int()], -1)

        self.input1 = ME.SparseTensor(features=X1.x, coordinates=coords1, device=device)
        self.input2 = ME.SparseTensor(features=X2.x, coordinates=coords2, device=device)