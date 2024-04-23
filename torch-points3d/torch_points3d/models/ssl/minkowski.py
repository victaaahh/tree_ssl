from torch_points3d.models.ssl.base import VICRegBase
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from torch_geometric.data import Batch

class MinkowskiVICReg(VICRegBase):
    def __init__(self, opt, model_type, dataset, modules):
        super().__init__(opt, model_type, dataset, modules)
        self.encoder = initialize_minkowski_unet(model_name=opt.encoder,
                                                 in_channels=dataset.feature_dimension,
                                                 out_channels=opt.representation_D,
                                                 D=opt.D,
                                                 **opt.encoder_options)
        
        if opt.expander_activation != "relu":
            raise NotImplementedError("Only 'relu' is supported right now")

        layers = []
        for i in range(len(opt.expander_layers)-1):
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(opt.expander_layers[i]))
            layers.append(nn.Linear(opt.expander_layers[i], opt.expander_layers[i+1]))

        self.expander = nn.Sequential(nn.Linear(opt.representation_D, opt.expander_layers[0]), *layers)

    def set_input(self, data, device):
        # unpack data from dataset and apply preprocessing

        # self.batch_idx = data.batch.squeeze()
        
        data = data.to_data_list()

        X1 = Batch.from_data_list(data[::2])
        X2 = Batch.from_data_list(data[1::2])

        coords1 = torch.cat([X1.batch.unsqueeze(-1).int(), X1.coords.int()], -1)
        coords2 = torch.cat([X2.batch.unsqueeze(-1).int(), X2.coords.int()], -1)

        self.input1 = ME.SparseTensor(features=X1.x, coordinates=coords1, device=device)
        self.input2 = ME.SparseTensor(features=X2.x, coordinates=coords2, device=device)

    def forward(self, *args, **kwargs):
        Y1 = self.encoder(self.input1)
        Y2 = self.encoder(self.input2)
        
        Z1 = self.expander(Y1.F)
        Z2 = self.expander(Y2.F)

        # Should we split loss into multiple losses and use the inbuilt scaling functionality?
        # Then perhaps we can monitor them individually
        self.loss = self.compute_vicreg_loss(Z1, Z2, self.loss_scaling, self.loss_eps, self.loss_gamma)
 