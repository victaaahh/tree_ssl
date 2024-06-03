import logging

import torch
import torch.nn as nn

import MinkowskiEngine as ME

from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet
from torch_points3d.models.instance.minkowski import MinkowskiBaselineModel, SeparateLinear

log = logging.getLogger(__name__)


class MinkowskiDownstream(MinkowskiBaselineModel):
    def __init__(self, option, model_type, dataset, modules):
        super(MinkowskiBaselineModel, self).__init__(option, model_type, dataset, modules)

        load_ssl_model = bool(option.get("ssl_model_path", None))
        load_downstream_model = bool(option.get("downstream_model_path", None))
        assert load_ssl_model != load_downstream_model, "Only specify ssl_model_path or downstream_model_path"

        self.model = initialize_minkowski_unet(model_name=option.model_name,
                                                 in_channels=dataset.feature_dimension,
                                                 out_channels=dataset.num_classes,
                                                 D=option.D,
                                                 **option.kwargs)
        
        in_channel = self.model.final.linear.weight.shape[1]

        if load_ssl_model:
            log.info("Will use encoder weights from self-supervised model")
            self.model.final = nn.Identity()
            state_dict = torch.load(option.ssl_model_path)["models"][option.ssl_weight_name]
            state_dict = {key[8:]: val for key, val in state_dict.items() if key.startswith("encoder")}
            self.model.load_state_dict(state_dict)
        
        if option.kwargs.get("dropout", 0) > 0:
            self.model.glob_avg = nn.Sequential(
                ME.MinkowskiGlobalSumPooling(),
                ME.MinkowskiDropout(option.kwargs.dropout)
            )
        else:
            self.model.glob_avg = ME.MinkowskiGlobalSumPooling()
        
        self._supports_mixed = True
        self.model.final = SeparateLinear(in_channel, self.num_reg_classes)

        for m in self.model.final.linears:
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        if load_downstream_model:
            log.info("Will use weight from previous downstream_model")
            state_dict = torch.load(option.downstream_model_path)["models"][option.downstream_weight_name]
            state_dict = {key[6:]: val for key, val in state_dict.items() if key.startswith("model")}
            self.model.load_state_dict(state_dict)

        self.head_namespace = option.get("head_namespace", "final.linears")
        self.head_optim_settings = option.get("head_optim_settings", {})
        self.backbone_optim_settings = option.get("backbone_optim_settings", {})

        self.add_pos = option.get("add_pos", False)
        
        self.mode = option.mode
        if self.mode == "freeze":
            log.info("Training with frozen encoder")
            self.model.requires_grad_(False)
            self.model.eval()
            self.model.final.requires_grad_(True)
            self.model.final.train()
            self.enable_dropout_in_eval()
        elif self.mode == "finetune":
            log.info("Training without frozen encoder")
            pass
        else:
            raise ValueError("Only 'freeze' and 'finetune' modes are supported")