from typing import Dict, Any

import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.models import model_interface

class SSLTracker(BaseTracker):
    def __init__(self, stage: str, wandb_log: bool, use_tensorboard: bool):
        super().__init__(stage, wandb_log, use_tensorboard)
        
    def reset(self, stage="train"):
        super().reset(stage)
        self.val_representation = []
        self.labels = []
        self.AGB_R2_score = None
        
    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        metrics = self.get_loss()
        if self.AGB_R2_score:
            metrics["AGB_R2_score"] = self.AGB_R2_score

        return metrics
    
    def finalise(self, *args, **kwargs):
        self._finalised = True
        if self._stage == "train":
            # Compute AGB metrics and insert in metrics dict
            X = torch.cat(self.val_representation, dim=0).numpy()
            y = torch.cat(self.labels, dim=0).numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            score = reg.score(X_test, y_test)
            self.AGB_R2_score = score
    
    def track(self, model: model_interface.TrackerInterface, **kwargs):
        super().track(model, **kwargs)
        self.val_representation.append(model.get_output())
        self.labels.append(model.get_labels())