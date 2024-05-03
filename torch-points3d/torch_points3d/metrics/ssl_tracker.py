from typing import Dict, Any

import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import wandb

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
        self.representations = wandb.Artifact(f"validation_representations_{wandb.run.id}", type="representations")
        
    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        metrics = self.get_loss()
        if self.AGB_R2_score:
            metrics["AGB_R2_score"] = self.AGB_R2_score
        return metrics
    
    def finalise(self, *args, **kwargs):
        self._finalised = True
        if self._stage == "val":
            # Compute AGB metrics and insert in metrics dict
            X = torch.cat(self.val_representation, dim=0).numpy()
            y = torch.cat(self.labels, dim=0).numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            score = reg.score(X_test, y_test)
            self.AGB_R2_score = score
            
            # dimensionality reduction logging of embeddings
            n_dim = 100
            pca = PCA(n_dim)
            self.representations.add(wandb.Table(columns=[f"D{i}" for i in range(n_dim)], data=pca.fit_transform(X)), "representations")
    
    def track(self, model: model_interface.TrackerInterface, **kwargs):
        super().track(model, **kwargs)
        if self._stage == "val":
            self.val_representation.append(model.get_output().cpu())
            self.labels.append(model.get_labels().cpu())
            
    def get_publish_metrics(self, epoch):
        """Publishes the current metrics to wandb and tensorboard
        Arguments:
            step: current epoch
        """
        metrics = self.get_metrics()

        return {
            "stage": self._stage,
            "epoch": epoch,
            "current_metrics": self._remove_stage_from_metric_keys(self._stage, metrics),
            "all_metrics": metrics
        }
    
    def publish_to_wandb(self, metrics, epoch):
        wandb.log_artifact(self.representations)
        super().publish_to_wandb(metrics, epoch)