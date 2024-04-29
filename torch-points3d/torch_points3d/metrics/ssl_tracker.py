from typing import Dict, Any

from torch_points3d.metrics.base_tracker import BaseTracker

class SSLTracker(BaseTracker):
    def __init__(self, stage: str, wandb_log: bool, use_tensorboard: bool):
        super().__init__(stage, wandb_log, use_tensorboard)
        
    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        metrics = self.get_loss()
        if self._stage == "train" and self._finalised:
            pass # Compute AGB metrics and append to metrics dict
        return metrics