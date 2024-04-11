from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.model_interface import InstanceTrackerInterface


class SSLBase(BaseModel, InstanceTrackerInterface):
    def __init__(self):
        pass