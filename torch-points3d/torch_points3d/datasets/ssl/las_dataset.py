import logging
import os
from collections import OrderedDict
from functools import partial
from glob import glob
from itertools import chain, product
from pathlib import Path
from typing import Sized, Iterator

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj
import scipy.stats as scstats
import torch
from omegaconf import OmegaConf
from plyfile import PlyData
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from torch.utils.data import Sampler, dataset
from torch_geometric.data import Dataset, Data
from tqdm.auto import tqdm

from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.metrics.instance_tracker import InstanceTracker
from torch_points3d.models import model_interface

from torch_points3d.datasets.instance.las_dataset import read_pt

log = logging.getLogger(__name__)

class LasSSL(Dataset):
    def __init__(
        self, root, raw_data_files, transform=None, pre_transform=None, pre_filter=None,
        xy_radius=15, feature_cols=[], min_pts: int = 0, min_high_vegetation = 0
    ):
        self.raw_data_files = raw_data_files
        self.xy_radius = xy_radius
        self.min_high_vegetation = min_high_vegetation
        self.feature_cols = feature_cols
        self.min_pts = min_pts
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return self.raw_data_files
    
    @property
    def processed_file_names(self):
        # Maybe implement better check
        return "done.flag"
    
    def process(self):
        log.info("### Processing starting!")

        (Path(self.processed_dir) / "pt_files").mkdir(exist_ok=True)
        (Path(self.processed_dir) / "label_files").mkdir(exist_ok=True)

        raw_files = []
        for f in self.raw_data_files:
            raw_files.extend(glob(str(Path(self.raw_dir) / f)))
        
        file_idx = 0
        for file in tqdm(raw_files):
            # add support for additional features?
            pos, features, crs = read_pt(file, ["classification"] + self.feature_cols, ",") # delimiter is in case of csv, but not needed here
            x_min, y_min, z_min = np.min(pos, axis=0)
            x_max, y_max, z_max = np.max(pos, axis=0)
            x_coords = np.arange(x_min+self.xy_radius, x_max-self.xy_radius, self.xy_radius*2) # Make hexagon grid
            y_coords = np.arange(y_min+self.xy_radius, y_max-self.xy_radius, self.xy_radius*2)
            point_list = []
            for y in y_coords:
                for x in x_coords:
                    point_list.append(Point(x, y))
            labels = gpd.GeoDataFrame(geometry=point_list, crs=crs)
            label_centers = np.stack([labels.geometry.x, labels.geometry.y], 1)

            kdtree = KDTree(pos[:,:2])
            label_point_idx = kdtree.query_radius(label_centers, self.xy_radius)
            
            first_file = file_idx
            for idx in range(len(label_point_idx)):
                uniq, counts = np.unique(features[label_point_idx[idx], 0], return_counts=True)
                index_of_5 = np.where(uniq==5)
                ratio_of_5 = counts[index_of_5]/np.sum(counts)
                if 5 not in uniq:
                    labels.drop(idx, inplace=True)
                    continue
                elif ratio_of_5 <= self.ratio_of_5_limit:
                    labels.drop(idx, inplace=True)
                    continue
                x = pos[label_point_idx[idx]]
                x_centered  = self.center_pos(x, label_centers[idx])
                data = self.convert_to_data_(x)
                pt_file = Path(self.processed_dir) / "pt_files" / f"{file_idx}.pt"
                torch.save(data, pt_file)
                file_idx += 1

            # save label file with name corresponding to pt file names
            label_file = Path(self.processed_dir) / "label_files" / f"{first_file}_{file_idx-1}.gpkg"
            labels.to_file(label_file, driver="GPKG")

        (Path(self.processed_dir) / "done.flag").touch()
        log.info("### Processing done!")
    
    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / "pt_files" / f"{idx}.pt")
        return data
    
    def len(self):
        files = glob(str(Path(self.processed_dir) / "pt_files" / "*.pt"))
        return len(files)

    def convert_to_data_(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        data = Data(pos=x)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
            if data.pos.shape[0] == 0:
                log.warning(f"Pre transform reduced sample to 0 points, skipping")
                return None

        return data

    def center_pos(self, x, center_point):
        x_center = np.amin(x, axis=0, keepdims=True)
        x_center[:, 0] = center_point[0]
        x_center[:, 1] = center_point[1]
        x -= x_center
        return x


class LasDatasetSSL(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.raw_data_files = dataset_opt.raw_data_files
        self.xy_radius = dataset_opt.xy_radius
        self.feature_cols = dataset_opt.get("features", None)
        self.min_pts = dataset_opt.get("min_pts", 500)
        self.ratio_of_5_limit = dataset_opt.get("ratio_of_5_limit", 0)
        self.log_train_metrics = dataset_opt.get("log_train_metrics", True)

        self.train_dataset = LasSSL(
            root=self._data_path, raw_data_files=self.raw_data_files, transform=self.train_transform,
            pre_transform=self.pre_transform, xy_radius=self.xy_radius,
            feature_cols=self.feature_cols, min_pts=self.min_pts, ratio_of_5_limit=self.ratio_of_5_limit
        )
        
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        return InstanceTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                               log_train_metrics=self.log_train_metrics)

# Add support for processing features.
# Add support for min points
# Turn check for high vegetation into a pre-filter?
# Change from ratio to min number of points in high vegetation check.
# Change to hdf5 data format. How can pytorch geometric load it into memory in chunks? How exactly is the training loop constructed? How are the batches? Can a PYG dataset be passed in as a chunk? Maybe it already loads in a bunch of files at a time using a chunk parameter. What is then to be gained by changing dataformat?
# Change to hexagon grid.