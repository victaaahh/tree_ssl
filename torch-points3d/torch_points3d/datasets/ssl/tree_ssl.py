import logging
from glob import glob
from pathlib import Path
from typing import Sized, Iterator, Optional

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from torch.utils.data import Sampler
from torch_geometric.data import Dataset, Data
from tqdm.auto import tqdm

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.models import model_interface

from torch_points3d.datasets.instance.las_dataset import read_pt

log = logging.getLogger(__name__)

class TreeSSL(Dataset):
    def __init__(
        self, root, raw_data_files, transform=None, xy_radius=15, feature_cols=[],
        min_pts: int = 500, min_high_vegetation: int = 100, pre_transform=None
    ):
        self.raw_data_files = raw_data_files
        self.xy_radius = xy_radius
        self.min_high_vegetation = min_high_vegetation
        self.feature_cols = feature_cols
        self.min_pts = min_pts
        super().__init__(root, transform, pre_transform, pre_filter=None)
    
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
            pos, classification, crs = read_pt(file, ["classification"] + self.feature_cols, ",") # delimiter is in case of csv, but not needed here

            if len(self.feature_cols) > 0:
                features = classification[:, 1:]
                classification = classification[:, 0]
            else:
                features = None

            # Making center points of hexagonal grid
            x_min, y_min, _ = np.min(pos, axis=0)
            x_max, y_max, _ = np.max(pos, axis=0)
            hex_grid_dist = np.sqrt(3*self.xy_radius*self.xy_radius)
            x_coords = np.arange(x_min+self.xy_radius, x_max-self.xy_radius, 1.5*self.xy_radius)
            y_coords = np.arange(y_min+hex_grid_dist/2, y_max-hex_grid_dist/2, hex_grid_dist)
            point_list = []
            for y in y_coords:
                for x_idx, x in enumerate(x_coords):
                    if x_idx % 2 == 0:
                        point_list.append(Point(x, y))
                    elif y_max - y >= hex_grid_dist:
                        point_list.append(Point(x, y+hex_grid_dist/2))
            
            labels = gpd.GeoDataFrame(geometry=point_list, crs=crs)
            
            # Getting points within xy_radius of the center points
            label_centers = np.stack([labels.geometry.x, labels.geometry.y], 1)

            kdtree = KDTree(pos[:,:2])
            label_point_idx = kdtree.query_radius(label_centers, self.xy_radius)
            
            # Looping over all the extracted point clouds,
            # filtering them and saving them to disk
            first_pt_file = file_idx
            for idx in range(len(label_point_idx)):
                # Checks for high_veg_count and min_points could be made into pre_filters instead
                high_veg_count = np.sum(classification[label_point_idx[idx]] == 5)
                if high_veg_count < self.min_high_vegetation:
                    labels.drop(idx, inplace=True)
                    continue
                pos_pt = pos[label_point_idx[idx]]
                if features is not None:
                    features_pt = features[label_point_idx[idx]]
                else:
                    features_pt = None
                if pos_pt.shape[0] < self.min_pts:
                    labels.drop(idx, inplace=True)
                    continue
                self.center_pos(pos_pt, label_centers[idx])
                data = self.convert_to_data_(pos_pt, features_pt)
                if data is None:
                    continue
                pt_file = Path(self.processed_dir) / "pt_files" / f"{file_idx}.pt"
                torch.save(data, pt_file)
                file_idx += 1

            # save label file with name corresponding to pt file names
            label_file = Path(self.processed_dir) / "label_files" / f"{first_pt_file}_{file_idx-1}.gpkg"
            labels.to_file(label_file, driver="GPKG")

        (Path(self.processed_dir) / "done.flag").touch()
        log.info("### Processing done!")
    
    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / "pt_files" / f"{idx}.pt")
        return data
    
    def len(self):
        files = glob(str(Path(self.processed_dir) / "pt_files" / "*.pt"))
        return len(files)

    def convert_to_data_(self, pos, features=None):
        '''Makes data object and applies pre_filter and pre_transform'''
        pos = torch.tensor(pos, dtype=torch.float32)

        if features is not None:
            features = torch.tensor(features, dtype=torch.float32)
        data = Data(x=features, pos=pos)

        if self.pre_transform is not None:
            data = self.pre_transform(data)
            if data.pos.shape[0] == 0:
                log.warning(f"Pre transform reduced sample to 0 points, skipping")
                return None

        return data

    def center_pos(self, pos, center_point):
        pos_center = np.amin(pos, axis=0, keepdims=True)
        pos_center[:, 0] = center_point[0]
        pos_center[:, 1] = center_point[1]
        # The centering is done in place
        pos -= pos_center
        
    @property
    def num_classes(self):
        return 0


class TreeSSLDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.raw_data_files = dataset_opt.raw_data_files
        self.xy_radius = dataset_opt.xy_radius
        self.feature_cols = dataset_opt.get("features", [])
        self.min_pts = dataset_opt.get("min_pts", 500)
        self.min_high_vegetation = dataset_opt.get("min_high_vegetation", 100)
        self.log_train_metrics = dataset_opt.get("log_train_metrics", True)

        self.train_dataset = TreeSSL(
            root=self._data_path, raw_data_files=self.raw_data_files, transform=self.train_transform,
            pre_transform=self.pre_transform, xy_radius=self.xy_radius,
            feature_cols=self.feature_cols, min_pts=self.min_pts, min_high_vegetation=self.min_high_vegetation
        )
        
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        return BaseTracker(stage="train", wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        
    def create_dataloaders(
        self,
        model: model_interface.DatasetInterface,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        precompute_multi_scale: bool):
        if batch_size % 2 != 0:
            raise ValueError("Only even batch sizes supported, since we need every point cloud twice.")
        if self.train_dataset:
            self.train_sampler = DoubleBatchSampler(self.train_dataset, batch_size)
            #if drop_last is False:
            #    log.warning("Cannot disable 'drop_last' with DoubleBatchSampler.")
        if not shuffle:
            log.warning("shuffle=False is unsupported.")
        super().create_dataloaders(model, batch_size, shuffle, drop_last, num_workers, precompute_multi_scale)
        

class DoubleBatchSampler(Sampler[int]):
    '''Samples each element randomly without replacement and yields each of them twice in
       a row. num_samples argument can be used to set the amount of elements to sample,
       each of which will then be sampled twice. The length is therefore num_samples*2.'''
    def __init__(self, data_source: Sized, batch_size, drop_last, num_samples: Optional[int] = None):
        super().__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        self._num_samples = num_samples
        self.drop_last = drop_last
        
    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
        
    def __len__(self) -> int:
        return self.num_samples * 2
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        left_over = 0
        for _ in range(self.num_samples // n):
            iterator = torch.randperm(n, generator=generator).tolist()
            iterator = np.array([[k, k] for k in iterator]).flatten().tolist()
            left_over += n % self.batch_size
            yield from iterator

        iterator = torch.randperm(n, generator=generator).tolist()[:self.num_samples % n] # Isnt this 0 when num_samples is not specified?
        iterator = np.array([[k, k] for k in iterator]).flatten().tolist()
        if self.drop_last:
            pass
            #iterator = iterator[]
        yield from iterator

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, drop_last={self.drop_last}, num_samples={self._num_samples})"