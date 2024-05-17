import logging
from glob import glob
from pathlib import Path
from typing import Sized, Iterator, Optional, Iterable
import os

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from torch.utils.data import Sampler, ConcatDataset
from torch_geometric.data import Dataset, Data
from tqdm.auto import tqdm

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.ssl_tracker import SSLTracker
from torch_points3d.models import model_interface
from torch_points3d.datasets.instance.las_dataset import Las, read_pt

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
        # A better check should be implemented
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
                # Checks for high_veg_count and min_points should be made into pre_filters instead.
                # Then a warning would appear if they were changed.
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
        
        self.sampler_num_samples = dataset_opt.get("training_num_samples", None)

        self.train_dataset = TreeSSL(
            root=self._data_path, raw_data_files=self.raw_data_files, transform=self.train_transform,
            pre_transform=self.pre_transform, xy_radius=self.xy_radius,
            feature_cols=self.feature_cols, min_pts=self.min_pts, min_high_vegetation=self.min_high_vegetation
        )
        
        self.AGB_val = dataset_opt.get("AGB_validation", False)
        if self.AGB_val:
            AGB_path = os.path.join(self.dataset_opt.dataroot, self.dataset_opt.AGB_val_options.dataset_name)
            AGB_processed_folder = dataset_opt.AGB_val_options.get("processed_folder", "processed")
            AGB_areas_file = AGB_path / Path(AGB_processed_folder) / "areas.pt"
            if not AGB_areas_file.exists(): # Perhaps a better check is needed
                raise Exception("Please process AGB data beforehand")

            AGB_areas = torch.load(AGB_areas_file)

            AGB_train_data = Las(
                root=AGB_path, areas=AGB_areas, split="train",
                targets=self.dataset_opt.AGB_val_options.targets,
                feature_cols=[], feature_scaling_dict=None, stats=[],
                transform=self.val_transform, pre_transform=self.pre_transform,
                save_processed=True, processed_folder=AGB_processed_folder,
                in_memory=False, xy_radius=self.xy_radius, save_local_stats=False,
                min_pts_outer=self.min_pts, min_pts_inner=0,
            )

            AGB_val_data = Las(
                root=AGB_path, areas=AGB_areas, split="val",
                targets=self.dataset_opt.AGB_val_options.targets,
                feature_cols=[], feature_scaling_dict=None, stats=[],
                transform=self.val_transform, pre_transform=self.pre_transform,
                save_processed=True, processed_folder=AGB_processed_folder,
                in_memory=False, xy_radius=self.xy_radius, save_local_stats=False,
                min_pts_outer=self.min_pts, min_pts_inner=0,
            )
            
            class AGB_concat(ConcatDataset):
                def __init__(self, datasets: Iterable[Dataset]) -> None:
                    super().__init__(datasets)
                    self.has_labels = True
            
            self.val_dataset = AGB_concat([AGB_train_data, AGB_val_data])
        
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        return SSLTracker(self, stage="train", wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        
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
            self.train_sampler = DoubleBatchSampler(self.train_dataset, self.sampler_num_samples)
        if not shuffle:
            log.warning("shuffle=False is unsupported. Continuing with shuffle.")

        super().create_dataloaders(model, batch_size, shuffle, drop_last, num_workers, precompute_multi_scale)
        
        # Overwriting val_loader to give it half batch size
        conv_type = model.conv_type
        self._val_loader = self._dataloader(
                self.val_dataset,
                self.val_pre_batch_collate_transform,
                conv_type,
                precompute_multi_scale,
                batch_size=batch_size // 2,
                shuffle=False,
                num_workers=num_workers,
                sampler=self.val_sampler,
            )


class DoubleBatchSampler(Sampler[int]):
    '''Samples each element randomly without replacement and yields each of them twice in
       a row. num_samples argument can be used to set the amount of elements to sample,
       each of which will then be sampled twice. The length is therefore num_samples*2.'''
    def __init__(self, data_source: Sized, num_samples: Optional[int] = None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        
    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
        
    def __len__(self) -> int:
        return self.num_samples * 2
    
    def __iter__(self) -> Iterator[int]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        n = len(self.data_source)
        full_iters = self.num_samples // n
        left_over = self.num_samples % n
        for i in range(full_iters):
            iterator = torch.randperm(n, generator=generator).tolist()
            iterator = np.array([[k, k] for k in iterator]).flatten().tolist()
            yield from iterator

        iterator = torch.randperm(n, generator=generator).tolist()[:left_over]
        iterator = np.array([[k, k] for k in iterator]).flatten().tolist()
        yield from iterator

    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={self._num_samples})"