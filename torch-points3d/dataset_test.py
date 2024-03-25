from omegaconf import OmegaConf
import hydra
from torch_points3d.datasets.instance.las_dataset import LasDataset
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    #print(OmegaConf.to_yaml(cfg))
    _dataset: BaseDataset = instantiate_dataset(cfg.data)
    #print(_dataset.train_dataset[0])
    #print(len(_dataset.train_dataset))


if __name__ == '__main__':
    main()