'''
'''

from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

import pathlib

from .mesh_loader import MeshLoader
from .mesh_dataset import get_dataset
from .mesh_sampler import PartitionSampler

'''
PT Lightning data module for unstructured point cloud data, possibly with an
associated mesh and quadrature.

Input:
    data_dir: data directory
    spatial_dim: spatial dimension of data
    num_points: number of points in data
    batch_size: batch size
    channels: which data channels to use
    quad_map: map to calculate quadrature
    normalize: whether or not to normalize the data
    split: percentage of data to use in training
    shuffle: whether or not to shuffle samples
    num_workers: number of data loading processes
    persistent_workers: whether or not to maintain data loading processes
    pin_memory: whether or not to pin data loading memory
'''
class MeshDataModule(pl.LightningDataModule):

    def __init__(self,*,
            mesh_file,
            feature_paths,
            spatial_dim,
            num_points,
            batch_size,
            channels,
            data_root = "./",
            weight_map = None,
            weight_args = {},
            normalize = True,
            split = 0.8,
            shuffle = False,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True,
            partition_sampler = False,
            batch_sampler = None,
            sampler_args = {}
        ):
        super().__init__()

        #channels
        if isinstance(channels, list):
            assert len(channels) != 0
        elif isinstance(channels, int):
            channels = [i for i in range(channels)]
        else:
            raise ValueError("Channels must be a list or an integer")

        args = locals()
        args.pop('self')

        for key, value in args.items():
            setattr(self, key, value)

        #join paths
        if not isinstance(self.feature_paths, list):
            self.feature_paths = [self.feature_paths]

        self.mesh_file = pathlib.Path(data_root).joinpath(self.mesh_file)
        self.feature_paths = [pathlib.Path(data_root).joinpath(path) for path in self.feature_paths]
        
        self.train, self.val, self.test, self.predict = None, None, None, None

        return

    @staticmethod
    def add_args(parent_parser):

        parser = parent_parser.add_argument_group("MeshDataModule")

        parser.add_argument('--data_root', type=str)

        return parent_parser

    @property
    def input_shape(self):
        return (1, len(self.channels), self.num_points)

    def setup(self, stage=None):

        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load dataset
            train_val = get_dataset(self.feature_paths, self.channels, normalize=self.normalize)

            train_size = round(self.split*len(train_val))
            val_size = len(train_val) - train_size

            self.train, self.val = random_split(train_val, [train_size, val_size])

        if (stage == "test" or stage is None) and self.test is None:
            #load dataset
            self.test = get_dataset(self.feature_paths, self.channels, normalize=self.normalize)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load dataset
            self.predict = get_dataset(self.feature_paths, self.channels, normalize=self.normalize)

        if stage not in ["fit", "test", "predict", None]:
            raise ValueError("Stage must be one of fit, test, predict")

        return

    def train_dataloader(self):
        if self.partition_sampler:
            sampler = PartitionSampler(self.train, shuffle=self.shuffle)

            self.shuffle = False
            batch_sampler = None
        else:
            sampler = None

            if self.batch_sampler is not None:
                batch_sampler = self.batch_sampler(self.train, **self.sampler_args)
                self.shuffle = False
            else:
                batch_sampler = None

        return DataLoader(self.train,
                            sampler=sampler,
                            batch_sampler=batch_sampler,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            shuffle=self.shuffle,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        if self.partition_sampler:
            sampler = PartitionSampler(self.val)
        else:
            sampler = None

        return DataLoader(self.val,
                            sampler=sampler,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def teardown(self, stage=None):
        return

    '''
    Load mesh from mesh.hdf5 file in data directory.
    '''
    def load_mesh(self):

        self.points, elements = MeshLoader(self.mesh_file).load_mesh()

        assert self.num_points == self.points.shape[0], f"Expected number of points ({self.num_points}) does not match actual number ({self.points.shape[0]})"
        assert self.spatial_dim == self.points.shape[1], f"Expected spatial dimension ({self.spatial_dim}) does not match actual number ({self.points.shape[1]})"

        if self.weight_map != None:
            weights = self.weight_map(self.points, self.num_points, **self.weight_args)
        else:
            weights = None

        return self.points, weights, elements
