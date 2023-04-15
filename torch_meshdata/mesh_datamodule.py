'''
'''

import os
from warnings import warn

import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

from .mesh_loader import MeshLoader
from .mesh_dataset import MeshDataset

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
            features_path,
            spatial_dim,
            num_points,
            batch_size,
            channels,
            quad_map = None,
            quad_args = {},
            normalize = True,
            split = 0.8,
            shuffle = False,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True,
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

        self.train, self.val, self.test, self.predict = None, None, None, None

        return

    @staticmethod
    def add_args(parent_parser):

        parser = parent_parser.add_argument_group("MeshDataModule")

        parser.add_argument('--data_dir', type=str)

        return parent_parser

    @property
    def input_shape(self):
        return (1, len(self.channels), self.num_points)

    def setup(self, stage=None):

        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load dataset
            train_val = MeshDataset(self.features_path, self.channels, normalize=self.normalize)

            train_size = round(self.split*len(train_val))
            val_size = len(train_val) - train_size

            self.train, self.val = random_split(train_val, [train_size, val_size])

        if (stage == "test" or stage is None) and self.test is None:
            #load dataset
            self.test = MeshDataset(self.features_path, self.channels, normalize=self.normalize)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load dataset
            self.predict = MeshDataset(self.features_path, self.channels, normalize=self.normalize)

        if stage not in ["fit", "test", "predict", None]:
            raise ValueError("Stage must be one of fit, test, predict")

        return

    def train_dataloader(self):
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            shuffle=self.shuffle,
                            pin_memory=self.pin_memory,
                            persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
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

        #load mesh file
        if self.mesh_file != None:
            self.points, elements = MeshLoader(self.mesh_file).load_mesh()

            assert self.num_points == self.points.shape[0], f"Expected number of points ({self.num_points}) does not match actual number ({self.points.shape[0]})"
            assert self.spatial_dim == self.points.shape[1], f"Expected spatial dimension ({self.spatial_dim}) does not match actual number ({self.points.shape[1]})"

            if self.quad_map != None:
                weights = getattr(quadrature, self.quad_map)(self.points, self.num_points, **self.quad_args)
            else:
                weights = None

        #no mesh file, so quad_map must be specified
        else:
            if self.quad_map != None:
                self.points, weights = getattr(quadrature, self.quad_map)(self.spatial_dim, self.num_points, **self.quad_args)
            else:
                raise ValueError("Quadrature map must be specified when no points file is provided")


        return self.points, weights, elements
