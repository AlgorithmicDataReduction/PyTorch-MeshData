'''
'''

import pathlib
from warnings import warn

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

################################################################################

def get_dataset(features_path, channels, channels_last=True, normalize=True):

    path = pathlib.Path(features_path)

    if path.is_file():
        return MeshTensorDataset(features_path, channels, channels_last, normalize)
    else:
        return MeshDataset(features_path, channels, channels_last, normalize)

################################################################################

'''
Torch dataset responsible for loading mesh features from a single sample file.
'''
class MeshTensorDataset(Dataset):
    def __init__(self,
            features_path,
            channels,
            channels_last=True,
            normalize=False
        ):
        super().__init__()

        try:
            features = torch.from_numpy(np.load(features_path).astype(np.float32))

        except FileNotFoundError:
            raise Exception(f'No features have been found in: {features_path}')

        except Exception as e:
            raise e

        assert features.dim() == 3, f"Features has {features.dim()} dimensions, but should only have 3"

        #extract channels and move to channels first
        if channels_last:
            features = torch.movedim(features, -1, 1)

        features = features[:,channels,:]

        #normalize
        if normalize == "z_score":
            mean = torch.mean(features, dim=(0,2), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,2), keepdim=True))

            features = (features-mean)/stdv

        elif normalize == "max":
            features = features/(torch.amax(torch.abs(features), dim=(0,2), keepdim=True))

        self.features = features

        return

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx,...]

################################################################################

'''
Torch dataset responsible for loading mesh features from multiple sample files.

NOTE: Do we want to support normalization
'''
class MeshDataset(Dataset):
    def __init__(self,
            features_path,
            channels,
            channels_last=True,
            normalize=False
        ):
        super().__init__()

        #set attributes
        self.channels = channels
        self.channels_last = channels_last

        #get feature files
        self.feature_files = sorted(pathlib.Path(features_path).glob("*.npy"))

        if len(self.feature_files) == 0: raise Exception(f'No features have been found in: {features_path}')

        #compute normalization transform
        #NOTE: This could be done in one pass, but the variance is a bit weird because we are computing over two dimensions
        if normalize == "z_score":

            num_samples = len(self)
            num_channels = len(self.channels)

            mean = torch.zeros(num_channels, 1)
            var = torch.zeros(num_channels, 1)
            
            for i in range(num_samples):
                features = self._loaditem(i)

                mean += torch.mean(features, dim=1, keepdim=True)/num_samples

            for i in range(num_samples):
                features = self._loaditem(i)

                var += torch.sum((features-mean)**2, dim=1, keepdim=True)/(num_samples*features.shape[1]-1)

            self.mean, self.std = mean, torch.sqrt(var)

            self.normalize = lambda x: (x-self.mean)/self.var

        elif normalize == "max":
            num_samples = len(self)
            num_channels = len(self.channels)

            max = torch.zeros(num_channels, 1)

            for i in range(num_samples):
                features = self._loaditem(i)

                max = torch.maximum(torch.amax(torch.abs(features), dim=1, keepdim=True), max)

            self.max = max

            self.normalize = lambda x: x/self.max

        else:
            self.normalize = lambda x: x

        return
    
    def _loaditem(self, idx):

        features = torch.from_numpy(np.load(self.feature_files[idx]).astype(np.float32))

        assert features.dim() == 2, f"Features has {features.dim()} dimensions, but should only have 2" 

        #extract channels
        if self.channels_last:
            features = features.transpose(0, 1)

        features = features[self.channels,:]

        return features

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):

        features = self._loaditem(idx)

        features = self.normalize(features)

        return features

################################################################################

'''
PyTorch iterable dataset.
'''
class MeshIterableDataset(IterableDataset):
    def __init__():
        super().__init__()

        raise NotImplementedError("Iterable mesh dataset is not yet supported.")

    def __iter__(self):
        pass
