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
            normalize=True
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
        if normalize:
            mean = torch.mean(features, dim=(0,2), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,2), keepdim=True))

            features = (features-mean)/stdv

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
            normalize=True
        ):
        super().__init__()

        #set attributes
        self.channels = channels
        self.channels_last = channels_last
        self.normalize = normalize

        #get feature files
        self.feature_files = sorted(pathlib.Path(features_path).glob("*.npy"))

        if len(self.feature_files) == 0: raise Exception(f'No features have been found in: {features_path}')

        #normalize
        if self.normalize:
            warn("Normalization not currently supported for MeshDataset.")

        return

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):

        features = torch.from_numpy(np.load(self.feature_files[idx]).astype(np.float32))

        assert features.dim() == 2, f"Features has {features.dim()} dimensions, but should only have 2" 

        #extract channels
        if self.channels_last:
            features = features.transpose(0, 1)

        features = features[self.channels,:]

        if self.normalize:
            pass
            # features = (features-self.mean)/(self.std*self.max)

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
