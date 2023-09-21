'''
'''

import pathlib
from warnings import warn
from natsort import natsorted

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
        if normalize == "z-score":
            mean = torch.mean(features, dim=(0,2), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,2), keepdim=True))

            features = (features-mean)/stdv

            self.denormalize = lambda f: stdv*f+mean

            print("\nUsing z-score normalization")

        elif normalize == "0:1":
            min = torch.amin(features, dim=(0,2), keepdim=True)
            max = torch.amax(features, dim=(0,2), keepdim=True)

            features = (features-min)/(max-min)

            self.denormalize = lambda f: (max-min)*f+min

            print("\nUsing [0,1] min-max normalization")

        elif normalize == "-1:1":
            min = torch.amin(features, dim=(0,2), keepdim=True)
            max = torch.amax(features, dim=(0,2), keepdim=True)

            features = -1+2*(features-min)/(max-min)

            self.denormalize = lambda f: (max.to(f.device)-min.to(f.device))*(f+1)/2 + min.to(f.device)

            print("\nUsing [-1,1] min-max normalization")

        elif normalize == "z-score-clip":
            mean = torch.mean(features, dim=(0,2), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,2), keepdim=True))

            features = (features-mean)/stdv

            max = torch.amax(torch.abs(features), dim=(0,2), keepdim=True)

            features = features/max

            self.denormalize = lambda f: stdv.to(f.device)*f*max.to(f.device) + mean.to(f.device)

            print("\nUsing clipped z-score normalization")

        elif normalize == "z-score-1:1":
            mean = torch.mean(features, dim=(0,2), keepdim=True)

            features -= mean

            min = torch.amin(features, dim=(0,2), keepdim=True)
            max = torch.amax(torch.abs(features), dim=(0,2), keepdim=True)

            features = -1+2*(features-min)/(max-min)

            self.denormalize = lambda f: (max-min)*(f+1)/2 + min + mean

            print("\nUsing 0 mean [-1,1] normalization")

        else:
            self.denormalize = lambda f: f

        self.features = features

        return

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx,...]
    
    def getall(self, denormalize=True):
        if denormalize:
            return self.denormalize(self.features)
        else:
            return self.features

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
        self.feature_files = natsorted(pathlib.Path(features_path).glob("*"))

        if len(self.feature_files) == 0: raise Exception(f'No features have been found in: {features_path}')

        #compute normalization transform
        #NOTE: This could be done in one pass, but the variance is a bit weird because we are computing over two dimensions
        if normalize == "z-score":

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

            mean, stdv = mean, torch.sqrt(var)

            self.normalize = lambda f: (f-mean)/stdv
            self.denormalize = lambda f: stdv*f+mean

            print("\nUsing z-score normalization")

        elif normalize == "0:1" or normalize == "-1:1":
            num_samples = len(self)
            num_channels = len(self.channels)

            min = torch.zeros(num_channels, 1)
            max = torch.zeros(num_channels, 1)

            for i in range(num_samples):
                features = self._loaditem(i)

                min = torch.minimum(torch.amin(features, dim=1, keepdim=True), min)
                max = torch.maximum(torch.amax(features, dim=1, keepdim=True), max)

            if normalize == "0:1":
                self.normalize = lambda f: (f-min)/(max-min)
                self.denormalize = lambda f: (max-min)*f+min
                print("\nUsing [0,1] min-max normalization")
            else:
                self.normalize = lambda f: -1+2*(f-min)/(max-min)
                self.denormalize = lambda f: (max-min)*(f+1)/2 + min
                print("\nUsing [-1,1] min-max normalization")

        else:
            self.normalize = lambda f: f
            self.denormalize = lambda f: f

        return
    
    def _loaditem(self, idx):

        file = self.feature_files[idx]

        if file.suffix == ".npy":
            features = np.load(file).astype(np.float32)
        else:
            features = np.loadtxt(file).astype(np.float32)

        features = torch.from_numpy(features)

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
    
    def getall(self, denormalize=True):
        if denormalize:
            return torch.stack([self._loaditem(i) for i in range(self.__len__())])
        else:
            return torch.stack([self.__getitem__(i) for i in range(self.__len__())])

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
