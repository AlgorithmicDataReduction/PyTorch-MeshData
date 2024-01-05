'''
'''

import pathlib
from warnings import warn
from natsort import natsorted

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

################################################################################

def get_dataset(feature_paths, channels, channels_last=True, normalize=True):

    #NOTE: This assumes all feature_paths are homogenous
    path_type = pathlib.Path(feature_paths[0])

    if path_type.is_file():
        return MeshTensorDataset(feature_paths, channels, channels_last, normalize)
    else:
        return MeshDataset(feature_paths, channels, channels_last, normalize)

################################################################################

'''
Torch dataset responsible for loading mesh features from a single sample file.
'''
class MeshTensorDataset(Dataset):
    def __init__(self,
            feature_paths,
            channels,
            channels_last=True,
            normalize=False
        ):
        super().__init__()

        #load all features
        features = []

        for path in feature_paths:
            try:
                f = torch.from_numpy(np.load(path).astype(np.float32))

                assert f.dim() == 3, f"Features has {f.dim()} dimensions, but should only have 3"

                #extract channels and move to channels first
                if channels_last:
                    f = torch.movedim(f, 2, 1)

                f = f[:,channels,:]

            except FileNotFoundError:
                raise Exception(f'Error loading features at {path}')

            except Exception as e:
                raise e

            features.append(f)

        if len(feature_paths) > 1:
            warn("Multiple feature paths assumes you are training on the same number of GPUs")
            assert dist.is_available(), "Torch Distributed must be available for multi-partition training"

            self.multi_partition = True
        else:
            self.multi_partition = False

        #compute partition indices
        self.partition_indices = [0]
        for i,f in enumerate(features):
            self.partition_indices.append(f.shape[2]+self.partition_indices[i])

        features = torch.cat(features, dim=2)

        #normalize
        #NOTE: I am normalizng all of the possible partitions together, but I could also normalize individually
        
        if normalize == "z-score-clip":
            mean = torch.mean(features, dim=(0,2), keepdim=True)
            stdv = torch.sqrt(torch.var(features, dim=(0,2), keepdim=True))

            features = (features-mean)/stdv

            max = torch.amax(torch.abs(features), dim=(0,2), keepdim=True)

            features = features/max

            self.denormalize = lambda f: stdv.to(f.device)*f*max.to(f.device) + mean.to(f.device)

            print("\nUsing clipped z-score normalization")

        elif normalize == "z-score":

            raise NotImplementedError("z-score normalization not implemented")

            #NOTE: Needs to be implemented with multiple parittions
            # mean = torch.mean(f_temp, dim=(0), keepdim=True).unsqueeze(2)
            # stdv = torch.sqrt(torch.var(f_temp, dim=(0), keepdim=True)).unsqueeze(2)

            # features = [(f-mean)/stdv for f in features]

            # self.denormalize = lambda f: stdv*f+mean

            # print("\nUsing z-score normalization")

        elif normalize == "0:1":
            
            raise NotImplementedError("0:1 normalization not implemented")

            #NOTE: Needs to be implemented with multiple parittions
            # min = torch.amin(f_temp, dim=(0), keepdim=True).unsqueeze(2)
            # max = torch.amax(f_temp, dim=(0), keepdim=True).unsqueeze(2)

            # features = [(f-min)/(max-min) for f in features]

            # self.denormalize = lambda f: (max-min)*f+min

            # print("\nUsing [0,1] min-max normalization")

        elif normalize == "-1:1":

            raise NotImplementedError("1:1 normalization not implemented")

            #NOTE: Needs to be implemented with multiple parittions
            # min = torch.amin(f_temp, dim=(0), keepdim=True).unsqueeze(2)
            # max = torch.amax(f_temp, dim=(0), keepdim=True).unsqueeze(2)

            # features = [-1+2*(f-min)/(max-min) for f in features]

            # self.denormalize = lambda f: (max.to(f.device)-min.to(f.device))*(f+1)/2 + min.to(f.device)

            # print("\nUsing [-1,1] min-max normalization")

        elif normalize == "z-score-1:1":

            raise NotImplementedError("z-score-1:1 normalization not implemented")

            #NOTE: Needs to be implemented with multiple parittions
            # mean = torch.mean(features, dim=(0), keepdim=True).unsqueeze(2)

            # features -= mean

            # min = torch.amin(features, dim=(0,2), keepdim=True)
            # max = torch.amax(torch.abs(features), dim=(0,2), keepdim=True)

            # features = [-1+2*(f-min)/(max-min) for f in features]

            # self.denormalize = lambda f: (max-min)*(f+1)/2 + min + mean

            # print("\nUsing 0 mean [-1,1] normalization")

        else:
            self.denormalize = lambda f: f

        self.features = features

        return

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        partition = 0 if not self.multi_partition else dist.get_rank()

        return self.features[idx,:,self.partition_indices[partition]:self.partition_indices[partition+1]]
    
    def getall(self, denormalize=True, partition=None):
        partition = 0 if partition == None else partition

        features = self.features[:,:,self.partition_indices[partition]:self.partition_indices[partition+1]]

        if denormalize:
            return self.denormalize(features)
        else:
            return features

################################################################################

'''
Torch dataset responsible for loading mesh features from multiple sample files.

NOTE: Do we want to support normalization
'''
class MeshDataset(Dataset):
    def __init__(self,
            feature_paths,
            channels,
            channels_last=True,
            normalize=False
        ):
        super().__init__()

        #NOTE: Not currently setup to handle multiple feature directories
        assert len(feature_paths)==1, "Multiple feature directories not supported"
        feature_paths = feature_paths[0]

        #set attributes
        self.channels = channels
        self.channels_last = channels_last

        #get feature files
        self.feature_files = natsorted(pathlib.Path(feature_paths).glob("*"))

        if len(self.feature_files) == 0: raise Exception(f'No features have been found in: {feature_paths}')

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
