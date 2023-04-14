'''
'''

import os

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

################################################################################

'''
Torch dataset responsible for loading mesh features.

NOTE: We want this to work with multiple files in a directory as well.
NOTE: We should support more arbitrary transforms
'''
class MeshDataset(Dataset):
    def __init__(self,
            features_path,
            channels,
            channels_last=True,
            normalize=True
        ):
        super().__init__()

        try:
            features = torch.from_numpy(np.load(features_path).astype(np.float32))

            assert features.dim() == 3, f"Features has {features.dim()} dimensions, but should only have 3"

            #extract channels
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

        except FileNotFoundError:
            raise Exception(f'No features have been found in: {features_path}')

        except Exception as e:
            raise e

        return

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx,...]

################################################################################

class MeshIterableDataset(IterableDataset):
    def __init__():
        super().__init__()

        raise NotImplementedError("Iterable mesh dataset is not yet supported.")

    def __iter__(self):
        pass
