'''
'''

import torch
from torch.utils.data.distributed import DistributedSampler

'''
Custom distributed sampler for multiple mesh partitions.

Input:

'''
class PartitionSampler(DistributedSampler):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        if isinstance(dataset, torch.utils.data.dataset.Subset):
            self.partition_indices = self.dataset.dataset.partition_indices
        else:
            self.partition_indices = self.dataset.partition_indices

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            batch_indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        #return generator
        return ([idx, slice(None), slice(*self.partition_indices[self.rank:self.rank+2])] for idx in batch_indices)