import math

import torch.distributed as dist
from torch.utils.data import Sampler

def is_main_process() -> bool:
    """
    Returns True if:
      - torch.distributed is not available or not initialized, i.e. single-process
      - OR if initialized, current process rank == 0
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class DistributedEvalSampler(Sampler):
    """
    Distributed sampler for evaluation without padding or duplicating samples.

    Each rank receives a disjoint shard of indices. The union across all ranks
    covers the dataset exactly once.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.total_size = len(self.dataset)

    def __iter__(self):
        if self.total_size == 0:
            return iter([])

        shard_size = int(math.ceil(float(self.total_size) / float(self.num_replicas)))
        start = self.rank * shard_size
        end = min(start + shard_size, self.total_size)
        return iter(range(start, end))

    def __len__(self):
        if self.total_size == 0:
            return 0

        shard_size = int(math.ceil(float(self.total_size) / float(self.num_replicas)))
        start = self.rank * shard_size
        end = min(start + shard_size, self.total_size)
        return max(0, end - start)
