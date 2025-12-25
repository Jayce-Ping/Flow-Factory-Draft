# src/flow_factory/data_utils/sampler.py
import math
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

class DistributedKRepeatSampler(Sampler):
    """
    """
    def __init__(self, dataset : Dataset, batch_size : int, group_size : int, unique_sample_num : int, num_replicas : int, rank : int, seed : int = 0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = group_size                # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas, process num, gpu num
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        self.m = unique_sample_num                    # `Least` number of unique sample per epoch
        
        if unique_sample_num > len(self.dataset):
            raise ValueError(f"`unique_sample_num` ({unique_sample_num}) must be <= dataset size ({len(self.dataset)}).")
        
        # Compute the number of samples for each batch iteration
        self.sample_num_per_iteration = self.num_replicas * self.batch_size
        step = self.sample_num_per_iteration // math.gcd(self.k, self.sample_num_per_iteration)
        new_m = (self.m + step - 1) // step * step  # Round up m to be multiple of step
        if new_m != self.m:
            logger.warning(f"Adjusted `unique_sample_num` from {self.m} to {new_m} to make sure `unique_sample_num`*`group_size` is multiple of `batch_size`*`num_replicas` for even distribution.")
            self.m = new_m
        
        self.num_batches_per_epoch = (self.m * self.k) // self.sample_num_per_iteration

        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples, less if dataset is smaller than m
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate m*k total samples.
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            for i in range(self.num_batches_per_epoch):
                # Offset for current iteration
                offset = i * self.sample_num_per_iteration
                # Compute start and end indices for current replica
                start = offset + self.rank * self.batch_size
                end = start + self.batch_size
                yield shuffled_samples[start:end]

            # Increment epoch for next iteration
            self.epoch += 1

    def set_epoch(self, epoch : int):
        self.epoch = epoch  # Used to synchronize random state across epochs