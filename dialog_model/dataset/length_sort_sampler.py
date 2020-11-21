from itertools import chain

import numpy as np
from more_itertools import chunked
from torch.distributed import get_rank, get_world_size
from torch.utils.data import Sampler


class LengthSortSampler(Sampler):
    def __init__(
            self,
            lengths,
            sort_chunk_size,
            samples_offset=0,
            shuffle_with_seed=42,
            is_distributed=True
    ):
        super().__init__(None)

        self._full_len = len(lengths)
        self._current_offset = samples_offset % self._full_len
        self._rank, self._world_size = _get_rank_and_world_size(is_distributed)

        self._len = _count_worker_dataset_size(total_size=self._full_len, world_size=self._world_size, rank=self._rank)

        inds = np.argsort(lengths)
        inds_chunks = list(chunked(inds, sort_chunk_size * self._world_size))
        largest_chunk = inds_chunks.pop()
        np.random.RandomState(seed=shuffle_with_seed).shuffle(inds_chunks)
        self._inds = list(chain(largest_chunk, *inds_chunks))

    def __iter__(self):
        start = self._current_offset + self._rank
        for i in range(start, self._full_len, self._world_size):
            yield self._inds[i]

        # Reset offset to 0, so the next iteration will start from the beginning.
        self._current_offset = 0

    def __len__(self):
        return self._len


def _get_rank_and_world_size(is_distributed):
    """Get rank and world size for a current gpu worker. If `is_distributed` is False, returns (0, 1) tuple."""
    if is_distributed:
        rank = get_rank()
        world_size = get_world_size()
    else:
        rank, world_size = 0, 1

    return rank, world_size


def _count_worker_dataset_size(total_size, world_size, rank):
    """Calculates number of samples for a single worker process."""
    assert rank < world_size, 'Rank must be lower than world_size.'
    whole_number = total_size // world_size
    remainder = (total_size % world_size) > rank
    return whole_number + remainder
