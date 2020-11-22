from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dialog_model.data_structures import DialogModelInput
from dialog_model.dataset.length_sort_sampler import LengthSortSampler
from dialog_model.dataset.serialization import read_index, open_data_file


class SerializedDataset(Dataset):
    def __init__(self, dataset_dir):
        dataset_dir = Path(dataset_dir)

        self._data_file = open_data_file(dataset_dir)

        self._offsets, self._lengths, self._dtype = read_index(dataset_dir)

    @property
    def lengths(self):
        return self._lengths

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, i):
        offset = self._offsets[i]
        length = self._lengths[i]

        token_ids = np.empty(length, dtype=self._dtype)
        self._data_file.seek(int(offset * self._dtype.itemsize))
        self._data_file.readinto(token_ids)

        return token_ids


def get_dataloader(
        dataset_dir,
        batch_size,
        num_workers,
        sort_chunk_size,
        samples_offset,
        data_shuffle_seed,
        is_distributed,
        pad_token_id,
        end_of_prefix_token_id):
    dataset = SerializedDataset(dataset_dir)

    sampler = LengthSortSampler(
        lengths=dataset.lengths,
        sort_chunk_size=sort_chunk_size,
        samples_offset=samples_offset,
        data_shuffle_seed=data_shuffle_seed,
        is_distributed=is_distributed
    )

    collate = Collate(pad_token_id=pad_token_id, end_of_prefix_token_id=end_of_prefix_token_id)

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate)

    return dataloader


class Collate:
    _LM_LOSS_IGNORE_LABEL = -100

    def __init__(self, pad_token_id, end_of_prefix_token_id):
        self._pad_token_id = pad_token_id
        self._end_of_prefix_token_id = end_of_prefix_token_id

    def __call__(self, samples) -> DialogModelInput:
        max_len = max(len(sample) for sample in samples)
        token_ids = np.empty((len(samples), max_len))
        lm_labels = np.empty_like(token_ids)
        token_ids.fill(self._pad_token_id)
        lm_labels.fill(self._LM_LOSS_IGNORE_LABEL)

        for i, sample in enumerate(samples):
            token_ids[i, :len(sample)] = sample
            lm_labels[i, :len(sample)] = sample

            prefix_end_pos = np.argmax(sample == self._end_of_prefix_token_id)
            lm_labels[i, :prefix_end_pos + 1] = self._LM_LOSS_IGNORE_LABEL

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        lm_labels = torch.tensor(lm_labels, dtype=torch.long)

        model_input = DialogModelInput(token_ids=token_ids, lm_labels=lm_labels, past=None)

        return model_input
