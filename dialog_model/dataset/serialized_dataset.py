import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dialog_model.dataset.serializer import open_data_file, read_dtype, read_lengths, read_offsets
from py_utils.torch_utils.length_sort_sampler import LengthSortSampler


class SerializedDataset(Dataset):
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir

        self._data_file = None

        self._offsets = read_offsets(self._dataset_dir)
        self._lengths = read_lengths(self._dataset_dir)
        self._dtype = read_dtype(self._dataset_dir)

    @property
    def lengths(self):
        return self._lengths

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, i):
        self._data_file = self._data_file or open_data_file(self._dataset_dir)

        offset = self._offsets[i]
        length = self._lengths[i]

        token_ids = np.empty(length, dtype=self._dtype)
        self._data_file.seek(int(offset * self._dtype.itemsize))
        self._data_file.readinto(token_ids)

        return token_ids


def get_dataloader(dataset_dir, batch_size, num_workers, sort_chunk_size, samples_offset, data_shuffle_seed,
                   is_distributed, pad_token_id, end_of_speaker_1_token_id, end_of_speaker_2_token_id):
    dataset = SerializedDataset(dataset_dir)

    sampler = LengthSortSampler(
        lengths=dataset.lengths,
        sort_chunk_size=sort_chunk_size,
        samples_offset=samples_offset,
        data_shuffle_seed=data_shuffle_seed,
        is_distributed=is_distributed)

    collate = Collate(
        pad_token_id=pad_token_id,
        end_of_speaker_1_token_id=end_of_speaker_1_token_id,
        end_of_speaker_2_token_id=end_of_speaker_2_token_id)

    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate)

    return dataloader


class Collate:
    _LM_LOSS_IGNORE_LABEL = -100

    def __init__(self, pad_token_id, end_of_speaker_1_token_id, end_of_speaker_2_token_id, device=None):
        self._pad_token_id = pad_token_id
        self._end_of_speaker_1_token_id = end_of_speaker_1_token_id
        self._end_of_speaker_2_token_id = end_of_speaker_2_token_id
        self._device = device

    def __call__(self, samples):
        max_len = max(len(sample) for sample in samples)
        token_ids = np.empty((len(samples), max_len))
        token_type_ids = np.zeros_like(token_ids)
        lm_labels = np.empty_like(token_ids)
        token_ids.fill(self._pad_token_id)
        lm_labels.fill(self._LM_LOSS_IGNORE_LABEL)

        for i, sample in enumerate(samples):
            sample = np.array(sample)
            token_ids[i, :len(sample)] = sample
            lm_labels[i, :len(sample)] = self._construct_lm_labels(sample)
            token_type_ids[i, :len(sample)] = self._construct_token_type_ids(sample)

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        lm_labels = torch.tensor(lm_labels, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        if self._device is not None:
            token_ids = token_ids.to(self._device)
            token_type_ids = token_type_ids.to(self._device)
            lm_labels = lm_labels.to(self._device) if lm_labels is not None else lm_labels

        return token_ids, token_type_ids, lm_labels

    def _construct_token_type_ids(self, token_ids):
        token_type_ids = np.zeros_like(token_ids, dtype=token_ids.dtype)
        current_speaker = None
        prev_speaker = None
        for i, token_id in enumerate(token_ids[::-1]):
            if token_id == self._end_of_speaker_1_token_id:
                current_speaker = 0
            elif token_id == self._end_of_speaker_2_token_id:
                current_speaker = 1

            if current_speaker is not None and prev_speaker is None:
                token_type_ids[-i:] = abs(current_speaker - 1)

            if current_speaker is not None:
                token_type_ids[-(i + 1)] = current_speaker

            prev_speaker = current_speaker

        return token_type_ids

    def _construct_lm_labels(self, token_ids):
        mask = (token_ids == self._end_of_speaker_1_token_id) | (token_ids == self._end_of_speaker_2_token_id)
        # Cast to int32 from uint16 to add lm loss ignore label (which equals -100):
        lm_labels = token_ids.copy().astype(np.int32)
        first_end_of_speaker_index = mask.argmax()
        lm_labels[:first_end_of_speaker_index + 1] = self._LM_LOSS_IGNORE_LABEL

        return lm_labels
