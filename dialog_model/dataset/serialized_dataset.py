from collections import Counter
import random

import numpy as np
from py_utils.torch_utils.length_sort_sampler import LengthSortSampler
import torch
from torch.utils.data import DataLoader, Dataset

from dialog_model.dataset.serializer import get_response_length, open_data_file, \
    read_dtype, read_offsets, read_response_lengths, read_sample_lengths


class SerializedDataset(Dataset):
    def __init__(self, dataset_dir, distractor_p, end_of_speaker_1_token_id):
        self._dataset_dir = dataset_dir
        self._distractor_p = distractor_p
        self._end_of_speaker_1_token_id = end_of_speaker_1_token_id

        self._data_file = None

        self._offsets = read_offsets(self._dataset_dir)
        self._response_lengths = read_response_lengths(self._dataset_dir)
        self._sample_lengths = read_sample_lengths(self._dataset_dir)
        self._dtype = read_dtype(self._dataset_dir)

        self._argsort_of_response_lengths = np.argsort(self._response_lengths)

        counter_of_response_lengths = Counter(self._response_lengths)
        # Cumulative count of the same length or shorter responses:
        self._response_length_to_cumcount = {}
        cumcount = 0
        for response_length in sorted(counter_of_response_lengths.keys()):
            cumcount += counter_of_response_lengths[response_length]
            self._response_length_to_cumcount[response_length] = cumcount

    @property
    def sample_lengths(self):
        return self._sample_lengths

    def __len__(self):
        return len(self._sample_lengths)

    def __getitem__(self, i):
        token_ids = self._get_token_ids(i)
        length_before_distraction = len(token_ids)

        if self._distractor_p > random.random():
            token_ids = self._distract_token_ids(token_ids)
            is_distracted = 1

            length_after_distraction = len(token_ids)
            assert length_after_distraction <= length_before_distraction
        else:
            is_distracted = 0

        return token_ids, is_distracted

    def _get_token_ids(self, i):
        self._data_file = self._data_file or open_data_file(self._dataset_dir)

        offset = self._offsets[i]
        sample_length = self._sample_lengths[i]

        token_ids = np.empty(sample_length, dtype=self._dtype)
        self._data_file.seek(int(offset * self._dtype.itemsize))
        self._data_file.readinto(token_ids)

        return token_ids

    def _distract_token_ids(self, token_ids):
        response_length = get_response_length(token_ids, self._end_of_speaker_1_token_id)
        context_length = len(token_ids) - response_length

        # Find the same length or shorter response:
        number_of_same_or_shorter_responses = self._response_length_to_cumcount[response_length]

        max_distractor_ind = number_of_same_or_shorter_responses - 1
        selected_distractor_ind = random.randint(0, max_distractor_ind)
        selected_distractor_ind = self._argsort_of_response_lengths[selected_distractor_ind]

        source_of_distractor_token_ids = self._get_token_ids(selected_distractor_ind)

        # Get distractor response:
        distractor_response_length = get_response_length(source_of_distractor_token_ids,
                                                         self._end_of_speaker_1_token_id)
        distractor_response_token_ids = source_of_distractor_token_ids[-distractor_response_length:]

        # Construct new token ids with distracted response:
        distracted_token_ids = np.zeros(context_length + distractor_response_length, dtype=token_ids.dtype)
        distracted_token_ids[:context_length] = token_ids[:context_length]
        distracted_token_ids[context_length:] = distractor_response_token_ids

        return distracted_token_ids


def get_dataloader(dataset_dir, distractor_p, batch_size, num_workers, sort_chunk_size, samples_offset,
                   data_shuffle_seed, is_distributed, pad_token_id, end_of_speaker_1_token_id,
                   end_of_speaker_2_token_id):
    dataset = SerializedDataset(
        dataset_dir=dataset_dir, distractor_p=distractor_p, end_of_speaker_1_token_id=end_of_speaker_1_token_id)

    sampler = LengthSortSampler(
        lengths=dataset.sample_lengths,
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

    def __call__(self, items):
        samples, is_distracted = zip(*items)
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


if __name__ == '__main__':
    from dialog_model.dataset.serializer import load_tokenizer
    dataset_dir = '/ssd_1/data/dialog_model/datasets/flibusta/train'
    t = load_tokenizer(dataset_dir)
    # d = get_dataloader(
    #     dataset_dir=dataset_dir,
    #     distractor_p=0.5,
    #     batch_size=8,
    #     num_workers=4,
    #     sort_chunk_size=600,
    #     samples_offset=0,
    #     data_shuffle_seed=2,
    #     is_distributed=False,
    #     pad_token_id=t.pad_token_id,
    #     end_of_speaker_1_token_id=t.end_of_speaker_1_token_id,
    #     end_of_speaker_2_token_id=t.end_of_speaker_2_token_id)

    # for batch in d:
    #     print(batch)
    dataset = SerializedDataset(
        dataset_dir=dataset_dir, distractor_p=0.5, end_of_speaker_1_token_id=t.end_of_speaker_1_token_id)

    for token_ids, is_distracted in dataset:
        if is_distracted:
            print(t.decode(token_ids))
