from typing import Iterable

import torch


class GenerationProgressTrackerError(Exception):
    pass


def _return_value_if_not_initialized(value):
    def _return_value_if_not_initialized_inner(function_or_prop):
        def decorated(self, *args, **kwargs):
            if self._n_samples is None:
                return value
            else:
                return function_or_prop(self, *args, **kwargs)

        return decorated

    return _return_value_if_not_initialized_inner


class GenerationProgressTracker:
    """Tracks generation progress."""

    @property
    @_return_value_if_not_initialized(value=False)
    def max_length_reached(self):
        """Will be True, when all sequence lengths will reach max_length."""
        return self.current_length >= self._max_length

    @property
    @_return_value_if_not_initialized(value=False)
    def all_samples_finished(self):
        """Will be True, when eos_token_id will appear in every sequence."""
        return self._unfinished_mask.max() == 0

    @property
    @_return_value_if_not_initialized(value=False)
    def finished(self):
        return self.max_length_reached or self.all_samples_finished

    @property
    def generated_sample_lengths(self):
        return self._gen_lengths

    def __init__(self, eos_token_ids: Iterable[int], max_length: int):
        """
        Args:
            eos_token_ids:
                End of string token ids. It's needed for GeneratorProgress to
                understand which sample is finished.
            max_length:
                Maximum length of the generated sequences.
        """
        self._eos_token_ids = eos_token_ids
        self._max_length = max_length
        self.current_length = 0

        self._n_samples = None
        self._unfinished_mask = None
        self._gen_lengths = None

        self._check_arguments_validity()

    def _check_arguments_validity(self) -> None:
        if self._max_length < 1:
            raise GenerationProgressTrackerError("`max_length` must be >= 1.")
        elif min(self._eos_token_ids) < 0:
            raise GenerationProgressTrackerError("`eos_token_ids` must be >= 0.")

    def update(self, next_token_ids) -> None:
        """Updates generation progress status."""
        self._assert_update_is_possible()
        self._initialize_if_needed(next_token_ids)

        not_eos_tokens_mask = torch.ones_like(next_token_ids).bool()
        for eos_token_id in self._eos_token_ids:
            not_eos_tokens_mask &= next_token_ids.ne(eos_token_id).bool()

        self._gen_lengths[self._unfinished_mask] += 1

        self._unfinished_mask *= not_eos_tokens_mask
        self.current_length += 1

    def _assert_update_is_possible(self):
        if self.finished:
            raise GenerationProgressTrackerError(
                "Can't update generation progress, because it's already finished.")

    def _initialize_if_needed(self, next_tokens):
        if self._n_samples is None:
            device = next_tokens.device
            self._n_samples = len(next_tokens)
            self._unfinished_mask = torch.ones(self._n_samples, dtype=torch.bool, device=device)
            self._gen_lengths = torch.zeros(self._n_samples, dtype=torch.long, device=device)
