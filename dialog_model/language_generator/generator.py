from typing import Sequence

import numpy as np
import torch
import torch.nn.functional

from dialog_model.data_structures import ModelInput
from dialog_model.dataset.serialized_dataset import Collate
from dialog_model.language_generator.logits_modifiers import IgnoredTokensModifier, \
    RepetitiveTokensModifier, TemperatureModifier, TopKNucleusModifier
from dialog_model.language_generator.progress import GenerationProgressTracker


class ResponseCandidatesGenerator:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    @torch.no_grad()
    def __call__(self,
                 context: Sequence[str],
                 n_candidates,
                 max_n_context_tokens,
                 repetition_penalty=3.0,
                 temperature=0.73,
                 top_k=100,
                 top_p=1.0):
        if max_n_context_tokens >= self._tokenizer.max_n_tokens:
            raise ValueError('`max_n_context_tokens` must be lower than `tokenizer.max_n_tokens`, '
                             'otherwise there are no tokens left for response.')

        self._model.eval()
        encoded_context = list(
            self._tokenizer.iterate_on_encoded_subdialogs(dialog=context,
                                                          encode_for_inference=True))[-1]
        encoded_context = encoded_context[-max_n_context_tokens:]
        max_number_of_generated_tokens = self._tokenizer.max_n_tokens - len(encoded_context)
        encoded = [list(encoded_context) for _ in range(n_candidates)]
        collate_fn = Collate(pad_token_id=self._tokenizer.pad_token_id,
                             end_of_speaker_1_token_id=self._tokenizer.end_of_speaker_1_token_id,
                             end_of_speaker_2_token_id=self._tokenizer.end_of_speaker_2_token_id,
                             device=self._model.device)

        model_input = collate_fn(encoded)
        input_ids, _, token_type_ids, _, _ = model_input
        new_token_type_ids = token_type_ids[:, -1:]
        new_token_type_ids = new_token_type_ids + 1 if new_token_type_ids[0] == 0 else new_token_type_ids - 1

        eos_input_ids = {self._tokenizer.end_of_speaker_1_token_id, self._tokenizer.end_of_speaker_2_token_id}
        progress = GenerationProgressTracker(eos_input_ids=eos_input_ids, max_length=max_number_of_generated_tokens)
        not_eos_positions = [i for i, token_id in enumerate(encoded_context) if token_id not in eos_input_ids]

        generated_input_ids = torch.zeros(n_candidates,
                                          max_number_of_generated_tokens,
                                          dtype=torch.long,
                                          device=self._model.device)

        past_input_ids = input_ids.detach().clone()
        past_input_ids = past_input_ids[:, not_eos_positions]
        past_key_values = None
        candidate_id_to_score = {}
        while not progress.finished:
            model_input = ModelInput(input_ids=input_ids,
                                     labels=None,
                                     token_type_ids=token_type_ids,
                                     lm_labels=None,
                                     past_key_values=past_key_values)
            model_output = self._model(model_input)
            next_token_logits = model_output.lm_logits[:, -1, :]
            past_input_ids = torch.cat(tensors=[past_input_ids, generated_input_ids], dim=1)
            _modify_next_token_logits(next_token_logits=next_token_logits,
                                      ignored_input_ids=[],
                                      input_ids_to_penalize=past_input_ids,
                                      repetition_penalty=repetition_penalty,
                                      temperature=temperature,
                                      top_k=top_k,
                                      top_p=top_p)
            next_input_ids = _sample_next_input_ids(next_token_logits)
            finished_just_now_mask = progress.update(next_input_ids)
            finished_just_now_inds = torch.where(finished_just_now_mask)[0]

            if len(finished_just_now_inds):
                scores = model_output.cls_logits[finished_just_now_inds].softmax(-1)[:, 0]
                for candidate_id, score in zip(finished_just_now_inds, scores):
                    candidate_id_to_score[candidate_id.item()] = score.item()

            generated_input_ids[:, progress.current_length - 1] = next_input_ids
            input_ids = next_input_ids.unsqueeze(1)
            token_type_ids = new_token_type_ids
            past_key_values = model_output.past_key_values

        candidates = _decode_candidates(tokenizer=self._tokenizer,
                                        generated_tokens=generated_input_ids,
                                        generated_sample_lengths=progress.generated_sample_lengths)
        scores = [candidate_id_to_score.get(i, -1) for i in range(len(candidates))]
        sorted_candidate_inds = np.argsort(scores)[::-1]
        candidates = [candidates[i] for i in sorted_candidate_inds]
        scores = [scores[i] for i in sorted_candidate_inds]
        return candidates, scores


def _modify_next_token_logits(next_token_logits, ignored_input_ids, input_ids_to_penalize, repetition_penalty,
                              temperature, top_k, top_p):
    modifiers = [
        IgnoredTokensModifier(ignored_input_ids=ignored_input_ids),
        RepetitiveTokensModifier(penalty=repetition_penalty, input_ids_to_penalize=input_ids_to_penalize),
        TemperatureModifier(temperature=temperature),
        TopKNucleusModifier(top_k=top_k, top_p=top_p)
    ]

    _ = [modifier(next_token_logits) for modifier in modifiers]


def _sample_next_input_ids(next_token_logits: torch.tensor) -> torch.tensor:
    probabilities = torch.nn.functional.softmax(input=next_token_logits, dim=-1)
    next_tokens = torch.multinomial(probabilities, num_samples=1)
    return next_tokens.squeeze(1)


def _decode_candidates(tokenizer, generated_tokens, generated_sample_lengths):
    encoded = []
    for i in range(generated_tokens.size()[0]):
        input_ids = generated_tokens[i, :generated_sample_lengths[i]]
        input_ids = input_ids.detach().cpu().numpy().tolist()
        encoded.append(input_ids)

    return [tokenizer.decode(sample) for sample in encoded]
