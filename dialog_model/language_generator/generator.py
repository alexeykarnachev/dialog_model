import torch
import torch.nn.functional

from dialog_model.data_structures import Dialog, DialogModelInput
from dialog_model.dataset.serialized_dataset import Collate
from dialog_model.language_generator.logits_modifiers import IgnoredTokensModifier, RepetitiveTokensModifier, \
    TemperatureModifier, TopKNucleusModifier
from dialog_model.language_generator.progress import GenerationProgressTracker
from dialog_model.modelling.model import DialogModel
from dialog_model.tokenization.dialogs_tokenizer import DialogsTokenizer


class LanguageGenerator:
    def __init__(self, model: DialogModel, tokenizer: DialogsTokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def __call__(
            self,
            dialog: Dialog,
            max_number_of_generated_tokens,
            num_return_sequences,
            repetition_penalty=3.0,
            temperature=0.73,
            top_k=100,
            top_p=1.0
    ):
        self._model.eval()

        dialogs = [dialog] * num_return_sequences
        encoded = self._tokenizer.encode(dialogs, with_subdialogs=False)
        collate_fn = Collate(
            pad_token_id=self._tokenizer.pad_token_id,
            end_of_prefix_token_id=self._tokenizer.end_of_prefix_token_id
        )

        model_input = collate_fn(encoded)
        model_input = DialogModelInput(*[x.to(self._model.device) if x is not None else x for x in model_input])

        progress = GenerationProgressTracker(
            eos_token_id=self._tokenizer.start_of_utterance_token_id,
            max_length=max_number_of_generated_tokens
        )

        generated_token_ids = torch.zeros(num_return_sequences, max_number_of_generated_tokens, dtype=torch.long)
        generated_token_ids = generated_token_ids.to(self._model.device)

        past_token_ids = model_input.token_ids.detach().clone()
        not_eos_mask = ~(past_token_ids == self._tokenizer.start_of_utterance_token_id).all(0)
        past_token_ids = past_token_ids[:, not_eos_mask]
        past_token_ids = past_token_ids.to(self._model.device)

        while not progress.finished:
            model_output = self._model.infer(model_input=model_input)
            next_token_logits = model_output.logits[:, -1, :]
            past_token_ids = torch.cat(tensors=[past_token_ids, generated_token_ids], dim=1)
            _modify_next_token_logits(
                next_token_logits=next_token_logits,
                ignored_token_ids=[],
                token_ids_to_penalize=past_token_ids,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            next_token_ids = _sample_next_token_ids(next_token_logits)
            progress.update(next_token_ids)
            generated_token_ids[:, progress.current_length - 1] = next_token_ids
            token_ids = next_token_ids.unsqueeze(1)

            model_input = DialogModelInput(token_ids=token_ids, past=model_output.past, lm_labels=None)

        candidates = _decode_candidates(
            tokenizer=self._tokenizer,
            generated_tokens=generated_token_ids,
            generated_sample_lengths=progress.generated_sample_lengths
        )

        return candidates


def _modify_next_token_logits(
        next_token_logits,
        ignored_token_ids,
        token_ids_to_penalize,
        repetition_penalty,
        temperature,
        top_k,
        top_p
):
    modifiers = [
        IgnoredTokensModifier(ignored_token_ids=ignored_token_ids),
        RepetitiveTokensModifier(penalty=repetition_penalty, token_ids_to_penalize=token_ids_to_penalize),
        TemperatureModifier(temperature=temperature),
        TopKNucleusModifier(top_k=top_k, top_p=top_p)
    ]

    _ = [modifier(next_token_logits) for modifier in modifiers]


def _sample_next_token_ids(next_token_logits: torch.tensor) -> torch.tensor:
    probabilities = torch.nn.functional.softmax(input=next_token_logits, dim=-1)
    next_tokens = torch.multinomial(probabilities, num_samples=1)
    return next_tokens.squeeze(1)


def _decode_candidates(tokenizer: DialogsTokenizer, generated_tokens, generated_sample_lengths):
    encoded = []
    for i in range(generated_tokens.size()[0]):
        token_ids = generated_tokens[i, :generated_sample_lengths[i]]
        token_ids = token_ids.detach().cpu().numpy().tolist()
        encoded.append(token_ids)

    return tokenizer.decode(encoded)
