from itertools import chain
from typing import List, Dict, Iterable

import numpy as np
from more_itertools import flatten, chunked
from transformers import GPT2TokenizerFast

from dialog_model.data_structures import Dialog
from dialog_model.tokenization.special_tokens import (
    START_OF_TAG,
    SPECIAL_TOKENS,
    START_OF_CONTEXT,
    START_OF_UTTERANCE,
    END_OF_PREFIX
)


class DialogsTokenizer:
    def __init__(
            self,
            tokenizer_name_or_path,
            tags_max_n_tokens,
            context_max_n_tokens,
            total_max_n_tokens
    ):
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name_or_path)

        self._tags_max_n_tokens = tags_max_n_tokens
        self._context_max_n_tokens = context_max_n_tokens
        self._total_max_n_tokens = total_max_n_tokens

        if self._tags_max_n_tokens + self._context_max_n_tokens >= self._total_max_n_tokens:
            raise ValueError('`tags_max_n_tokens` + `context_max_n_tokens` must be < `total_max_n_tokens`.')

        self._tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        self._dtype = np.uint16 if self._tokenizer.vocab_size < 65500 else np.int32

        self._end_of_prefix_token_id = self._tokenizer.convert_tokens_to_ids(END_OF_PREFIX)
        self._start_of_utterance_token_id = self._tokenizer.convert_tokens_to_ids(START_OF_UTTERANCE)
        self._pad_token_id = 0

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def end_of_prefix_token_id(self):
        return self._end_of_prefix_token_id

    @property
    def start_of_utterance_token_id(self):
        return self._start_of_utterance_token_id

    @property
    def vocab_size(self):
        return max(self._tokenizer.all_special_ids) + 1

    def encode(self, dialogs: Iterable[Dialog], with_subdialogs, append_start_of_utterance_token=False):
        contexts = list(set(dialog.context for dialog in dialogs))
        tags = list(set(flatten(dialog.tags for dialog in dialogs)))
        utterances = list(set(flatten(dialog.utterances for dialog in dialogs)))

        context_to_ids = self._get_text_to_token_ids(contexts, START_OF_CONTEXT)
        tag_to_ids = self._get_text_to_token_ids(tags, START_OF_TAG)
        utterance_to_ids = self._get_text_to_token_ids(utterances, START_OF_UTTERANCE)

        encoded = []
        for dialog in dialogs:
            context_token_ids = context_to_ids[dialog.context][-self._context_max_n_tokens:]
            tags_token_ids = list(flatten(tag_to_ids[t] for t in dialog.tags))[-self._tags_max_n_tokens:]
            prefix_token_ids = list(chain(tags_token_ids, context_token_ids, [self.end_of_prefix_token_id]))
            utterances_token_ids = list(flatten(utterance_to_ids[u] for u in dialog.utterances))

            utterances_n_tokens = self._total_max_n_tokens - len(prefix_token_ids) - 1

            if not with_subdialogs:
                utterances_token_ids = utterances_token_ids[-utterances_n_tokens:]

            for utterances_token_ids_chunk in chunked(utterances_token_ids, n=utterances_n_tokens):
                token_ids = list(chain(prefix_token_ids, utterances_token_ids_chunk))
                token_ids.append(self._start_of_utterance_token_id)
                token_ids = np.array(token_ids, dtype=self._dtype)
                encoded.append(token_ids)

        return encoded

    def decode(self, encoded):
        return self._tokenizer.batch_decode(encoded)

    def _get_text_to_token_ids(self, texts: List[str], bos_token) -> Dict[str, List[int]]:
        texts_with_eos = [bos_token + text for text in texts]
        token_ids = self._tokenizer.batch_encode_plus(texts_with_eos, add_special_tokens=False)['input_ids']
        text_to_token_ids = dict(zip(texts, token_ids))
        return text_to_token_ids
