from typing import Iterable, Sequence

import numpy as np
from more_itertools import flatten, chunked
from transformers import GPT2TokenizerFast

END_OF_SPEAKER_1_TOKEN = '[END_OF_SPEAKER_1]'
END_OF_SPEAKER_2_TOKEN = '[END_OF_SPEAKER_2]'
SPECIAL_TOKENS = [END_OF_SPEAKER_1_TOKEN, END_OF_SPEAKER_2_TOKEN]


class DialogsTokenizer:
    def __init__(self, tokenizer_name_or_path, max_n_tokens):
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name_or_path)

        self._max_n_tokens = max_n_tokens

        self._tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        self._dtype = np.uint16 if self._tokenizer.vocab_size < 65500 else np.int32
        self._end_of_speaker_1_token_id = self._tokenizer.convert_tokens_to_ids(END_OF_SPEAKER_1_TOKEN)
        self._end_of_speaker_2_token_id = self._tokenizer.convert_tokens_to_ids(END_OF_SPEAKER_2_TOKEN)

    @property
    def pad_token_id(self):
        return 0

    @property
    def end_of_speaker_1_token_id(self):
        return self._end_of_speaker_1_token_id

    @property
    def end_of_speaker_2_token_id(self):
        return self._end_of_speaker_2_token_id

    @property
    def vocab_size(self):
        return max(self._tokenizer.all_special_ids) + 1

    @property
    def max_n_tokens(self):
        return self._max_n_tokens

    def encode(self, dialogs: Iterable[Sequence[str]]):
        dialogs_with_speaker_tokens = []
        for dialog in dialogs:
            dialogs_with_speaker_tokens.append([])
            for i, utterance in enumerate(dialog):
                speaker_token = END_OF_SPEAKER_1_TOKEN if i % 2 == 0 else END_OF_SPEAKER_2_TOKEN
                dialogs_with_speaker_tokens[-1].append(utterance + speaker_token)

        utterances = list(set(flatten(dialogs_with_speaker_tokens)))
        token_ids = self._tokenizer.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
        utterance_to_ids = dict(zip(utterances, token_ids))

        encoded = []
        for dialog in dialogs_with_speaker_tokens:
            dialog_token_ids = list(flatten(utterance_to_ids[u] for u in dialog))
            dialog_token_ids = dialog_token_ids[:self._max_n_tokens]
            token_ids = np.array(dialog_token_ids, dtype=self._dtype)
            encoded.append(token_ids)

        return encoded

    def decode(self, encoded):
        return self._tokenizer.batch_decode(encoded)
