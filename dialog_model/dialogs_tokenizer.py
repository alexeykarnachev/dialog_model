from typing import Iterable, Sequence

import numpy as np
from more_itertools import flatten, chunked
from transformers import GPT2TokenizerFast

END_OF_UTTERANCE = '—'


class DialogsTokenizer:
    def __init__(self, tokenizer_name_or_path, max_n_tokens):
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name_or_path)

        self._max_n_tokens = max_n_tokens

        self._dtype = np.uint16 if self._tokenizer.vocab_size < 65500 else np.int32
        self._end_of_utterance_token_id = self._tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE)
        self._reference_token_id = self._tokenizer.convert_tokens_to_ids('—')

    @property
    def pad_token_id(self):
        return 0

    @property
    def end_of_utterance_token_id(self):
        return self._end_of_utterance_token_id

    @property
    def reference_token_id(self):
        return self._reference_token_id

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    @property
    def max_n_tokens(self):
        return self._max_n_tokens

    def encode(self, dialogs: Iterable[Sequence[str]], with_subdialogs):
        utterances = [utterance + END_OF_UTTERANCE for utterance in flatten(dialogs)]
        token_ids = self._tokenizer.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
        utterance_to_ids = dict(zip(utterances, token_ids))

        encoded = []
        for dialog in dialogs:
            dialog_token_ids = list(flatten(utterance_to_ids[u + END_OF_UTTERANCE] for u in dialog))

            if not with_subdialogs:
                dialog_token_ids = dialog_token_ids[-self._max_n_tokens:]

            for utterances_token_ids_chunk in chunked(dialog_token_ids, n=self._max_n_tokens):
                token_ids = np.array(utterances_token_ids_chunk, dtype=self._dtype)
                encoded.append(token_ids)

        return encoded

    def decode(self, encoded):
        return self._tokenizer.batch_decode(encoded)


if __name__ == '__main__':
    t = GPT2TokenizerFast.from_pretrained('sberbank-ai/rugpt3medium_based_on_gpt2')
    a = 1
