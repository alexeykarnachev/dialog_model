from itertools import cycle

import numpy as np
from transformers import GPT2TokenizerFast

END_OF_SPEAKER_1_TOKEN = '[END_OF_SPEAKER_1]'
END_OF_SPEAKER_2_TOKEN = '[END_OF_SPEAKER_2]'
SPECIAL_TOKENS = [END_OF_SPEAKER_1_TOKEN, END_OF_SPEAKER_2_TOKEN]


class DialogsTokenizer:
    def __init__(self, tokenizer_name_or_path, max_n_tokens, max_n_utterances):
        self._tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name_or_path)

        self._max_n_tokens = max_n_tokens
        self._max_n_utterances = max_n_utterances

        self._tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        self._dtype = np.uint16 if self._tokenizer.vocab_size < 65500 else np.int32
        self._end_of_speaker_1_token_id = self._tokenizer.convert_tokens_to_ids(END_OF_SPEAKER_1_TOKEN)
        self._end_of_speaker_2_token_id = self._tokenizer.convert_tokens_to_ids(END_OF_SPEAKER_2_TOKEN)

    @property
    def pad_token_id(self):
        return self._tokenizer.eos_token_id

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

    @property
    def max_n_utterances(self):
        return self._max_n_utterances

    def iterate_on_encoded_subdialogs(self, dialog, skip_incomplete, encode_for_inference=False):
        encoded_dialog_utterances = list(map(self.encode_utterance, dialog))
        skip_incomplete = False is encode_for_inference
        min_subdialog_n_messages = len(dialog) if encode_for_inference else 2

        for subdialog_n_messages in range(min_subdialog_n_messages, len(encoded_dialog_utterances) + 1):
            encoded_subdialog_utterances = encoded_dialog_utterances[:subdialog_n_messages]
            if skip_incomplete:
                length_of_two_last_utterances = len(encoded_subdialog_utterances[-1]) + len(
                    encoded_subdialog_utterances[-2])
                length_of_two_last_utterances += 2  # Including 2 end-of-speaker tokens
                if length_of_two_last_utterances > self._max_n_tokens:
                    continue

            encoded_subdialog_utterances = encoded_subdialog_utterances[-self._max_n_utterances:]
            if not encode_for_inference:
                end_of_speaker_input_ids = (self.end_of_speaker_1_token_id, self.end_of_speaker_2_token_id)
            else:
                end_of_speaker_input_ids = (self.end_of_speaker_2_token_id, self.end_of_speaker_1_token_id)
            
            if len(encoded_subdialog_utterances) % 2 == 0:
                end_of_speaker_input_ids_cycle = cycle(end_of_speaker_input_ids)
            else:
                end_of_speaker_input_ids_cycle = cycle(reversed(end_of_speaker_input_ids))

            encoded_subdialog = []
            utterance_lengths = []
            for end_of_speaker_token_id, encoded_subdialog_utterance in zip(end_of_speaker_input_ids_cycle,
                                                                            encoded_subdialog_utterances):
                encoded_subdialog.extend(encoded_subdialog_utterance)
                encoded_subdialog.append(end_of_speaker_token_id)
                utterance_lengths.append(len(encoded_subdialog_utterance) + 1)

            encoded_subdialog = encoded_subdialog[-self._max_n_tokens:]
            encoded_subdialog = np.array(encoded_subdialog, dtype=self._dtype)

            yield encoded_subdialog

    def encode_utterance(self, utterance):
        encoded_utterance = self._tokenizer.encode(utterance, add_special_tokens=False)
        return encoded_utterance

    def decode(self, input_ids):
        return self._tokenizer.decode(input_ids, skip_special_tokens=True)
