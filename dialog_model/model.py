from collections import namedtuple

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

# input_ids - token ids of the input dialog sequence (including end-of-speaker separators)
# labels - some classification labels (for example, is-distractor labels)
# token_type_ids - 0 or 1, depending on current speaker
# lm_labels - input ids, where -100 placed on positions to be ignored by lm
ModelInput = namedtuple('ModelInput', ('input_ids', 'labels', 'token_type_ids', 'lm_labels'))

# lm_loss - loss from GPT-2 language model
# cls_loss - classification loss, calculated via classification on last end-of-speaker-2 tokens
ModelOutput = namedtuple('ModelOutput', ('lm_loss', 'cls_loss'))


class DialogModel(nn.Module):
    def __init__(self, gpt2_name_or_path, n_classes, end_of_speaker_2_token_id):
        super().__init__()
        self._gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name_or_path)
        self._hidden_size = self._gpt2.config.hidden_size
        self._classifier = nn.Linear(self._hidden_size, n_classes)
        self._end_of_speaker_2_token_id = end_of_speaker_2_token_id

    def forward(self, model_input):
        gpt2_output = self._gpt2(input_ids=model_input.input_ids,
                                 token_type_ids=model_input.token_type_ids,
                                 labels=model_input.lm_labels,
                                 return_dict=True,
                                 output_hidden_states=True)

        cls_token_vectors = self._extract_cls_token_vectors(model_input.input_ids, gpt2_output.hidden_states)
        cls_logits = self._classifier(cls_token_vectors)
        cls_loss = nn.CrossEntropyLoss()(cls_logits, model_input.labels)

        model_output = ModelOutput(lm_loss=gpt2_output.loss, cls_loss=cls_loss)
        return model_output

    def _extract_cls_token_vectors(self, input_ids, hidden_states):
        input_ids_lr_flipped = input_ids.fliplr()
        mask_flipped = (input_ids_lr_flipped == self._end_of_speaker_2_token_id).long()
        cls_token_positions = input_ids.size()[1] - mask_flipped.argmax(1) - 1
        cls_token_vectors = hidden_states[-1][torch.arange(len(input_ids)), cls_token_positions[None, :]].squeeze()

        return cls_token_vectors


# if __name__ == '__main__':
#     model = DialogModel('distilgpt2', 2, 10)
#     input_ids = torch.randint(low=7, high=11, size=(16, 30))
#     inp = ModelInput(input_ids=input_ids,
#                      labels=torch.randint(low=0, high=2, size=(16, )),
#                      token_type_ids=torch.randint_like(input_ids, low=0, high=1),
#                      lm_labels=input_ids)
#     model(inp)
#
