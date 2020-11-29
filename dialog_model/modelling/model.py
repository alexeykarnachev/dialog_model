from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPastAndCrossAttentions

from dialog_model.data_structures import DialogModelInput, DialogModelOutput


class DialogModel(nn.Module):

    def __init__(self, gpt2_lm_head: GPT2LMHeadModel):
        super().__init__()

        self._gpt2_lm_head = gpt2_lm_head

    @property
    def device(self):
        return self.parameters().__next__().device

    def forward(self, model_input: DialogModelInput):
        model_input = _cast_model_input(model_input)
        hf_output: CausalLMOutputWithPastAndCrossAttentions = self._gpt2_lm_head(
            input_ids=model_input.token_ids, labels=model_input.lm_labels, return_dict=True)

        loss = hf_output.loss

        output = DialogModelOutput(
            lm_loss=hf_output.loss, ul_loss=ul_loss, loss=loss, logits=hf_output.logits, past=None, hidden=None)

        return output

    @torch.no_grad()
    def infer(self, model_input: DialogModelInput) -> DialogModelOutput:
        """Performs forward pass without loss calculation."""

        model_input = _cast_model_input(model_input)
        hf_output: CausalLMOutputWithPastAndCrossAttentions = self._gpt2_lm_head(
            input_ids=model_input.token_ids, past_key_values=model_input.past, return_dict=True)
        output = DialogModelOutput(
            lm_loss=None,
            ul_loss=None,
            loss=None,
            logits=hf_output.logits,
            past=hf_output.past_key_values,
            hidden=hf_output.hidden_states)

        return output


def _cast_model_input(model_input) -> DialogModelInput:
    if not isinstance(model_input, DialogModelInput):
        model_input = DialogModelInput(*model_input)

    return model_input
