from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPastAndCrossAttentions

from dialog_model.data_structures import DialogModelInput, DialogModelOutput
from dialog_model.modelling.unlikelihood_loss import unlikelihood_loss_fn


class DialogModel(nn.Module):

    def __init__(self, gpt2_lm_head: GPT2LMHeadModel, unlikelihood_alpha: Optional[float]):
        super().__init__()

        self._gpt2_lm_head = gpt2_lm_head
        self._ul_alpha = unlikelihood_alpha

    @property
    def device(self):
        return self.parameters().__next__().device

    def forward(self, model_input: DialogModelInput) -> DialogModelOutput:
        model_input = _cast_model_input(model_input)
        hf_output: CausalLMOutputWithPastAndCrossAttentions = self._gpt2_lm_head(
            input_ids=model_input.token_ids, labels=model_input.lm_labels, return_dict=True)

        loss = hf_output.loss

        if self._ul_alpha:
            ul_loss = unlikelihood_loss_fn(logits=hf_output.logits, target=model_input.lm_labels) * self._ul_alpha
            loss += ul_loss
        else:
            ul_loss = torch.tensor(0)

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
