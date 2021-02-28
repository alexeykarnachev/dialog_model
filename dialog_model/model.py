from pathlib import Path
import re

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from dialog_model.data_structures import ModelOutput
from dialog_model.dataset.serializer import load_tokenizer
from dialog_model.language_generator.generator import ResponseCandidatesGenerator


class DialogModel(nn.Module):
    def __init__(self, gpt2_name_or_path, vocab_size, n_classes, end_of_speaker_2_token_id, cls_loss_weight):
        super().__init__()
        self._gpt2 = get_pretrained_gpt2_with_lm_head(gpt2_name_or_path, vocab_size=vocab_size)
        self._hidden_size = self._gpt2.config.hidden_size
        self._classifier = nn.Linear(self._hidden_size, n_classes)
        self._end_of_speaker_2_token_id = end_of_speaker_2_token_id
        self._cls_loss_weight = cls_loss_weight

    @property
    def gpt2(self):
        return self._gpt2

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, model_input):
        gpt2_output = self._gpt2(input_ids=model_input.input_ids,
                                 token_type_ids=model_input.token_type_ids,
                                 labels=model_input.lm_labels,
                                 return_dict=True,
                                 output_hidden_states=True,
                                 use_cache=True,
                                 past_key_values=model_input.past_key_values)
        if model_input.lm_labels is not None:
            lm_loss = gpt2_output.loss
        else:
            lm_loss = None
        cls_token_vectors = self._extract_cls_token_vectors(model_input.input_ids, gpt2_output.hidden_states)
        cls_logits = self._classifier(cls_token_vectors)
        if model_input.labels is not None:
            cls_loss = nn.CrossEntropyLoss()(cls_logits, model_input.labels)
        else:
            cls_loss = None

        if None not in (lm_loss, cls_loss):
            loss = cls_loss * self._cls_loss_weight + lm_loss
        else:
            loss = None

        model_output = ModelOutput(lm_loss=lm_loss,
                                   cls_loss=cls_loss,
                                   loss=loss,
                                   lm_logits=gpt2_output.logits,
                                   cls_logits=cls_logits,
                                   past_key_values=gpt2_output.past_key_values)

        return model_output

    def _extract_cls_token_vectors(self, input_ids, hidden_states):
        input_ids_lr_flipped = input_ids.fliplr()
        mask_flipped = (input_ids_lr_flipped == self._end_of_speaker_2_token_id).long()
        cls_token_positions = input_ids.size()[1] - mask_flipped.argmax(1) - 1
        cls_token_vectors = hidden_states[-1][torch.arange(len(input_ids)), cls_token_positions[None, :]].squeeze()

        return cls_token_vectors


def get_pretrained_gpt2_with_lm_head(gpt2_name_or_path, vocab_size=None, freeze_n_layers=None) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(gpt2_name_or_path, output_hidden_states=True)

    if vocab_size is not None:
        _resize_embeddings(model=model, vocab_size=vocab_size)

    if freeze_n_layers is not None:
        _freeze_layers(model=model, freeze_n_layers=freeze_n_layers)

    return model


CHECKPOINTS_DIR_NAME = 'checkpoint'


def load_model_from_checkpoint(checkpoint_file_path, device) -> GPT2LMHeadModel:
    checkpoint = torch.load(f=checkpoint_file_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    state_dict = {re.sub(r'^module\.', '', name): weights for name, weights in state_dict.items()}
    model = DialogModel(**checkpoint['model_params'])
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def load_response_candidates_generator_from_experiment_dir(experiment_dir, checkpoint_name,
                                                           device) -> ResponseCandidatesGenerator:
    experiment_dir = Path(experiment_dir)
    tokenizer = load_tokenizer(experiment_dir)
    checkpoint_file_path = experiment_dir / CHECKPOINTS_DIR_NAME / checkpoint_name
    model = load_model_from_checkpoint(checkpoint_file_path, device=device)
    generator = ResponseCandidatesGenerator(model=model, tokenizer=tokenizer)

    return generator


def _freeze_layers(model, freeze_n_layers: int):
    freeze_layers_prefixes = {f'transformer.h.{i}.' for i in range(freeze_n_layers)}
    frozen_layer_prefixes = set()
    for name, param in model.named_parameters():
        match = re.search(r'transformer\.h\.\d+\.', name)
        if match:
            layer_prefix = match.group(0)
            if layer_prefix in freeze_layers_prefixes:
                param.requires_grad = False
                frozen_layer_prefixes.add(layer_prefix)

    assert len(frozen_layer_prefixes) == len(freeze_layers_prefixes)


def _resize_embeddings(model, vocab_size: int):
    old_size = model.base_model.wte.weight.data.size()[0]
    n_new = vocab_size - old_size

    if n_new < 0:
        raise ValueError(f"Can't resize embeddings: new vocab size ({vocab_size}) can not be less than the "
                         f"old embeddings number ({old_size}).")

    model.resize_token_embeddings(vocab_size)
    idx = vocab_size - n_new
    reference_emb = model.base_model.wte.weight.data.mean(0)
    model.base_model.wte.weight.data[idx:] = reference_emb.unsqueeze(0)
