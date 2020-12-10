import re
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Config
import re
from dialog_model.dataset.serialization import load_tokenizer, TOKENIZER_PARAMS_FILE_NAME
from dialog_model.language_generator.generator import ResponseCandidatesGenerator

CHECKPOINTS_DIR_NAME = 'checkpoint'


def get_pretrained_gpt2_with_lm_head(name_or_path, vocab_size=None, freeze_n_layers=None) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(name_or_path, output_hidden_states=True)

    if vocab_size is not None:
        _resize_embeddings(model=model, vocab_size=vocab_size)

    if freeze_n_layers is not None:
        _freeze_layers(model=model, freeze_n_layers=freeze_n_layers)

    return model


def load_model_from_checkpoint(checkpoint_file_path, device) -> GPT2LMHeadModel:
    checkpoint = torch.load(f=checkpoint_file_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    state_dict = {re.sub(r'^module\.', '', name): weights for name, weights in state_dict.items()}
    gpt2_config_dict = checkpoint['gpt2_config_dict']
    model = GPT2LMHeadModel(config=GPT2Config(**gpt2_config_dict))
    vocab_size = state_dict['transformer.wte.weight'].size()[0]
    _resize_embeddings(model=model, vocab_size=vocab_size)
    model.load_state_dict(state_dict)
    model = model.to(device)
    if device != 'cpu':
        model = model.half()

    return model


def load_response_candidates_generator_from_experiment_dir(
        experiment_dir, checkpoint_name, device
) -> ResponseCandidatesGenerator:
    experiment_dir = Path(experiment_dir)
    tokenizer = load_tokenizer(experiment_dir / TOKENIZER_PARAMS_FILE_NAME)
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
        raise ValueError(
            f"Can't resize embeddings: new vocab size ({vocab_size}) can not be less than the "
            f"old embeddings number ({old_size}).")

    model.resize_token_embeddings(vocab_size)
    idx = vocab_size - n_new
    reference_emb = model.base_model.wte.weight.data.mean(0)
    model.base_model.wte.weight.data[idx:] = reference_emb.unsqueeze(0)
