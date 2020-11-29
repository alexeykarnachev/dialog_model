import re

from transformers import GPT2LMHeadModel


def get_pretrained_gpt2_with_lm_head(name_or_path, freeze_n_layers=None):
    model = GPT2LMHeadModel.from_pretrained(name_or_path, output_hidden_states=True)

    if freeze_n_layers is not None:
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

    return model


def _resize_embeddings(model, vocab_size: int):
    mean_emb = model.base_model.wte.weight.data.mean(0)
    old_size = model.base_model.wte.weight.data.size()[0]
    n_new = vocab_size - old_size

    if n_new < 0:
        raise ValueError(
            f"Can't resize embeddings: new vocab size ({vocab_size}) can not be less than the "
            f"old embeddings number ({old_size}).")

    model.resize_token_embeddings(vocab_size)
    idx = vocab_size - n_new
    model.base_model.wte.weight.data[idx:] = mean_emb.unsqueeze(0)
