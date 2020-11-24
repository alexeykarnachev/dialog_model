from transformers import GPT2LMHeadModel


def get_pretrained_gpt2_with_lm_head(name_or_path, vocab_size):
    model = GPT2LMHeadModel.from_pretrained(name_or_path, output_hidden_states=True)
    if vocab_size is not None:
        _resize_embeddings(model=model, vocab_size=vocab_size)

    return model


def _resize_embeddings(model, vocab_size: int):
    mean_emb = model.base_model.wte.weight.data.mean(0)
    old_size = model.base_model.wte.weight.data.size()[0]
    n_new = vocab_size - old_size

    if n_new < 0:
        raise ValueError(
            "Can't resize embeddings: new vocab size can not be less than the "
            "old embeddings number (old vocab size).")

    model.resize_token_embeddings(vocab_size)
    idx = vocab_size - n_new
    model.base_model.wte.weight.data[idx:] = mean_emb.unsqueeze(0)
