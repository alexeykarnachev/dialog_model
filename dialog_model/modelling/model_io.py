from transformers import GPT2LMHeadModel


def get_pretrained_gpt2_lm_head(name_or_path):
    return GPT2LMHeadModel.from_pretrained(name_or_path, output_hidden_states=True)
