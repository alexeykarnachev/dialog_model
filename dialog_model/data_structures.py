from collections import namedtuple

# input_ids - token ids of the input dialog sequence (including end-of-speaker separators)
# labels - some classification labels (for example, is-distractor labels)
# token_type_ids - 0 or 1, depending on current speaker
# lm_labels - input ids, where -100 placed on positions to be ignored by lm
# past_key_values - cached key values representations, which could be used to speed up inference
ModelInput = namedtuple('ModelInput', ('input_ids', 'labels', 'token_type_ids', 'lm_labels', 'past_key_values'))

# lm_loss - loss from GPT-2 language model
# cls_loss - classification loss, calculated via classification on last end-of-speaker-2 tokens
# loss - summed lm loss and cls loss
# lm_logits - logits from language model. Could be used for sampling new tokens
# cls_logits - logits from classification model. Could be used for generated sequences scoring
# past_key_values - cached key values representations, which could be used to speed up inference
ModelOutput = namedtuple('ModelOutput', ('lm_loss', 'cls_loss', 'loss', 'lm_logits', 'cls_logits', 'past_key_values'))
