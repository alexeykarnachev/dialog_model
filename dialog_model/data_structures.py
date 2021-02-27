from collections import namedtuple

# input_ids - token ids of the input dialog sequence (including end-of-speaker separators)
# labels - some classification labels (for example, is-distractor labels)
# token_type_ids - 0 or 1, depending on current speaker
# lm_labels - input ids, where -100 placed on positions to be ignored by lm
ModelInput = namedtuple('ModelInput', ('input_ids', 'labels', 'token_type_ids', 'lm_labels'))

# lm_loss - loss from GPT-2 language model
# cls_loss - classification loss, calculated via classification on last end-of-speaker-2 tokens
ModelOutput = namedtuple('ModelOutput', ('lm_loss', 'cls_loss', 'loss'))
