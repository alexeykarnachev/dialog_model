from collections import namedtuple

DialogModelInput = namedtuple('DialogModelInput', field_names=('token_ids', 'lm_labels', 'past'))
DialogModelOutput = namedtuple(
    'DialogModelOutput', field_names=('lm_loss', 'ul_loss', 'loss', 'logits', 'past', 'hidden'))
