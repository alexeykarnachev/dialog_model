import argparse
from functools import partial
from itertools import islice, chain
from pathlib import Path

from dialog_model.dataset.serialization import build_dataset
from dialog_model.raw_dialogs.flibusta_dialogs_iterator import FlibustaDialogsIterator
from dialog_model.raw_dialogs.pikabu_dialogs_iterator import PikabuDialogsIterator

_TOKENIZATION_CHUNK_SIZE = 10000


def _parse_args():
    parser = argparse.ArgumentParser(description='Constructs dataset from the given raw raw_dialogs jsonl file.')

    parser.add_argument('--pikabu_file_path', type=str, required=True, help='Path to the raw pikabu jsonl file.')
    parser.add_argument('--flibusta_file_path', type=str, required=True, help='Path to the raw flibusta jsonl file.')
    parser.add_argument('--n_valid_dialogs', type=int, required=True, help='Number of dialogs for validation set.')
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True, help='Huggingface tokenizer name or path.')
    parser.add_argument('--max_n_tokens', type=int, required=True, help='Max number of tokens in total.')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the out dir with train and valid sub-dirs.')

    args = parser.parse_args()

    return args


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=False, parents=True)

    _build_dataset = partial(
        build_dataset,
        tokenization_chunk_size=_TOKENIZATION_CHUNK_SIZE,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        max_n_tokens=args.max_n_tokens)

    pikabu_dialogs = PikabuDialogsIterator(args.pikabu_file_path)
    flibusta_dialogs = FlibustaDialogsIterator(args.flibusta_file_path)
    pikabu_valid_dialogs = islice(pikabu_dialogs, args.n_valid_dialogs // 2)
    pikabu_train_dialogs = islice(pikabu_dialogs, args.n_valid_dialogs // 2, None)
    flibusta_valid_dialogs = islice(flibusta_dialogs, args.n_valid_dialogs // 2)
    flibusta_train_dialogs = islice(flibusta_dialogs, args.n_valid_dialogs // 2, None)

    valid_dialogs = chain(pikabu_valid_dialogs, flibusta_valid_dialogs)
    train_dialogs = chain(pikabu_train_dialogs, flibusta_train_dialogs)

    _build_dataset(out_dir=out_dir / 'valid', dialogs=valid_dialogs)
    _build_dataset(out_dir=out_dir / 'train', dialogs=train_dialogs)


if __name__ == '__main__':
    main()
