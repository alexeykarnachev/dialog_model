import argparse
from functools import partial
from itertools import islice
from pathlib import Path

from dialog_model.dataset.serialization import build_dataset
from dialog_model.raw_dialogs.pikabu_dialogs_iterator import PikabuDialogsIterator

_TOKENIZATION_CHUNK_SIZE = 10000


def _parse_args():
    parser = argparse.ArgumentParser(description='Constructs dataset from the given raw raw_dialogs jsonl file.')

    parser.add_argument('--file_path', type=str, required=True, help='Path to the raw dialogs jsonl file.')
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

    dialogs = PikabuDialogsIterator(args.file_path)
    valid_dialogs = islice(dialogs, args.n_valid_dialogs)
    train_dialogs = islice(dialogs, args.n_valid_dialogs, None)

    _build_dataset(out_dir=out_dir / 'valid', dialogs=valid_dialogs)
    _build_dataset(out_dir=out_dir / 'train', dialogs=train_dialogs)


if __name__ == '__main__':
    main()
