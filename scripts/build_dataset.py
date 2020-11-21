import argparse
from functools import partial
from pathlib import Path

from dialog_model.dataset.serialization import build_dataset

_TOKENIZATION_CHUNK_SIZE = 10000


def _parse_args():
    parser = argparse.ArgumentParser(description='Constructs dataset from the given raw raw_dialogs jsonl file.')

    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the raw raw_dialogs jsonl file.')
    parser.add_argument('--valid_file_path', type=str, required=True, help='Path to the raw raw_dialogs jsonl file.')
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True, help='Huggingface tokenizer name or path.')
    parser.add_argument('--tags_max_n_tokens', type=int, required=True, help='Max number of tokens in a tags sequence.')
    parser.add_argument('--context_max_n_tokens', type=int, required=True, help='Max number of tokens in context.')
    parser.add_argument('--total_max_n_tokens', type=int, required=True, help='Max number of tokens in total.')
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
        tags_max_n_tokens=args.tags_max_n_tokens,
        context_max_n_tokens=args.context_max_n_tokens,
        total_max_n_tokens=args.total_max_n_tokens
    )

    _build_dataset(out_dir=out_dir / 'train', dialogs_file_path=args.train_file_path)
    _build_dataset(out_dir=out_dir / 'valid', dialogs_file_path=args.valid_file_path)


if __name__ == '__main__':
    main()
