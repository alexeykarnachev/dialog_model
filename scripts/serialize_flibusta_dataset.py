import argparse
from functools import partial
from itertools import islice
import multiprocessing
from pathlib import Path

from dialogs_data_parsers.flibusta.dialogs_iterator import FlibustaDialogsIterator

from dialog_model.dataset.serializer import DialogsDatasetSerializer


def _parse_args():
    parser = argparse.ArgumentParser(description='Constructs dataset from the given raw dialogs jsonl file.')

    parser.add_argument('--flibusta_file_path', type=str, required=True, help='Path to the raw flibusta jsonl file.')
    parser.add_argument('--n_valid_dialogs', type=int, required=True, help='Number of dialogs for validation set.')
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True, help='Huggingface tokenizer name or path.')
    parser.add_argument('--max_n_tokens', type=int, required=True, help='Max number of tokens in total.')
    parser.add_argument('--max_n_utterances', type=int, required=True, help='Max number of dialog utterances.')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the out dir with train and valid sub-dirs.')
    parser.add_argument('--n_workers',
                        type=int,
                        required=False,
                        default=multiprocessing.cpu_count(),
                        help='Number of multiprocessing workers.')

    args = parser.parse_args()

    return args


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)

    flibusta_dialogs = FlibustaDialogsIterator(args.flibusta_file_path, verbose=False)
    flibusta_valid_dialogs = islice(flibusta_dialogs, args.n_valid_dialogs)
    flibusta_train_dialogs = islice(flibusta_dialogs, args.n_valid_dialogs, None)

    _get_dialogs_dataset_serializer = partial(DialogsDatasetSerializer,
                                              tokenizer_name_or_path=args.tokenizer_name_or_path,
                                              n_workers=args.n_workers,
                                              max_n_tokens=args.max_n_tokens,
                                              max_n_utterances=args.max_n_utterances)

    _get_dialogs_dataset_serializer(dialogs=flibusta_valid_dialogs, out_serialized_dataset_dir=out_dir / 'valid').run()
    _get_dialogs_dataset_serializer(dialogs=flibusta_train_dialogs, out_serialized_dataset_dir=out_dir / 'train').run()


if __name__ == '__main__':
    main()
