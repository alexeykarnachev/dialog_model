import argparse
from functools import partial
from itertools import chain, islice
import logging
import multiprocessing
from pathlib import Path

from dialogs_data_parsers.flibusta.dialogs_iterator import FlibustaDialogsIterator
from dialogs_data_parsers.pikabu.dialogs_iterator import PikabuDialogsIterator

from dialog_model.dataset.serializer import DialogsDatasetSerializer

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])


def _parse_args():
    parser = argparse.ArgumentParser(description='Constructs dataset from the given raw dialogs jsonl file.')

    parser.add_argument('--pikabu_file_path', type=str, required=True, help='Path to the raw pikabu jsonl file.')
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

    n_valid_dialogs_per_dataset = args.n_valid_dialogs // 2

    pikabu_dialogs = PikabuDialogsIterator(args.pikabu_file_path, 10000)
    pikabu_valid_dialogs = islice(pikabu_dialogs, n_valid_dialogs_per_dataset)
    pikabu_train_dialogs = islice(pikabu_dialogs, n_valid_dialogs_per_dataset, None)

    flibusta_dialogs = FlibustaDialogsIterator(args.flibusta_file_path, 50000)
    flibusta_valid_dialogs = islice(flibusta_dialogs, n_valid_dialogs_per_dataset)
    flibusta_train_dialogs = islice(flibusta_dialogs, n_valid_dialogs_per_dataset, None)

    valid_dialogs = chain(pikabu_valid_dialogs, flibusta_valid_dialogs)
    train_dialogs = chain(pikabu_train_dialogs, flibusta_train_dialogs)

    _get_dialogs_dataset_serializer = partial(DialogsDatasetSerializer,
                                              tokenizer_name_or_path=args.tokenizer_name_or_path,
                                              n_workers=args.n_workers,
                                              max_n_tokens=args.max_n_tokens,
                                              max_n_utterances=args.max_n_utterances)

    _get_dialogs_dataset_serializer(dialogs=valid_dialogs, out_serialized_dataset_dir=out_dir / 'valid').run()
    _get_dialogs_dataset_serializer(dialogs=train_dialogs, out_serialized_dataset_dir=out_dir / 'train').run()


if __name__ == '__main__':
    main()
