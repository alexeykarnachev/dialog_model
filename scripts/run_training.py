import argparse
import hashlib
import json
from pathlib import Path

from dialog_model.log_config import prepare_logging
from dialog_model.trainer import Trainer
from dialog_model.utils import get_file_md5_checksum

_HASH_ARGS = (
    'gpt2_name_or_path', 'worker_batch_size', 'data_shuffle_seed', 'learning_rate', 'n_epochs', 'warmup_ratio')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_root_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--gpt2_name_or_path', type=str, required=True)
    parser.add_argument('--worker_batch_size', type=int, required=True)
    parser.add_argument('--data_shuffle_seed', type=int, required=False, default=228)
    parser.add_argument('--freeze_n_layers', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--validate_each_n_steps', type=int, required=True)
    parser.add_argument('--warmup_ratio', type=float, required=True)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    experiment_hash = _calc_experiment_hash(args)
    experiment_dir = Path(args.experiments_root_dir) / experiment_hash
    experiment_dir.mkdir(exist_ok=True, parents=True)
    prepare_logging(experiment_dir / 'logs')

    train_dataset_dir = Path(args.dataset_dir) / 'train'
    valid_dataset_dir = Path(args.dataset_dir) / 'valid'

    trainer = Trainer(
        experiment_dir=experiment_dir,
        train_dataset_dir=train_dataset_dir,
        valid_dataset_dir=valid_dataset_dir,
        gpt2_name_or_path=args.gpt2_name_or_path,
        worker_batch_size=args.worker_batch_size,
        data_shuffle_seed=args.data_shuffle_seed,
        freeze_n_layers=args.freeze_n_layers,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        validate_each_n_steps=args.validate_each_n_steps,
        warmup_ratio=args.warmup_ratio
    )

    trainer.run()


def _calc_experiment_hash(args):
    args_dict = vars(args)
    hash_values = []
    for hash_arg in _HASH_ARGS:
        hash_values.append(args_dict[hash_arg])

    for data_dir in (args.train_dataset_dir, args.valid_dataset_dir):
        for file_name in ('data.bin', 'data.idx'):
            file_path = Path(data_dir) / file_name
            hash_values.append(get_file_md5_checksum(file_path))

    return hashlib.md5(json.dumps(hash_values).encode()).hexdigest()[:8]


if __name__ == '__main__':
    main()
