import argparse
from functools import partial

import torch
import torch.multiprocessing as mp

from dialog_model.modelling.model_io import get_pretrained_gpt2_lm_head
from dialog_model.training.trainer import train


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_dir', type=str, required=True)
    parser.add_argument('--valid_dataset_dir', type=str, required=True)
    parser.add_argument('--gpt2_name_or_path', type=str, required=True)
    parser.add_argument('--unlikelihood_alpha', type=float, required=False)
    parser.add_argument('--worker_batch_size', type=int, required=True)
    parser.add_argument('--data_shuffle_seed', type=int, required=False, default=228)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    get_pretrained_gpt2_lm_head(args.gpt2_name_or_path)

    world_size = torch.cuda.device_count()

    _train = partial(
        train,
        world_size=world_size,
        train_dataset_dir=args.train_dataset_dir,
        valid_dataset_dir=args.valid_dataset_dir,
        gpt2_name_or_path=args.gpt2_name_or_path,
        unlikelihood_alpha=args.unlikelihood_alpha,
        worker_batch_size=args.worker_batch_size,
        data_shuffle_seed=args.data_shuffle_seed,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs
    )

    mp.spawn(_train, nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
