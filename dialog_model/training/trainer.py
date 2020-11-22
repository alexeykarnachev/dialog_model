import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from transformers import AdamW

from dialog_model.dataset.serialization import load_tokenizer
from dialog_model.dataset.serialized_dataset import get_dataloader
from dialog_model.modelling.model import DialogModel
from dialog_model.modelling.model_io import get_pretrained_gpt2_lm_head

_logger = logging.getLogger(__name__)


class Trainer:
    _MASTER_ADDR = 'localhost'
    _MASTER_PORT = '12355'

    def __init__(
            self,
            train_dataset_dir,
            valid_dataset_dir,
            gpt2_name_or_path,
            unlikelihood_alpha,
            worker_batch_size,
            data_shuffle_seed,
            learning_rate,
            n_epochs,
            validate_each_n_steps
    ):
        self._train_dataset_dir = train_dataset_dir
        self._valid_dataset_dir = valid_dataset_dir
        self._gpt2_name_or_path = gpt2_name_or_path
        self._unlikelihood_alpha = unlikelihood_alpha
        self._worker_batch_size = worker_batch_size
        self._data_shuffle_seed = data_shuffle_seed
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._validate_each_n_steps = validate_each_n_steps

        self._world_size = torch.cuda.device_count()
        self._tokenizer = load_tokenizer(dataset_dir=self._train_dataset_dir)

    def run(self):
        get_pretrained_gpt2_lm_head(self._gpt2_name_or_path)
        load_tokenizer(self._train_dataset_dir)
        mp.spawn(self._train, nprocs=self._world_size, join=True)

    def _train(self, rank):
        _logger.info(f'Running ddp training on rank: {rank}.')
        self._setup_ddp(rank)

        model = self._get_model(rank)
        train_dl = self._get_dataloader(is_train=True, samples_offset=0)
        valid_dl = self._get_dataloader(is_train=False, samples_offset=0)
        optimizer = AdamW(params=model.parameters(), lr=self._learning_rate)
        epochs_iter = range(self._n_epochs)

        if rank == 0:
            epochs_iter = tqdm.tqdm(epochs_iter, desc='Epoch', total=len(epochs_iter), position=0, leave=False)
            train_dl = tqdm.tqdm(train_dl, desc='Train step', total=len(train_dl), position=1, leave=False)
            valid_dl = tqdm.tqdm(valid_dl, desc='Valid step', total=len(valid_dl), position=2, leave=False)

        log_postfix = {}
        scaler = GradScaler()
        for i_epoch in epochs_iter:
            for i_step, model_input in enumerate(train_dl):

                if i_step and i_step % self._validate_each_n_steps == 0:
                    if rank == 0:
                        valid_results = self._validate(model, valid_dl)
                        log_postfix.update(valid_results)

                    # dist.barrier()

                optimizer.zero_grad()
                with autocast():
                    model_output = model(model_input)

                scaler.scale(model_output.loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # if rank == 0:
                #     log_postfix.update({'loss/Train': dist.all_reduce(model_output.loss) / self._world_size})
                #     train_dl.set_postfix(log_postfix)

        dist.destroy_process_group()

    def _setup_ddp(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=self._world_size)

    def _get_model(self, rank):
        gpt2 = get_pretrained_gpt2_lm_head(self._gpt2_name_or_path)
        model = DialogModel(gpt2_lm_head=gpt2, unlikelihood_alpha=self._unlikelihood_alpha).to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])

        return model

    def _get_dataloader(self, is_train, samples_offset):
        return get_dataloader(
            dataset_dir=self._train_dataset_dir if is_train else self._valid_dataset_dir,
            batch_size=self._worker_batch_size,
            num_workers=4,
            sort_chunk_size=self._worker_batch_size * 1000,
            samples_offset=samples_offset,
            data_shuffle_seed=self._data_shuffle_seed,
            is_distributed=is_train,
            pad_token_id=self._tokenizer.pad_token_id,
            end_of_prefix_token_id=self._tokenizer.end_of_prefix_token_id
        )

    @torch.no_grad()
    def _validate(self, model: DialogModel, valid_dl):
        print('1'*20)
        was_training = model.training
        model.eval()
        print('2' * 20)

        valid_results = defaultdict(lambda: 0)
        n_samples = 0
        print('3' * 20)
        for model_input in valid_dl:
            print('4' * 20)
            with autocast():
                print('5' * 20)
                model_output = model(model_input)
                print('6' * 20)

            valid_results['lm_loss/Valid'] += model_output.lm_loss
            valid_results['ul_loss/Valid'] += model_output.ul_loss
            valid_results['loss/Valid'] += model_output.loss

            n_samples += len(model_output.token_ids)

        valid_results = {k: v / n_samples for k, v in valid_results.items()}

        if was_training:
            model.train()

        return valid_results
