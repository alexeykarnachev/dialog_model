import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
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
            experiment_dir,
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
        self._experiment_dir = Path(experiment_dir)
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

        self._optimizer = None
        self._scaler = None
        self._rank = None
        self._model = None
        self._train_dl = None
        self._valid_dl = None
        self._samples_seen = None

    def run(self):
        get_pretrained_gpt2_lm_head(self._gpt2_name_or_path)
        load_tokenizer(self._train_dataset_dir)
        mp.spawn(self._train, nprocs=self._world_size, join=True)

    def _train(self, rank):
        _logger.info(f'Running ddp training on rank: {rank}.')
        self._setup_ddp(rank)
        self._rank = rank
        self._scaler = GradScaler()
        self._model = self._get_model(self._rank)
        self._optimizer = AdamW(params=self._model.parameters(), lr=self._learning_rate)
        self._train_dl = self._get_dataloader(is_train=True, samples_offset=0)
        self._valid_dl = self._get_dataloader(is_train=False, samples_offset=0)
        self._samples_seen = 0

        if self._rank == 0:
            self._writer = SummaryWriter(self._experiment_dir / 'tb_logs')
            self._train_dl = tqdm.tqdm(self._train_dl, desc='Train step', total=len(self._train_dl), position=1)

        for i_epoch in range(self._n_epochs):
            for i_step, model_input in enumerate(self._train_dl):
                self._model.train()

                if self._rank == 0 and i_step and i_step % self._validate_each_n_steps == 0:
                    valid_losses = self._validate(self._model, self._valid_dl)
                    self._write_tb_logs(valid_losses)

                train_losses = self._train_step(model_input)
                self._samples_seen += len(model_input.token_ids) * self._world_size

                if rank == 0:
                    self._train_dl.set_postfix({'loss': train_losses['loss/train']})
                    self._write_tb_logs(train_losses)
                    self._write_tb_logs({'learning-rate': self._optimizer.param_groups[0]['lr']})
                    self._write_tb_logs({'epoch': i_epoch})

        dist.destroy_process_group()

    def _write_tb_logs(self, values_dict):
        for tag, val in values_dict.items():
            self._writer.add_scalar(tag=tag, scalar_value=val, global_step=self._samples_seen)

    def _train_step(self, model_input):
        self._optimizer.zero_grad()
        with autocast():
            model_output = self._model(model_input)

        # self._scaler.scale(model_output.loss).backward()
        # self._scaler.step(self._optimizer)
        # self._scaler.update()

        dist.all_reduce(model_output.lm_loss)
        dist.all_reduce(model_output.ul_loss)
        dist.all_reduce(model_output.loss)

        losses = {
            'lm_loss/train': (model_output.lm_loss / self._world_size).item(),
            'ul_loss/train': (model_output.ul_loss / self._world_size).item(),
            'loss/train': (model_output.loss / self._world_size).item()
        }

        return losses

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
        model.eval()

        valid_results = defaultdict(lambda: 0)
        valid_dl = tqdm.tqdm(valid_dl, desc='Valid step', total=len(valid_dl), position=2)
        for model_input in valid_dl:
            with autocast():
                model_output = model(model_input)

            valid_results['lm_loss/valid'] += model_output.lm_loss
            valid_results['ul_loss/valid'] += model_output.ul_loss
            valid_results['loss/valid'] += model_output.loss

        valid_results = {k: (v / len(valid_dl)).item() for k, v in valid_results.items()}

        return valid_results
