import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from dialog_model.dataset.serialized_dataset import get_dataloader
from dialog_model.dataset.serializer import load_tokenizer, read_meta
from dialog_model.model import DialogModel
from dialog_model.model import CHECKPOINTS_DIR_NAME, get_pretrained_gpt2_with_lm_head

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Trainer:
    _MASTER_ADDR = 'localhost'
    _MASTER_PORT = '12355'

    def __init__(self, experiment_dir, train_dataset_dir, valid_dataset_dir, gpt2_name_or_path,
                 init_weights_from_checkpoint, worker_batch_size, data_shuffle_seed, freeze_n_layers, learning_rate,
                 n_epochs, validate_each_n_steps, warmup_ratio):
        self._experiment_dir = Path(experiment_dir)
        self._train_dataset_dir = Path(train_dataset_dir)
        self._valid_dataset_dir = Path(valid_dataset_dir)
        self._gpt2_name_or_path = gpt2_name_or_path
        self._init_weights_from_checkpoint = init_weights_from_checkpoint
        self._worker_batch_size = worker_batch_size
        self._data_shuffle_seed = data_shuffle_seed
        self._freeze_n_layers = freeze_n_layers
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._validate_each_n_steps = validate_each_n_steps
        self._warmup_ratio = warmup_ratio

        self._world_size = torch.cuda.device_count()
        self._tokenizer = load_tokenizer(self._train_dataset_dir)

        self._optimizer = None
        self._scaler = None
        self._rank = None
        self._model = None
        self._train_dl = None
        self._valid_dl = None
        self._global_step = None
        self._samples_seen = None
        self._writer = None
        self._model_params = None

        checkpoint_dir = self._experiment_dir / CHECKPOINTS_DIR_NAME
        checkpoint_dir.mkdir(exist_ok=True)
        self._checkpoint_file_path = checkpoint_dir / 'last.ckpt'
        dataset_meta = read_meta(train_dataset_dir)
        with open(self._experiment_dir / 'meta.json', 'w') as file:
            json.dump(dataset_meta, file, indent=2)

    def run(self):
        get_pretrained_gpt2_with_lm_head(self._gpt2_name_or_path)
        mp.spawn(self._train, nprocs=self._world_size, join=True)

    def _save_checkpoint(self):
        checkpoint = {
            'scaler_state_dict': self._scaler.state_dict(),
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'global_step': self._global_step,
            'samples_seen': self._samples_seen,
            'world_size': self._world_size,
            'gpt2_config_dict': self._model.module.gpt2.config.to_dict(),
            'model_params': self._model_params
        }

        torch.save(checkpoint, self._checkpoint_file_path)

    def _load_checkpoint(self):
        checkpoint = torch.load(self._checkpoint_file_path, map_location='cpu')
        checkpoint_world_size = checkpoint['world_size']
        if checkpoint_world_size != self._world_size:
            raise ValueError(f'Checkpoint world size {checkpoint_world_size} does not match with the current '
                             f'world size {self._world_size}.')

        self._scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._global_step = checkpoint['global_step']
        self._samples_seen = checkpoint['samples_seen']
        self._train_dl = self._get_dataloader(is_train=True, samples_offset=self._samples_seen)

    def _load_only_weights_from_checkpoint(self):
        checkpoint = torch.load(self._init_weights_from_checkpoint, map_location='cpu')
        self._model.load_state_dict(checkpoint['model_state_dict'])

    def _train(self, rank):
        _seed_everything(self._data_shuffle_seed)
        self._setup_ddp(rank)
        self._rank = rank
        self._scaler = GradScaler()
        self._model = self._get_model(self._rank)
        self._optimizer = AdamW(params=self._model.parameters(), lr=self._learning_rate)

        self._global_step = 0
        self._samples_seen = 0

        self._train_dl = self._get_dataloader(is_train=True, samples_offset=0)
        self._valid_dl = self._get_dataloader(is_train=False, samples_offset=0)

        steps_per_epoch = len(self._train_dl)
        num_training_steps = steps_per_epoch * self._n_epochs
        num_warmup_steps = self._warmup_ratio * num_training_steps
        self._scheduler = get_linear_schedule_with_warmup(optimizer=self._optimizer,
                                                          num_warmup_steps=num_warmup_steps,
                                                          num_training_steps=num_training_steps)

        if self._checkpoint_file_path.is_file():
            self._load_checkpoint()
        elif self._init_weights_from_checkpoint:
            self._load_only_weights_from_checkpoint()

        while True:
            if self._rank == 0:
                self._writer = self._writer or SummaryWriter(self._experiment_dir / 'tb_logs')
                self._train_dl = tqdm.tqdm(self._train_dl,
                                           desc='Train step',
                                           total=num_training_steps,
                                           position=1,
                                           initial=self._global_step)

            for i_step, model_input in enumerate(self._train_dl):
                train_losses_dict = self._train_step(model_input)

                if rank == 0:
                    self._train_dl.set_postfix({
                        'samples_seen': self._samples_seen,
                        'epoch': self._global_step / steps_per_epoch
                    })
                    self._write_tb_logs(train_losses_dict)
                    self._write_tb_logs({'learning-rate': self._optimizer.param_groups[0]['lr']})
                    self._write_tb_logs({'max_seq_len': model_input.input_ids.size()[1]})

                if self._rank == 0 and self._global_step % self._validate_each_n_steps == 0:
                    valid_loss = self._validate()
                    self._save_checkpoint()
                    self._write_tb_logs({'loss/valid': valid_loss})

                if self._global_step >= num_training_steps:
                    break

        dist.destroy_process_group()

    def _write_tb_logs(self, values_dict):
        for tag, val in values_dict.items():
            self._writer.add_scalar(tag=tag, scalar_value=val, global_step=self._global_step)

    def _train_step(self, model_input):
        self._model.train()
        self._optimizer.zero_grad()

        with autocast():
            model_output = self._model(model_input)

        loss = model_output.loss
        lm_loss = model_output.lm_loss
        cls_loss = model_output.cls_loss

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()

        dist.all_reduce(loss)
        dist.all_reduce(lm_loss)
        dist.all_reduce(cls_loss)

        loss = loss.item() / self._world_size
        lm_loss = lm_loss.item() / self._world_size
        cls_loss = cls_loss.item() / self._world_size

        samples_seen = torch.tensor(len(model_input.input_ids), device=self._rank)
        dist.all_reduce(samples_seen)

        self._samples_seen += samples_seen.item()
        self._global_step += 1
        self._scheduler.step()

        losses_dict = {'loss/train': loss, 'lm_loss/train': lm_loss, 'cls_loss/train': cls_loss}

        return losses_dict

    def _setup_ddp(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=self._world_size)

    def _get_model(self, rank):
        model_params = {
            'gpt2_name_or_path': self._gpt2_name_or_path,
            'vocab_size': self._tokenizer.vocab_size,
            'n_classes': 2,
            'end_of_speaker_2_token_id': self._tokenizer.end_of_speaker_2_token_id,
            'cls_loss_weight': 0.25
        }

        model = DialogModel(**model_params)
        model = model.to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])
        self._model_params = model_params

        return model

    def _get_dataloader(self, is_train, samples_offset):
        return get_dataloader(dataset_dir=self._train_dataset_dir if is_train else self._valid_dataset_dir,
                              distractor_p=0.5,
                              batch_size=self._worker_batch_size,
                              num_workers=4,
                              sort_chunk_size=self._worker_batch_size * 10,
                              samples_offset=samples_offset,
                              data_shuffle_seed=self._data_shuffle_seed,
                              is_distributed=is_train,
                              pad_token_id=self._tokenizer.pad_token_id,
                              end_of_speaker_1_token_id=self._tokenizer.end_of_speaker_1_token_id,
                              end_of_speaker_2_token_id=self._tokenizer.end_of_speaker_2_token_id)

    @torch.no_grad()
    def _validate(self):
        self._model.eval()

        loss = 0
        valid_dl = tqdm.tqdm(self._valid_dl, desc='Valid step', total=len(self._valid_dl), position=2)
        for model_input in valid_dl:
            with autocast():
                model_output = self._model(model_input)
                loss_on_step = model_output.loss
                loss += loss_on_step.item()

        loss /= len(valid_dl)

        return loss


def _seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
