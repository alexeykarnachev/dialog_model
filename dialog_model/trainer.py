import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from dialog_model.dataset.serialization import load_tokenizer
from dialog_model.dataset.serialized_dataset import get_dataloader
from dialog_model.language_generator.generator import LanguageGenerator
from dialog_model.model_io import get_pretrained_gpt2_with_lm_head

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Trainer:
    _MASTER_ADDR = 'localhost'
    _MASTER_PORT = '12355'

    def __init__(
            self,
            experiment_dir,
            train_dataset_dir,
            valid_dataset_dir,
            gpt2_name_or_path,
            worker_batch_size,
            data_shuffle_seed,
            freeze_n_layers,
            learning_rate,
            n_epochs,
            validate_each_n_steps,
            warmup_ratio
    ):
        self._experiment_dir = Path(experiment_dir)
        self._train_dataset_dir = train_dataset_dir
        self._valid_dataset_dir = valid_dataset_dir
        self._gpt2_name_or_path = gpt2_name_or_path
        self._worker_batch_size = worker_batch_size
        self._data_shuffle_seed = data_shuffle_seed
        self._freeze_n_layers = freeze_n_layers
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._validate_each_n_steps = validate_each_n_steps
        self._warmup_ratio = warmup_ratio

        self._world_size = torch.cuda.device_count()
        self._tokenizer = load_tokenizer(dataset_dir=self._train_dataset_dir)

        self._optimizer = None
        self._scaler = None
        self._rank = None
        self._model = None
        self._train_dl = None
        self._valid_dl = None
        self._global_step = None
        self._samples_seen = None

    def run(self):
        get_pretrained_gpt2_with_lm_head(self._gpt2_name_or_path)
        load_tokenizer(self._train_dataset_dir)
        mp.spawn(self._train, nprocs=self._world_size, join=True)

    def _train(self, rank):
        _seed_everything(self._data_shuffle_seed)
        self._setup_ddp(rank)
        self._rank = rank
        self._scaler = GradScaler()
        self._model = self._get_model(self._rank)
        self._optimizer = AdamW(params=self._model.parameters(), lr=self._learning_rate)
        self._train_dl = self._get_dataloader(is_train=True, samples_offset=0)
        self._valid_dl = self._get_dataloader(is_train=False, samples_offset=0)
        self._global_step = 0
        self._samples_seen = 0

        num_training_steps = len(self._train_dl) * self._n_epochs
        num_warmup_steps = self._warmup_ratio * num_training_steps
        self._scheduler = get_linear_schedule_with_warmup(
            optimizer=self._optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        if self._rank == 0:
            self._writer = SummaryWriter(self._experiment_dir / 'tb_logs')
            self._train_dl = tqdm.tqdm(
                self._train_dl, desc='Train step', total=len(self._train_dl), position=1, initial=self._global_step)

        for i_epoch in range(self._n_epochs):
            for i_step, (token_ids, lm_labels) in enumerate(self._train_dl):

                if self._rank == 0 and i_step and i_step % self._validate_each_n_steps == 0:
                    self._generate()
                    valid_loss = self._validate()
                    self._write_tb_logs({'loss/valid': valid_loss})

                train_loss = self._train_step(token_ids, lm_labels)

                if rank == 0:
                    self._train_dl.set_postfix({'loss/train': train_loss, 'samples_sees': self._samples_seen})
                    self._write_tb_logs({'loss/train': train_loss})
                    self._write_tb_logs({'learning-rate': self._optimizer.param_groups[0]['lr']})
                    self._write_tb_logs({'max_seq_len': token_ids.size()[1]})
                    self._write_tb_logs({'epoch': i_epoch})

        dist.destroy_process_group()

    def _write_tb_logs(self, values_dict):
        for tag, val in values_dict.items():
            self._writer.add_scalar(tag=tag, scalar_value=val, global_step=self._global_step)

    def _train_step(self, token_ids, lm_labels):
        self._model.train()
        self._optimizer.zero_grad()

        with autocast():
            loss, *_ = self._model(token_ids, labels=lm_labels)

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        dist.all_reduce(loss)
        loss = loss.item() / self._world_size

        samples_seen = torch.tensor(len(token_ids), device=self._rank)
        dist.all_reduce(samples_seen)
        self._samples_seen += samples_seen.item()
        self._global_step += 1
        self._scheduler.step()

        return loss

    def _setup_ddp(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=self._world_size)

    def _get_model(self, rank):
        model = get_pretrained_gpt2_with_lm_head(
            self._gpt2_name_or_path,
            vocab_size=self._tokenizer.vocab_size,
            freeze_n_layers=self._freeze_n_layers,
            reference_token_id=self._tokenizer.reference_token_id)
        model = model.to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])

        return model

    def _get_dataloader(self, is_train, samples_offset):
        return get_dataloader(
            dataset_dir=self._train_dataset_dir if is_train else self._valid_dataset_dir,
            batch_size=self._worker_batch_size,
            num_workers=4,
            sort_chunk_size=self._worker_batch_size * 10,
            samples_offset=samples_offset,
            data_shuffle_seed=self._data_shuffle_seed,
            is_distributed=is_train,
            pad_token_id=self._tokenizer.pad_token_id
        )

    @torch.no_grad()
    def _validate(self):
        self._model.eval()

        loss = 0
        valid_dl = tqdm.tqdm(self._valid_dl, desc='Valid step', total=len(self._valid_dl), position=2)
        for token_ids, lm_labels in valid_dl:
            with autocast():
                loss_on_step, *_ = self._model(token_ids, labels=lm_labels)
                loss += loss_on_step.item()

        loss /= len(valid_dl)

        return loss

    def _generate(self):
        out_dir = self._experiment_dir / 'generated'
        out_dir.mkdir(exist_ok=True, parents=False)
        config_file_path = out_dir / 'config.json'
        generated_file_path = out_dir / 'generated.jsonl'

        if config_file_path.is_file():
            with open(config_file_path) as file:
                config = json.load(file)
                generator_params = config['generator_params']
                dialog = config['dialog']
        else:
            generator_params = {
                'num_return_sequences': 4,
                'repetition_penalty': 3,
                'temperature': 0.73,
                'top_k': 100,
                'top_p': 1.0
            }
            dialog = ['Привет, как дела?', 'Нормально, сам как?', 'Я тоже хорошо. Расскажи о себе.']
            config = {'generator_params': generator_params, 'dialog': dialog}

            with open(config_file_path, 'w') as file:
                json.dump(config, file, indent=2, ensure_ascii=False)

        try:
            generator = LanguageGenerator(self._model.module, self._tokenizer)
            candidates = generator(dialog=dialog, **generator_params)
            payload = {'generator_params': generator_params, 'dialog': dialog, 'candidates': candidates}
        except:
            tb = traceback.format_exc()
            payload = {'exception': tb}

        with open(generated_file_path, 'a') as file:
            payload = json.dumps(payload, ensure_ascii=False, indent=2)
            file.write(payload)
            file.write('\n')


def _seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
