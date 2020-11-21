import logging
import math
from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from dialog_model.data_structures import DialogModelOutput, DialogModelInput
from dialog_model.dataset.serialization import read_tokenizer, read_number_of_samples
from dialog_model.dataset.serialized_dataset import SerializedDataset
from dialog_model.modelling.model import DialogModel
from dialog_model.modelling.model_io import get_pretrained_gpt2_lm_head

_logger = logging.getLogger(__name__)


class DialogPLModule(pl.LightningModule):
    def __init__(
            self,
            gpt2_lm_head_name_or_path,
            unlikelihood_alpha,
            train_dataset_dir,
            valid_dataset_dir,
            batch_size,
            shuffle_with_seed,
            learning_rate,
            warmup_steps_ratio
    ):
        super().__init__()

        self.gpt2_lm_head_name_or_path = gpt2_lm_head_name_or_path
        self.unlikelihood_alpha = unlikelihood_alpha
        self.train_dataset_dir = train_dataset_dir
        self.valid_dataset_dir = valid_dataset_dir
        self.batch_size = batch_size
        self.shuffle_with_seed = shuffle_with_seed
        self.learning_rate = learning_rate
        self.warmup_steps_ratio = warmup_steps_ratio

        self.model: Optional[DialogModel] = None
        self.samples_offset = 0
        self.save_hyperparameters()

    def prepare_data(self):
        get_pretrained_gpt2_lm_head(self._gpt2_lm_head_name_or_path)

    def setup(self, stage: str):
        gpt2_lm_head = get_pretrained_gpt2_lm_head(self._gpt2_lm_head_name_or_path)

        self.model = DialogModel(gpt2_lm_head=gpt2_lm_head, unlikelihood_alpha=self.unlikelihood_alpha)

        self.hparams['gpt2_config'] = gpt2_lm_head.config.__dict__

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(is_train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(is_train=False)

    def _get_dataloader(self, is_train) -> DataLoader:
        dataset_dir = self.train_dataset_dir if is_train else self.valid_dataset_dir
        dataset = SerializedDataset(dataset_dir=dataset_dir)
        tokenizer = read_tokenizer(dataset_dir)
        dataloader = dataset.get_dataloader(
            batch_size=self.batch_size,
            num_workers=4 if is_train else 1,
            samples_offset=self.samples_offset if is_train else 0,
            shuffle_with_seed=self.shuffle_with_seed,
            is_distributed=is_train,
            end_of_prefix_token_id=tokenizer.end_of_prefix_token_id,
            pad_token_id=tokenizer.pad_token_id,
            sort_chunk_size=self.batch_size * 1000)

        return dataloader

    def forward(self, model_input: DialogModelInput) -> DialogModelOutput:
        model_output: DialogModelOutput = self.model(model_input)
        return model_output

    def training_step(self, model_input, batch_idx):
        model_output = self.forward(model_input)

        self.log('loss/Train', model_output.loss, prog_bar=True, sync_dist=True)
        self.log('lm_loss/Train', model_output.lm_loss, prog_bar=True, sync_dist=True)
        self.log('ul_loss/Train', model_output.ul_loss, prog_bar=True, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        lr = torch.tensor(lr, device=self.model.device)
        self.log(f'learning_rate', lr, prog_bar=True, sync_dist=True)

        return model_output.loss

    def validation_step(self, model_input, batch_idx):
        return self.forward(model_input)

    def validation_epoch_end(self, model_outputs: Sequence[DialogModelOutput]):
        lm_losses, ul_losses, losses, *_ = zip(*model_outputs)

        metrics = {
            'lm_loss/Valid': torch.mean(lm_losses),
            'ul_loss/Valid': torch.mean(ul_losses),
            'loss/Valid': torch.mean(losses)
        }
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.learning_rate)
        num_samples = read_number_of_samples(self.train_dataset_dir)
        num_steps = self.trainer.max_epochs * math.ceil(num_samples / (self.batch_size * self.trainer.world_size))
        num_warmup_steps = int(num_steps * self.warmup_steps_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps)

        lr_scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}

        _logger.info(f'Total training steps: {num_steps}, Warmup steps: {num_warmup_steps}')

        return [optimizer], [lr_scheduler]

    def on_load_checkpoint(self, checkpoint):
        self._validate_checkpoint(checkpoint)
        self.samples_offset = checkpoint['n_samples_seen']

    def _validate_checkpoint(self, checkpoint):
        """Checks if the loaded checkpoint is consistent with the actual trainer parameters."""
        current_world_size = self.trainer.world_size
        checkpoint_world_size = checkpoint['world_size']
        if current_world_size != checkpoint_world_size:
            raise ValueError(f'You current world size ({current_world_size}) must be equal to the checkpoint world '
                             f'size ({checkpoint_world_size})')

    def on_save_checkpoint(self, checkpoint) -> None:
        world_size = self.trainer.world_size
        n_samples_seen = self.batch_size * world_size * checkpoint['global_step']

        checkpoint['n_samples_seen'] = n_samples_seen
        checkpoint['world_size'] = world_size
        checkpoint['epoch'] = self.trainer.current_epoch

        _logger.info(f'Data samples seen so far: {n_samples_seen}.')
