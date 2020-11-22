import logging
import os
from functools import partial

import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel
from transformers import AdamW

from dialog_model.dataset.serialization import load_tokenizer
from dialog_model.dataset.serialized_dataset import get_dataloader
from dialog_model.modelling.model import DialogModel
from dialog_model.modelling.model_io import get_pretrained_gpt2_lm_head

_logger = logging.getLogger(__name__)


def train(
        rank,
        world_size,
        train_dataset_dir,
        valid_dataset_dir,
        gpt2_name_or_path,
        unlikelihood_alpha,
        worker_batch_size,
        data_shuffle_seed,
        learning_rate,
        n_epochs
):
    _logger.info(f'Running ddp training on rank: {rank}.')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tokenizer = load_tokenizer(dataset_dir=train_dataset_dir)
    gpt2 = get_pretrained_gpt2_lm_head(gpt2_name_or_path)
    model = DialogModel(gpt2_lm_head=gpt2, unlikelihood_alpha=unlikelihood_alpha).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    _get_dataloader = partial(
        get_dataloader,
        batch_size=worker_batch_size,
        num_workers=1,
        sort_chunk_size=worker_batch_size * 1000,
        samples_offset=0,
        data_shuffle_seed=data_shuffle_seed,
        is_distributed=True,
        pad_token_id=tokenizer.pad_token_id,
        end_of_prefix_token_id=tokenizer.end_of_prefix_token_id
    )

    train_dataloader = _get_dataloader(dataset_dir=train_dataset_dir)
    valid_dataloader = _get_dataloader(dataset_dir=valid_dataset_dir)
    optimizer = AdamW(params=model.parameters(), lr=learning_rate)

    for i_epoch in range(n_epochs):
        for i_epoch_step, model_input in enumerate(train_dataloader):
            optimizer.zero_grad()
            model_output = model(model_input)
            model_output.loss.backward()
            optimizer.step()

            print(f'Epoch: {i_epoch}, Step: {i_epoch_step}')

    dist.destroy_process_group()



