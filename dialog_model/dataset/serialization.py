import json
import struct
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from more_itertools import chunked

from dialog_model.raw_dialogs.pikabu_dialogs_iterator import PikabuDialogsIterator
from dialog_model.dialogs_tokenizer import DialogsTokenizer

_DATA_FILE_NAME = 'data.bin'
_INDEX_FILE_NAME = 'data.idx'
_TOKENIZER_PARAMS_FILE_NAME = 'tokenizer_params.json'
_CODE_TO_DTYPE = {
    0: np.dtype('uint16'),
    1: np.dtype('int32')
}
_DTYPE_TO_CODE = {v: k for k, v in _CODE_TO_DTYPE.items()}


def open_data_file(dataset_dir):
    return open(dataset_dir / _DATA_FILE_NAME, 'rb', buffering=0)


def open_index_file(dataset_dir):
    return open(dataset_dir / _INDEX_FILE_NAME, 'rb', buffering=0)


def read_index(dataset_dir):
    dataset_dir = Path(dataset_dir)
    index_file = open_index_file(dataset_dir)
    n_samples, dtype_code = struct.unpack("<QQ", index_file.read(16))
    dtype = _CODE_TO_DTYPE[dtype_code]
    offsets = _read_array(index_file, n_elements=n_samples + 1, dtype=np.uint64)
    lengths = _read_array(index_file, n_elements=n_samples, dtype=np.uint16)
    index_file.close()

    return offsets, lengths, dtype


def read_number_of_samples(dataset_dir):
    index_file = open_index_file(dataset_dir)
    n_samples, _ = struct.unpack("<QQ", index_file.read(16))
    index_file.close()

    return n_samples


def build_dataset(
        dialogs_file_path,
        out_dir,
        tokenization_chunk_size,
        tokenizer_name_or_path,
        max_n_tokens
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dialogs = PikabuDialogsIterator(dialogs_file_path)
    tokenizer_params = {'tokenizer_name_or_path': tokenizer_name_or_path, 'max_n_tokens': max_n_tokens}
    tokenizer = DialogsTokenizer(**tokenizer_params)
    token_ids_iter = _iterate_on_token_ids(
        dialogs=dialogs, tokenizer=tokenizer, tokenization_chunk_size=tokenization_chunk_size)

    offsets, lengths, dtype = _write_data(out_dir, token_ids_iter=token_ids_iter)
    _write_index(out_dir, offsets=offsets, lengths=lengths, dtype=dtype)
    _write_tokenizer_params(out_dir, tokenizer_params=tokenizer_params)


def load_tokenizer(dataset_dir) -> DialogsTokenizer:
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / _TOKENIZER_PARAMS_FILE_NAME) as file:
        tokenizer_params = json.load(file)

    tokenizer = DialogsTokenizer(**tokenizer_params)

    return tokenizer


def _write_index(out_dir, offsets, lengths, dtype):
    index_file = open(out_dir / _INDEX_FILE_NAME, 'wb')

    n_samples = len(lengths)
    dtype_code = _DTYPE_TO_CODE[dtype]
    index_file.write(struct.pack("<QQ", n_samples, dtype_code))
    _write_array(index_file, offsets, dtype=np.uint64)
    _write_array(index_file, lengths, dtype=np.uint16)

    index_file.close()


def _write_data(out_dir, token_ids_iter: Iterable):
    data_file = open(out_dir / _DATA_FILE_NAME, 'wb')

    lengths = []
    dtype = None
    offsets = [0]
    for i_sample, token_ids in enumerate(token_ids_iter, start=1):
        if dtype is None:
            dtype = token_ids.dtype
        elif token_ids.dtype != dtype:
            raise ValueError(f'Expected dtype of token ids: {token_ids.dtype}, but sample has {dtype}')

        n_bytes = data_file.write(token_ids)
        offset = int(offsets[-1] + n_bytes / dtype.itemsize)

        offsets.append(offset)
        lengths.append(len(token_ids))

    data_file.close()

    return offsets, lengths, dtype


def _write_tokenizer_params(out_dir, tokenizer_params):
    with open(out_dir / _TOKENIZER_PARAMS_FILE_NAME, 'w') as file:
        json.dump(tokenizer_params, file, indent=2)


def _read_array(file, n_elements, dtype):
    a = np.empty(n_elements, dtype=dtype)
    file.readinto(a)
    return a


def _write_array(file, arr, dtype):
    file.write(np.array(arr, dtype=dtype))


def _iterate_on_token_ids(dialogs: Iterable[Sequence[str]], tokenizer: DialogsTokenizer, tokenization_chunk_size):
    for dialogs_chunk in chunked(dialogs, n=tokenization_chunk_size):
        yield from tokenizer.encode(dialogs_chunk)
