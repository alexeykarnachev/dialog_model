"""
Module for raw jsonl tokenization and serialization into binary dataset on disk.
"""
from itertools import cycle
import json
import logging
from multiprocessing import Manager, Process
from pathlib import Path
import struct
import numpy as np
from more_itertools import chunked

from dialog_model.dialogs_tokenizer import DialogsTokenizer

_logger = logging.getLogger(__name__)

_CODE_TO_DTYPE = {0: np.dtype('uint16'), 1: np.dtype('int32')}
_DTYPE_TO_CODE = {v: k for k, v in _CODE_TO_DTYPE.items()}


class DialogsDatasetSerializer:

    _SERIALIZATION_CHUNK_SIZE = 10000
    _LOG_PROGRESS_EACH_N_DIALOGS = 1000

    def __init__(self, dialogs, out_serialized_dataset_dir, tokenizer_name_or_path, n_workers, max_n_tokens,
                 max_n_utterances):
        self._dialogs = dialogs
        self._out_serialized_dataset_dir = Path(out_serialized_dataset_dir)
        self._tokenizer_name_or_path = tokenizer_name_or_path
        self._n_workers = n_workers
        self._max_n_tokens = max_n_tokens
        self._max_n_utterances = max_n_utterances

        self._out_serialized_dataset_dir.mkdir(exist_ok=False, parents=True)
        self._data_file_path = self._out_serialized_dataset_dir / 'data.bin'
        self._offsets_file_path = self._out_serialized_dataset_dir / 'offsets.bin'
        self._lengths_file_path = self._out_serialized_dataset_dir / 'lengths.bin'
        self._meta_file_path = self._out_serialized_dataset_dir / 'meta.json'
        self._tokenizer_params_file_path = self._out_serialized_dataset_dir / 'tokenizer_params.json'

        sync_manager = Manager()
        self._lock = sync_manager.Lock()
        self._prev_offset = sync_manager.Value('i', 0)
        self._n_samples = sync_manager.Value('i', 0)
        self._dtype_code = sync_manager.Value('i', -1)

        self._log_progress_each_n_dialogs = self._LOG_PROGRESS_EACH_N_DIALOGS - (
            self._LOG_PROGRESS_EACH_N_DIALOGS % n_workers)
        self._tokenizer = DialogsTokenizer(
                self._tokenizer_name_or_path, max_n_tokens=self._max_n_tokens, max_n_utterances=self._max_n_utterances)

    def run(self):
        self._write_initial_offset()

        worker_processes = []
        for worker_id in range(self._n_workers):
            worker_process = Process(target=self._run_worker_job, kwargs={'worker_id': worker_id})
            worker_process.start()
            worker_processes.append(worker_process)

        for worker_process in worker_processes:
            worker_process.join()

        self._write_meta()

    def _run_worker_job(self, worker_id):
        worker_encoded_subdialogs = self._iterate_on_worker_encoded_subdialogs(worker_id)
        for worker_encoded_subdialogs_chunk in chunked(worker_encoded_subdialogs, n=self._SERIALIZATION_CHUNK_SIZE):
            self._write_encoded_dialogs(worker_encoded_subdialogs_chunk)

    def _iterate_on_worker_encoded_subdialogs(self, worker_id):
        worker_dialogs = self._iterate_on_worker_dialogs(worker_id)

        for n_worker_dialogs_done, dialog in enumerate(worker_dialogs, start=1):
            total_dialogs_done = n_worker_dialogs_done * self._n_workers

            if worker_id == 0 and total_dialogs_done % self._log_progress_each_n_dialogs == 0:
                print(f'Dialogs done: {total_dialogs_done}')

            yield from self._tokenizer.iterate_on_encoded_subdialogs(dialog)

    def _iterate_on_worker_dialogs(self, worker_id):
        if worker_id >= self._n_workers:
            raise ValueError(f"Can't run worker job for worker_id={worker_id}. There are only {self._n_workers} "
                             f"workers and worker_id={self._n_workers - 1} is a maximum id.")

        worker_id_cycle = cycle(range(self._n_workers))
        for _worker_id, dialog in zip(worker_id_cycle, self._dialogs):
            if _worker_id == worker_id:
                yield dialog

    def _iterate_on_dialogs(self):
        with open(self._inp_dialogs_file_path) as file:
            for line in file:
                dialog = json.loads(line)
                yield dialog

    def _write_encoded_dialogs(self, encoded_dialogs):
        with self._lock:
            prev_offset = self._prev_offset.value
            data_file = open(self._data_file_path, 'ab')
            offsets_file = open(self._offsets_file_path, 'ab')
            lengths_file = open(self._lengths_file_path, 'ab')
            dtype_code = self._dtype_code.value

            for encoded_dialog in encoded_dialogs:

                dtype = encoded_dialog.dtype
                if dtype_code == -1:
                    dtype_code = _DTYPE_TO_CODE[dtype]
                elif dtype != _CODE_TO_DTYPE[dtype_code]:
                    raise ValueError(
                        f'Expected dtype of token ids: {_CODE_TO_DTYPE[dtype_code]}, but sample has {dtype}')

                n_bytes = data_file.write(encoded_dialog)
                offset = int(prev_offset + n_bytes / encoded_dialog.dtype.itemsize)
                offsets_file.write(struct.pack("<Q", offset))
                lengths_file.write(struct.pack("<H", len(encoded_dialog)))
                prev_offset = offset

            data_file.close()
            offsets_file.close()
            lengths_file.close()

            self._n_samples.value += len(encoded_dialogs)
            self._prev_offset.value = prev_offset
            self._dtype_code.value = dtype_code

    def _write_meta(self):
        with open(self._meta_file_path, 'w') as meta_file:
            meta = {
                'dataset': {
                    'n_samples': self._n_samples.value,
                    'dtype_code': self._dtype_code.value
                },
                'tokenization': {
                    'tokenizer_name_or_path': self._tokenizer_name_or_path,
                    'max_n_tokens': self._max_n_tokens,
                    'max_n_utterances': self._max_n_utterances
                }
            }

            json.dump(meta, meta_file, indent=2)

    def _write_initial_offset(self):
        with open(self._offsets_file_path, 'wb') as offsets_file:
            offsets_file.write(struct.pack("<Q", 0))


def read_offsets(dataset_dir):
    file_path = Path(dataset_dir) / 'offsets.bin'

    return _read_array(file_path, _read_n_samples(dataset_dir) + 1, np.uint64)


def read_lengths(dataset_dir):
    file_path = Path(dataset_dir) / 'lengths.bin'

    return _read_array(file_path, _read_n_samples(dataset_dir), np.uint16)


def open_data_file(dataset_dir):
    file_path = Path(dataset_dir) / 'data.bin'

    return open(file_path, 'rb', buffering=0)


def read_meta(dataset_dir):
    file_path = Path(dataset_dir) / 'meta.json'
    with open(file_path) as meta_file:
        meta = json.load(meta_file)

    return meta


def read_dtype(dataset_dir):
    dtype_code = read_meta(dataset_dir)['dataset']['dtype_code']
    dtype = _CODE_TO_DTYPE[dtype_code]

    return dtype


def load_tokenizer(dataset_dir):
    tokenization_config = read_meta(dataset_dir)['tokenization']
    tokenizer = DialogsTokenizer(**tokenization_config)

    return tokenizer


def _read_n_samples(dataset_dir):
    return read_meta(dataset_dir)['dataset']['n_samples']


def _read_array(file_path, n_elements, dtype):
    file = open(file_path, 'rb', buffering=0)
    array = np.empty(n_elements, dtype=dtype)
    file.readinto(array)

    return array


if __name__ == '__main__':
    from dialogs_data_parsers.flibusta.dialogs_iterator import FlibustaDialogsIterator
    import shutil

    dialogs = FlibustaDialogsIterator('/tmp.jsonl', verbose=False)
    dir_ = Path('/tmp')
    if dir_.exists():
        shutil.rmtree(dir_)
    d = DialogsDatasetSerializer(
        dialogs,
        out_serialized_dataset_dir=dir_,
        tokenizer_name_or_path='sberbank-ai/rugpt3large_based_on_gpt2',
        n_workers=24,
        max_n_tokens=120,
        max_n_utterances=10)
    d.run()
