"""
Module for raw jsonl tokenization and serialization into binary file on disk.
"""
from itertools import cycle
import json
from multiprocessing import Manager, Process
from pathlib import Path
import struct
import logging
from dialog_model.dialogs_tokenizer import DialogsTokenizer

_logger = logging.getLogger(__name__)


class DialogsDatasetSerializer:
    def __init__(self, inp_dialogs_file_path, out_serialized_dataset_dir, tokenizer_name_or_path, n_workers,
                 max_n_tokens, max_n_utterances):
        self._inp_dialogs_file_path = inp_dialogs_file_path
        self._out_serialized_dataset_dir = Path(out_serialized_dataset_dir)
        self._tokenizer_name_or_path = tokenizer_name_or_path
        self._n_workers = n_workers
        self._max_n_tokens = max_n_tokens
        self._max_n_utterances = max_n_utterances

        self._out_serialized_dataset_dir.mkdir(exist_ok=False, parents=True)
        self._data_file_path = self._out_serialized_dataset_dir / 'data.bin'
        self._offsets_file_path = self._out_serialized_dataset_dir / 'offsets.bin'
        self._lengths_file_path = self._out_serialized_dataset_dir / 'lengths.bin'
        self._tokenizer_params_file_path = self._out_serialized_dataset_dir / 'tokenizer_params.json'

        sync_manager = Manager()
        self._lock = sync_manager.Lock()
        self._prev_offset = sync_manager.Value('i', 0)
        self._n_samples = sync_manager.Value('i', 0)
        self._n_dialogs = sync_manager.Value('i', 0)

    def run(self):
        worker_processes = []
        for worker_id in range(self._n_workers):
            worker_process = Process(target=self._run_worker_job, kwargs={'worker_id': worker_id})
            worker_process.start()
            worker_processes.append(worker_process)

        for worker_process in worker_processes:
            worker_process.join()

    def _run_worker_job(self, worker_id):
        worker_encoded_subdialogs = self._iterate_on_worker_encoded_subdialogs(worker_id)
        for worker_encoded_subdialog in worker_encoded_subdialogs:
            self._write_sample(worker_encoded_subdialog)

    def _iterate_on_worker_encoded_subdialogs(self, worker_id):
        worker_dialogs = self._iterate_on_worker_dialogs(worker_id)
        tokenizer = DialogsTokenizer(
            self._tokenizer_name_or_path, max_n_tokens=self._max_n_tokens, max_n_utterances=self._max_n_utterances)
        for dialog in worker_dialogs:
            yield from tokenizer.iterate_on_encoded_subdialogs(dialog)

    def _iterate_on_worker_dialogs(self, worker_id):
        if worker_id >= self._n_workers:
            raise ValueError(f"Can't run worker job for worker_id={worker_id}. There are only {self._n_workers} "
                             f"workers and worker_id={self._n_workers - 1} is a maximum id.")

        dialogs = self._iterate_on_dialogs()
        worker_id_cycle = cycle(range(self._n_workers))
        for _worker_id, dialog in zip(worker_id_cycle, dialogs):
            if _worker_id == worker_id:
                self._n_dialogs.value += 1
                # _logger.info(f'Dialogs done: {self._n_dialogs.value}')
                print(self._n_dialogs.value)
                yield dialog

    def _iterate_on_dialogs(self):
        with open(self._inp_dialogs_file_path) as file:
            for line in file:
                dialog = json.loads(line)
                yield dialog

    def _write_sample(self, sample):
        with self._lock:
            with open(self._data_file_path, 'ab') as data_file:
                n_bytes = data_file.write(sample)

            offset = int(self._prev_offset.value + n_bytes / sample.dtype.itemsize)
            with open(self._offsets_file_path, 'ab') as offsets_file:
                offsets_file.write(struct.pack("<Q", offset))

            with open(self._lengths_file_path, 'ab') as lengths_file:
                lengths_file.write(struct.pack("<H", len(sample)))

            self._prev_offset.value = offset
            self._n_samples.value += 1


if __name__ == '__main__':
    d = DialogsDatasetSerializer(
        '/ssd_1/data/flibusta/filtered_dialogs.jsonl',
        out_serialized_dataset_dir='/root/dialog_model/dialog_model/dataset/tmp',
        tokenizer_name_or_path='gpt2',
        n_workers=24,
        max_n_tokens=100,
        max_n_utterances=10)
    d.run()
