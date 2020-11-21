import numpy as np
import pytest

from dialog_model.dataset.serialization import _write_data, _write_index
from dialog_model.dataset.serialized_dataset import SerializedDataset


def _generate_token_ids(size):
    return np.random.randint(low=0, high=100, size=size, dtype=np.uint16)


def _iterate_on_token_ids(n_samples, min_size, max_size):
    for _ in range(n_samples):
        size = np.random.randint(low=min_size, high=max_size + 1)
        yield _generate_token_ids(size)


@pytest.mark.parametrize('n_samples,min_size,max_size', [
    (1, 1, 1), (1, 1, 10), (10, 10, 10), (1000, 1, 1), (1000, 100, 100)
])
def test_build_dataset(n_samples, min_size, max_size, tmpdir):
    token_ids_expected = list(_iterate_on_token_ids(n_samples=n_samples, min_size=min_size, max_size=max_size))
    offsets, lengths, dtype = _write_data(tmpdir, token_ids_iter=token_ids_expected)
    _write_index(tmpdir, offsets=offsets, lengths=lengths, dtype=dtype)
    dataset = SerializedDataset(tmpdir)
    token_ids_actual = list(dataset)

    for actual, expected in zip(token_ids_actual, token_ids_expected):
        np.testing.assert_equal(actual, expected)
