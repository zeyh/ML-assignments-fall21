import numpy as np
import random, string
import csv
from .test_utils import write_random_csv_file

def test_load_data():
    from src import load_data

    n_features = np.random.randint(5, 20)
    n_samples = np.random.randint(50, 150)
    features, targets, attribute_names = write_random_csv_file(n_features, n_samples)

    _features, _targets, _attribute_names = load_data('tests/test.csv')
    assert attribute_names == _attribute_names
    assert np.allclose(features, _features) and np.allclose(targets.flatten(), _targets.flatten())

def test_train_test_split():
    from src import train_test_split

    n_features = np.random.randint(5, 20)
    n_samples = np.random.randint(50, 150)
    features, targets, attribute_names = write_random_csv_file(n_features, n_samples)
    fraction = np.random.rand()

    output = train_test_split(features, targets, fraction)
    expected_train_size = int(n_samples * fraction)
    expected_test_size = n_samples - expected_train_size

    for o in output:
        assert o.shape[0] == expected_train_size or o.shape[0] == expected_test_size

    full_output = train_test_split(features, targets, 1.0)

    for o in full_output:
        assert o.shape[0] == n_samples
