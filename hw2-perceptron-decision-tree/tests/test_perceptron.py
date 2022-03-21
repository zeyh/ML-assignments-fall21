import numpy as np 
from src import Perceptron, transform_data, load_data

def test_perceptron():
    features, targets, _ = load_data('data/parallel_lines.csv')
    p = Perceptron(max_iterations=100)
    
    p.fit(features, targets)
    targets_hat = p.predict(features)

    # your perceptron should fit this dataset perfectly
    assert np.allclose(targets, targets_hat)

def test_transform_data():
    features, targets, _ = load_data('data/transform_me.csv')
    features_transform = transform_data(features)

    p = Perceptron(max_iterations=100)

    p.fit(features_transform, targets)
    targets_hat = p.predict(features_transform)

    # your perceptron should fit this dataset perfectly after transforming the data
    assert np.allclose(targets, targets_hat)
