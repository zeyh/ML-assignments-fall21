#copied and modified from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py

import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

def accuracy(ground_truth, predictions):
    """
    Reports the classification accuracy.

    Arguments:
        ground_truth - (np.array) A 1D numpy array of length N. The true class
            labels.
        predictions - (np.array) A 1D numpy array of length N. The class labels
            predicted by the model.
    Returns:
        accuracy - (float) The accuracy of the predictions.
    """
    return np.mean(ground_truth == predictions)

# Load data from https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0

# rescale the data, use the traditional train/test split
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=None)

mlp = MLPClassifier(
    hidden_layer_sizes=(1,), # ! question 7(a) - default 50
    max_iter=10,
    alpha=0.1, # ! question 7(c) - default 1e-4
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.1,
    activation="relu", # ! question 7(b) - default relu
    shuffle = True,
)

# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
# fig.savefig('hiddenLayer.png', dpi=fig.dpi)