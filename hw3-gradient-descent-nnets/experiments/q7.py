#copied and modified from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py

import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

#copied from metrics.py
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

# ! question 7(a)
hidden_layer_sizes = [1, 4, 16, 64, 256]
# ! question 7(b)
activation_fcn = ["identity", "logistic", "tanh", "relu"]
# ! question 7(c)
regularizations = [1,0.1,0.01,0.001,0.0001]

currTestingHParam = hidden_layer_sizes # CHANGE THIS!!
accuracy_dict = {}
for testing_val in currTestingHParam: #testing out different values
    accuracy_log = []
    for i in tqdm(range(10)): #iterate 10 times for each hyperparameter value
        mlp = MLPClassifier(
            hidden_layer_sizes=(testing_val,), # ! question 7(a) - default 50
            max_iter=10,
            alpha=1e-4, # ! question 7(c) - default 1e-4
            solver="sgd",
            verbose=10,
            learning_rate_init=0.1,
            activation="relu", # ! question 7(b) - default relu
            shuffle = True,
            random_state=None,
        )
        # this example won't converge because of CI's time constraints, so we catch the
        # warning and are ignore it here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            mlp.fit(X_train, y_train)

        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Test set score: %f" % mlp.score(X_test, y_test))
        acc_score = accuracy(y_test, mlp.predict(X_test))
        print("Accuracy: %f" % acc_score)
        accuracy_log.append(acc_score)
    accuracy_dict[testing_val] = accuracy_log
    
print(">>>>>>>",accuracy_dict)
for item in accuracy_dict.items():
    print("Hyperparameter Value: ",item[0])
    # print("Values: ",item[1])
    print("std: ", np.std(item[1]))
    print("mean: ", np.mean(item[1]))

