from os import SEEK_CUR
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. The data
    in `data/transform_me.csv` is not linearly separable; you should implement
    this function such that the perceptron algorithm can achieve perfect performance.
    This function should only apply for this specific dataset -- it should not be
    used for the other datasets in this assignment.
    Refer to `tests/test_perceptron.py` for how this function will be used.

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    
    #map a cartesian coordinate to polar coordinate r, theta
    features_transformed = features
    for d, feature in enumerate(features):
        features_transformed[d][0] = np.sqrt(feature[0]**2+feature[1]**2)
        features_transformed[d][1] = np.arctan(feature[1]/feature[0])
    return features_transformed

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single layer
        perceptron is an algorithm for supervised learning of a binary classifier. The
        idea is to draw a linear decision boundary in the space that separates the
        points in the space into two partitions. Points on one side of the line are one
        class and points on the other side are the other class.

        To simplify comparisons and allow for reproducibility, the Perceptron's
        initial weights should be set to 1 rather than randomly initialized.

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.name = "perceptron"
        self.converge = False
        self.iteration = 1
        

    
    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets. Note
        that the csv datasets use class labels 0 and 1, but the Perceptron
        algorithm requires classes (-1 or 1). This function should terminate
        either after convergence (the decision boundary does not change between iterations)
        or after max_iterations (defaults to 200) iterations are done. Here is pseudocode
        for the perceptron learning algorithm:

        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that in the above pseudocode, label_for_example is either -1 or 1.

        Note that your weights vector should include a bias term, so if your data has
        two features your should initialize weights as (w_0=1, w_1=1, w_2=1).

        Use only numpy to implement this algorithm. 

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        # initialize weights to 1
        self.dim = features.shape[1]
        self.N = features.shape[0]
        self.weights = np.ones(self.dim+1)
        self.eta = 0.85 #learning rate
        self.bias = float(1)
        
        # Relabel targets to (-1, 1) from (0, 1)
        targets = np.where(targets == 0, -1, targets)
        
        # #attach a column of 1s as bias for w_0
        # features = np.insert(features, 0, 1, axis=1)
        
        for iter in range(self.max_iterations):
            counter_pt = 0
            for i, feature in enumerate(features): #iterate one time
                feature_tmp = np.append(feature, self.bias)
                target_hat = -1 if self.weights.dot(feature_tmp) <= 0 else 1
                target = targets[i]
                if target != target_hat:
                    # print(target_hat, target)
                    for d in range(self.dim):
                        self.weights[d] = self.weights[d] + self.eta * feature[d] * target
                    self.weights[-1] = self.weights[-1] + self.eta * target
                    counter_pt += 1
            # print("misclassification: ", counter_pt, self.weights)        
            if counter_pt == 0:
                self.converge = True
                return
        self.iteration = iter+1
            


    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        NOTE: to comport with the other models in this homework,
        you should output these predictions as 0 or 1, not as -1 or 1.
        This can be done with `np.where(predictions == -1, 0, predictions)`
        as shown below.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        targets_hat = []
        for i, feature in enumerate(features): #iterate one time
            feature_tmp = np.append(feature, self.bias)
            target_hat = -1 if self.weights.dot(feature_tmp) <= 0 else 1
            targets_hat.append(target_hat)
            
        targets_hat = np.asarray(targets_hat)
        targets_hat = np.where(targets_hat == -1, 0, targets_hat)

        return targets_hat
    


if __name__ == "__main__":
    from data import load_data
    from visualize import plot_decision_regions
    from metrics import *
    # print("...testing....")
    path = 'data/xor.csv'
    features, targets, _ = load_data(path)
    print("data size: ",features.shape, targets.shape)
    p = Perceptron(max_iterations=200)
    p.fit(features, targets)
    targets_hat = p.predict(features)
    print(np.allclose(targets, targets_hat))
    plot_decision_regions(features, targets, p, path)
    # confusion_matrix = compute_confusion_matrix(targets, targets_hat)
    # accuracy = compute_accuracy(targets, targets_hat)
    # precision, recall = compute_precision_and_recall(targets, targets_hat)
    # f1_measure = compute_f1_measure(targets, targets_hat)
    
    # print(accuracy)



    #? =========================================================
    # features, targets, _ = load_data('data/transform_me.csv')
    # features_transform = transform_data(features)
    # p.fit(features, targets)
    # targets_hat = p.predict(features_transform)
    # plot_decision_regions(features, targets, p)
    


    

    
    