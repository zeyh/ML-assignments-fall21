import numpy as np
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import L1Regularization, L2Regularization
# from your_code import accuracy

# from loss import HingeLoss, SquaredLoss
# from regularization import L1Regularization, L2Regularization
# from metrics import accuracy

class GradientDescent:
    """
    This is a linear classifier similar to the one you implemented in the
    linear regressor homework. This is the classification via regression
    case. The goal here is to learn some hyperplane, y = w^T x + b, such that
    when features, x, are processed by our model (w and b), the result is
    some value y. If y is in [0.0, +inf), the predicted classification label
    is +1 and if y is in (-inf, 0.0) the predicted classification label is
    -1.

    The catch here is that we will not be using the closed form solution,
    rather, we will be using gradient descent. In your fit function you
    will determine a loss and update your model (w and b) using gradient
    descent. More details below.

    Arguments:
        loss - (string) The loss function to use. Either 'hinge' or 'squared'.
        regularization - (string or None) The type of regularization to use.
            One of 'l1', 'l2', or None. See regularization.py for more details.
        learning_rate - (float) The size of each gradient descent update step.
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """
    def __init__(self, loss, regularization=None,
                 learning_rate=0.01, reg_param=0.05):
        self.learning_rate = learning_rate

        # Select regularizer
        if regularization == 'l1':
            self.regularizer = L1Regularization(reg_param)
        elif regularization == 'l2':
            self.regularizer = L2Regularization(reg_param)
        elif regularization is None:
            self.regularizer = None
        else:
            raise ValueError(
                'Regularizer {} is not defined'.format(regularization))

        # Select loss function
        if loss == 'hinge':
            self.loss = HingeLoss(self.regularizer)
        elif loss == 'squared':
            self.loss = SquaredLoss(self.regularizer)
        elif loss == '0-1':
            self.loss = ZeroOneLoss(self.regularizer)
        else:
            raise ValueError('Loss function {} is not defined'.format(loss))

        self.model = None

    def fit(self, features, targets, batch_size=None, max_iter=1000, isPrinting=False, isMinibatch=False, test_features=[], test_targets=[]):
        """
        Fits a gradient descent learner to the features and targets. The
        pseudocode for the fitting algorithm is as follow:
          - Initialize the model parameters to uniform random values in the
            interval [-0.1, +0.1].
          - While not converged:
            - Compute the gradient of the loss with respect to the current
              batch.
            - Update the model parameters by moving them in the direction
              opposite to the current gradient. Use the learning rate as the
              step size.
        For the convergence criteria, compute the loss over all examples. If
        this loss changes by less than 1e-4 during an update, assume that the
        model has converged. If this convergence criteria has not been met
        after max_iter iterations, also assume convergence and terminate.

        You should include a bias term by APPENDING a column of 1s to your
        feature matrix. The bias term is then the last value in self.model.

        Arguments:
            features - (np.array) An Nxd array of features, where N is the
                number of examples and d is the number of features.
            targets - (np.array) A 1D array of targets of length N.
            batch_size - (int or None) The number of examples used in each
                iteration. If None, use all of the examples in each update.
            max_iter - (int) The maximum number of updates to perform.
        Modifies:
            self.model - (np.array) A 1D array of model parameters of length
                d+1. The +1 refers to the bias term.
        """
        #add bias term to features N*2 matrix
        n = features.shape[0]
        features = np.insert(features, features.shape[1], 1.0, axis=1) 
        
        #initialize the model params to uniform rand [-0.1, 0.1]
        self.model = np.random.uniform(-0.1, 0.1, features.shape[1])
        #compute the gradient of the loss wrt the current batch
        iter = 0
        loss_log = np.zeros(max_iter)
        params_log = np.zeros((max_iter, features.shape[1]))
        accuracies_log = np.zeros(max_iter)
        
        while iter < max_iter: #max_iter:
            if batch_size != None:
                indices = np.random.permutation(n)
                features = features[indices]
                targets = targets[indices]
                for idx in range(0, n, batch_size):
                    X_i = features[idx:idx+batch_size]
                    Y_i = targets[idx:idx+batch_size]
                    gradient = self.loss.backward(X_i, self.model, Y_i)
                    self.model  -= self.learning_rate * gradient

                loss = self.loss.forward(features, self.model, targets)
                if iter > 0 and np.abs(loss_log[iter-1]-loss) < 1e-4: #considered as converged
                    print("CONVERGED! @", iter, " - Loss: ", loss)
                    return params_log, loss_log, accuracies_log
                
                predictions = self.predict(test_features)
                # accuracies_log[iter] = accuracy(test_targets, predictions)
                loss_log[iter] = loss
                # print("!!",accuracies_log, loss_log)


            # # * -------------------------------------------------------------
            if not isMinibatch:
                gradient = self.loss.backward(features, self.model, targets)
                self.model  -= self.learning_rate * gradient
                
                #compute the loss as convergence criteria
                loss = self.loss.forward(features, self.model, targets)
                loss_log[iter] = loss
                params_log[iter] = self.model
            
                #! calcualte accuracy
                if test_features != [] and test_targets != []:
                    predictions = self.predict(test_features)
                    # accuracies_log[iter] = accuracy(test_targets, predictions)
                    # print(accuracy(test_targets, predictions))
                    #converging criteria
                    if iter > 0 and np.abs(loss_log[iter-1]-loss) < 1e-4: #considered as converged
                        print("CONVERGED! @", iter, " - Loss: ", loss)
                        return params_log, loss_log, accuracies_log
                    if isPrinting and iter % 200 == 0:
                        print("model's loss: ",loss)
                        print("current iteration: ", iter)
                        print("Gradient: ", sum(gradient))   
            iter += 1
        

        return params_log, loss_log, accuracies_log

    def predict(self, features):
        """
        Predicts the class labels of each example in features. Model output
        values at and above 0 are predicted to have label +1. Non-positive
        output values are predicted to have label -1.

        NOTE: your predict function should make use of your confidence
        function (see below).

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            predictions - (np.array) A 1D array of predictions of length N,
                where index d corresponds to the prediction of row N of
                features.
        """
        confidence = self.confidence(features)
        confidence[confidence >= 0] = 1
        confidence[confidence < 0] = -1
        # confidence = np.sign(confidence)
        return confidence

    def confidence(self, features):
        """
        Returns the raw model output of the prediction. In other words, rather
        than predicting +1 for values above 0 and -1 for other values, this
        function returns the original, unquantized value.

        Arguments:
            features - (np.array) A Nxd array of features, where N is the
                number of examples and d is the number of features.
        Returns:
            confidence - (np.array) A 1D array of confidence values of length
                N, where index d corresponds to the confidence of row N of
                features.
        """
        features = np.insert(features, features.shape[1], 1.0, axis=1) 
        return np.dot(features, self.model)
    
if __name__ == "__main__":
    print("testing...")
    
    import matplotlib.pyplot as plt
    from load_data import load_data
    from metrics import accuracy
    
    #TODO for visualization
    train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary', fraction=1.0)
    
    # ! 1b
    learner = GradientDescent(loss='hinge', regularization=None,
                            learning_rate=1e-4, reg_param=0.05)
    
    params_log, loss_log, accuracies_log = learner.fit(
        train_features, train_targets, batch_size=64, max_iter=1000, 
        isPrinting=True, isMinibatch=True, test_features=train_features, test_targets=train_targets)


    # ! 1a
    # learner = GradientDescent(loss='hinge', regularization=None,
    #                             learning_rate=1e-4, reg_param=0.05)
    # params_log, loss_log, accuracies_log = learner.fit(
    #     train_features, train_targets, batch_size=None, max_iter=1000, 
    #     isPrinting=True, isMinibatch=False, test_features=train_features, test_targets=train_targets)
    
    # ! 1
    y_values = loss_log
    y_values = y_values[y_values != 0]
    x_values = [i for i in range(y_values.shape[0])]
    plt.plot(x_values, y_values)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy Scores")
    plt.title('Accuracy')
    plt.show()
    
    # for dev testing
    '''
    features, _, targets, _ = load_data('blobs')
    np.random.seed(0)
    #features, targets, 'hinge'/'squared', 'l1'/l2/None
    learner = GradientDescent(loss='squared', regularization=None,
                            learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000, isPrinting=False)
    # learner.confidence(features)
    __est = learner.predict(features)

    
    print(">>>>>> Test 1: ",np.all(__est == targets),accuracy(__est, targets), "<<<<<<<")
    train_features, test_features, train_targets, test_targets = \
    load_data('mnist-binary', fraction=0.8)
    accuracies = []
    for s in [0, 1, 13, 21, 234, 12093, 123901, 101, 101010, 1020]: #[0, 1, 13, 21, 234, 12093, 123901, 101, 101010, 1020]
        np.random.seed(s)
        learner = GradientDescent(loss='squared', regularization=None,
                                learning_rate=0.01, reg_param=0.05)
        learner.fit(train_features, train_targets, batch_size=None, max_iter=1000, isPrinting=False)
        predictions = learner.predict(test_features)
        accuracies.append(accuracy(test_targets, predictions))
    accuracies = np.array(accuracies)
    print("Accuracies: ",accuracies)
    # ! --------------
 
    # print(accuracies)
    # assert accuracies.mean() > 0.95
    '''