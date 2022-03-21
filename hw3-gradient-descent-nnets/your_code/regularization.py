import numpy as np

class Regularization:
    """
    Abstract base class for regularization terms in gradient descent.

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, reg_param=0.05):
        self.reg_param = reg_param

    def forward(self, w):
        """
        Implements the forward pass through the regularization term.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        pass

    def backward(self, w):
        """
        Implements the backward pass through the regularization term.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        pass


class L1Regularization(Regularization):
    """
    L1 Regularization for gradient descent.
    """

    def forward(self, w):
        """
        Implements the forward pass through the regularization term. For L1,
        this is the L1-norm of the model parameters weighted by the
        regularization parameter. Note that the bias (the last value in w)
        should NOT be included in regularization.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        return np.sum(np.abs(w[:-1])) * self.reg_param

    def backward(self, w):
        """
        Implements the backward pass through the regularization term. The
        backward pass is the gradient of the forward pass with respect to the
        model parameters.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        
        gradient = self.reg_param * (w[:-1] / np.abs(w[:-1]))
        return np.insert(gradient, gradient.shape[0], 0, axis=0)  #add the bias term's 0 gradient


class L2Regularization(Regularization):
    """
    L2 Regularization for gradient descent.
    """

    def forward(self, w):
        """
        Implements the forward pass through the regularization term. For L2,
        this is half the squared L2-norm of the model parameters weighted by
        the regularization parameter. Note that the bias (the last value in w)
        should NOT be included in regularization.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        return np.sum(np.power(w[:-1],2)) * self.reg_param * 0.5

    def backward(self, w):
        """
        Implements the backward pass through the regularization term. The
        backward pass is the gradient of the forward pass with respect to the
        model parameters.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        w[w.shape[0]-1] = 0 #set the bias term's gradient to 0
        return self.reg_param * w

if __name__ == "__main__":
    print("test")
    
    # test L1 forward
    # X = np.array([[-1, 2, 1], [-3, 4, 1]]) 
    # regularizer = L1Regularization(reg_param=0.5)
    # _true = np.array([1.5, 3.5])
    # _est = np.array([regularizer.forward(x) for x in X])
    # print(np.allclose(_true, _est),_true, _est)
    
    # test l1 backword
    # X = np.array([[-1, 2, 1], [-3, 4, 1]]) 
    # regularizer = L1Regularization(reg_param=0.5)
    # _true = np.array([[-0.5, 0.5, 0], [-0.5, 0.5, 0]])
    # _est = np.array([regularizer.backward(x) for x in X])
    # print(np.allclose(_true, _est), _true, _est)
    

    # test l2 forward
    # X = np.array([[-1, 2, 1],[-3, 4, 1]]) 
    # regularizer = L2Regularization(reg_param=0.5)
    # _true = np.array([1.25, 6.25])
    # _est = np.array([regularizer.forward(x) for x in X])
    # print(np.allclose(_true, _est),_true, _est)

    # test l2 backward
    X = np.array([[-1, 2, 1], [-3, 4, 1]]) #
    regularizer = L2Regularization(reg_param=0.5)
    _true = np.array([[-0.5, 1, 0], [-1.5, 2, 0]]) #
    _est = np.array([regularizer.backward(x) for x in X])
    print(np.allclose(_true, _est), "Real",_true,"Output", _est)