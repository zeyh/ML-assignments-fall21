import numpy as np

class Loss:
    """
    An abstract base class for a loss function that computes both the prescribed
    loss function (the forward pass) as well as its gradient (the backward
    pass).

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        regularization - (`Regularization` or None) The type of regularization to
            perform. Either a derived class of `Regularization` or None. If None,
            no regularization is performed.
    """

    def __init__(self, regularization=None):
        self.regularization = regularization

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        pass

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        pass


class SquaredLoss(Loss):
    """
    The squared loss function.
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_s(x, y; w) = (1/2) (y - w^T x)^2

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        reg = 0
        if self.regularization != None:
            reg = self.regularization.forward(w)
        w = np.array(w, dtype=np.float64)
        Ls = 0.5 * np.mean(np.square(y - np.dot(w.T, X.T)))
        return Ls + reg

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        reg = 0.0
        if self.regularization != None:
            reg = self.regularization.backward(w)
            
        #y_hat = w*x + b
        #gradient = - 1/n sum_i (y - y_hat)*x
        # gradient = - 1/X.shape[0] * np.dot(X.T, (y - np.dot(X, w)))
        # # print(np.matmul(np.transpose(w), X[0]))
        # print("!!!",tmp_gradient)
        # print("!",(np.matmul(w, X.T) - y), X)
        gradient =  -1 * 1/X.shape[0] * np.dot((y - np.dot(w, X.T)), X)
        return gradient + reg


class HingeLoss(Loss):
    """
    The hinge loss function.

    https://en.wikipedia.org/wiki/Hinge_loss
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The hinge loss for a single example
        is given as follows:

        L_h(x, y; w) = max(0, 1 - y w^T x)

        The hinge loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The calculated loss normalized by the number of
                examples, N.
        """
        reg = 0
        if self.regularization != None:
            reg = self.regularization.forward(w)
        l = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            l[i] = max(0, 1 - np.dot(y[i], np.dot(w, x))) 
        return l.mean() + reg


    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        reg = 0
        if self.regularization != None:
            reg = self.regularization.backward(w)
        
        gradient = np.zeros(w.shape)
        for i,x in enumerate(X):
            # print(i, x, np.dot(y[i], np.dot(w, x)))
            if np.dot(y[i], np.dot(w, x)) < 1:
                gradient += -1 * np.dot(X[i].T, y[i]) 
        return (1/X.shape[0]) * gradient + reg


class ZeroOneLoss(Loss):
    """
    The 0-1 loss function.

    The loss is 0 iff w^T x == y, else the loss is 1.

    *** YOU DO NOT NEED TO IMPLEMENT THIS ***
    """

    def forward(self, X, w, y):
        """
        Computes the forward pass through the loss function. If
        self.regularization is not None, also adds the forward pass of the
        regularization term to the loss. The squared loss for a single example
        is given as follows:

        L_0-1(x, y; w) = {0 iff w^T x == y, else 1}

        The squared loss over a dataset of N points is the average of this
        expression over all N examples.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            loss - (float) The average loss.
        """
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        if self.regularization:
            loss += self.regularization.forward(w)
        return loss

    def backward(self, X, w, y):
        """
        Computes the gradient of the loss function with respect to the model
        parameters. If self.regularization is not None, also adds the backward
        pass of the regularization term to the loss.

        Arguments:
            X - (np.array) An Nx(d+1) array of features, where N is the number
                of examples and d is the number of features. The +1 refers to
                the bias term.
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
            y - (np.array) A 1D array of targets of length N.
        Returns:
            gradient - (np.array) The (d+1)-dimensional gradient of the loss
                function with respect to the model parameters. The +1 refers to
                the bias term.
        """
        # This function purposefully left blank
        raise ValueError('No need to use this function for the homework :p')

if __name__ == "__main__":
    print("test...")
    #test square loss forward
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])
    loss = SquaredLoss(regularization=None)
    _true = 26.5
    _est = loss.forward(X, w, y)
    print(">> test forward: ",np.allclose(_true, _est), "| Real", _true, "Output", _est)

    #test hinge loss forward
    # X = np.array([[-1, 2, 1], [-3, 4, 1]])
    # w = np.array([1, 2, 3])
    # y = np.array([1, -1])
    # loss = HingeLoss(regularization=None)
    # _true = 4.5
    # _est = loss.forward(X, w, y)

    #test square loss backward
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])
    loss = SquaredLoss(regularization=None)
    _true = np.array([-16, 23, 7])
    _est = loss.backward(X, w, y)
    
    
    
    #test hinge loss backward
    # X = np.array([[-1, 2, 1], [-3, 4, 1]])
    # w = np.array([1, 2, 3])
    # y = np.array([1, -1])
    # loss = HingeLoss(regularization=None)
    # _true = np.array([-1.5, 2, 0.5])
    # _est = loss.backward(X, w, y)
    
    print(">> test backward: ",np.allclose(_true, _est), "| Real", _true, "Output", _est)
    
    
    
    



