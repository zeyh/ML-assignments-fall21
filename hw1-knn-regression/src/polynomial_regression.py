import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.
        
        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c
        
        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf
    
        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np
            
            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)


        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
    
    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.
        

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        # (X^T X)^-1 X^T y
        n = features.size
        features = np.asarray(features).reshape(n,1)
        # X = np.c_[np.ones((n,1)),features]

        #get x matrix 
        X = np.ones((n, self.degree+1))
        for i in range(1, self.degree+1):
            X[:, i] = np.power(features.reshape(n,), i)
        # print(X) 
        
        # z = X.T.dot(X) 
        self.model = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, targets))
        # print(self.model)


    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        n = features.size
        features = np.asarray(features).reshape(n,1)
        # X = np.c_[np.ones((n,1)),features]
        
        #get x matrix 
        X = np.ones((n, self.degree+1))
        for i in range(1, self.degree+1):
            X[:, i] = np.power(features.reshape(n,), i)
        
        self.predictVal = np.dot(X, self.model).reshape(n,)
        return self.predictVal


    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        plt.scatter(features, targets)
        plt.scatter(x, self.predictVal,color="black")
        plt.savefig()
        # plt.show() 

if __name__ == "__main__":
    from generate_regression_data import generate_regression_data
    from metrics import mean_squared_error
    degrees = range(10)
    amounts = [10, 100, 1000, 10000]

    # For debugging purposes only
    # degrees = [degrees[1]]
    # amounts = [amounts[0]]
    # from sk///learn.pipeline import make_pipeline
    # from sk///learn.linear_model import LinearRegression
    # from sk///learn.preprocessing import PolynomialFeatures
    for degree in degrees:
        p = PolynomialRegression(degree)
        for amount in amounts:
            x, y = generate_regression_data(degree, amount, amount_of_noise=0.0)
            p.fit(x, y)
            y_hat = p.predict(x)
            mse = mean_squared_error(y, y_hat)
            print(mse < 1e-1, degree, amount, mse)
            # print(y, y_hat)
            # p.visualize(x,y)
            
            # polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
            # polyreg.fit(x.reshape(-1, 1),y)
            # plt.figure()
            # plt.scatter(x,y)
            # testRes = x,polyreg.predict(x.reshape(-1, 1))
            # print(x, testRes)
            # plt.plot(x, testRes[1],color="black")
            # plt.title("Polynomial regression with degree "+str(degree))
            # plt.show()
            # print("Test!: ",mean_squared_error(y, testRes[1]),degree, amount)
            
    