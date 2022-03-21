import numpy as np 
from k_nearest_neighbor import KNearestNeighbor
from generate_regression_data import generate_regression_data
from polynomial_regression import PolynomialRegression
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from metrics import mean_squared_error


def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {ord}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

if __name__ == "__main__":
    data_x, data_y = generate_regression_data(4, 100, 0.1)
    print(len(data_x), len(data_y))
    splitA_data_train, splitA_data_test, splitA_labels_train, splitA_labels_test = train_test_split(data_x, data_y, test_size=0.90, random_state=42)
    splitB_data_train, splitB_data_test, splitB_labels_train, splitB_labels_test = train_test_split(data_x, data_y, test_size=0.50, random_state=24)

    print(len(splitA_data_train), len(splitA_data_test), len(splitA_labels_train), len(splitA_labels_test))
    print(len(splitB_data_train), len(splitB_data_test), len(splitB_labels_train), len(splitB_labels_test))

    # splitA_data_train, splitA_data_test, splitA_labels_train, splitA_labels_test = splitB_data_train, splitB_data_test, splitB_labels_train, splitB_labels_test
    import os
    from load_json_data import load_json_data
    datasets = [
            os.path.join('data', x)
            for x in os.listdir('../data')
            if os.path.splitext(x)[-1] == '.json'
    ]
    aggregators = ['mean', 'mode', 'median']
    distances = ['euclidean', 'manhattan']
    
    # aggregators = [aggregators[0]]
    # distances = [distances[0]]
    trainning_errors = []
    testing_errors = []
    klist = [1, 3, 5, 7, 9]
    
    # for k in klist:
    #     for d in distances:
    #         for a in aggregators:
    #             knn = KNearestNeighbor(k, distance_measure=d, aggregator=a)
    #             knn.fit(splitA_data_train, splitA_labels_train)
    #             labels = knn.predict(np.asarray(splitA_data_test))
    #             mse = mean_squared_error(labels, splitA_labels_test)
    #             testing_errors.append(mse)
                
    #             labels = knn.predict(np.asarray(splitA_data_train))
    #             mse = mean_squared_error(labels, splitA_labels_train)
    #             trainning_errors.append(mse)
    # # trainning_errors = np.asarray(np.log10(trainning_errors))
    # # testing_errors = np.asarray(np.log10(testing_errors))
    
    # plt.plot(klist, trainning_errors, label='Training Errors')
    # plt.plot(klist, testing_errors, label='Testing Errors')
    # plt.legend()
    # plt.ylabel('MSE')
    # plt.xlabel('Choice of K')
    # plt.show()            
    datasets = [datasets[0]]
    klist = [1, 3, 5]
    errorList = []
    param = []
    for k in klist:
        for data_path in datasets:
            features, targets = load_json_data("../"+data_path)
            targets = targets[:, None]  # expand dims
            for d in distances:
                for a in aggregators:
                    knn = KNearestNeighbor(k, distance_measure=d, aggregator=a)
                    knn.fit(features, targets)
                    labels = knn.predict(features)
                    mse = mean_squared_error(targets, labels)
                    errorList.append(mse)
                    param.append((d, a, k))
    print(errorList)
    print(param)
    features, targets = load_json_data("../"+datasets[0])
    targets = targets[:, None]  # expand dims
    cdict = {1: 'red', -1: 'blue'}
    x = [i[0] for i in features]
    y = [i[1] for i in features]
    for g in np.unique(targets):
        print(g)
        plt.scatter(x, y, c = cdict[g])
    plt.show()
                    
    # #? split A
    # # ! 1 line chart of errors
    # degrees = [i for i in range(10)]
    # trainning_errors = []
    # testing_errors = []
    # for i in degrees:
    #     p = PolynomialRegression(i)
    #     p.fit(splitA_data_train, splitA_labels_train)
        
    #     y_hat = p.predict(np.asarray(splitA_data_train))
    #     mse = mean_squared_error(splitA_labels_train, y_hat)
    #     trainning_errors.append(mse)
        
    #     y_hat = p.predict(np.asarray(splitA_data_test))
    #     mse = mean_squared_error(splitA_labels_test, y_hat)
    #     testing_errors.append(mse)
    
    # trainning_errors = np.asarray(np.log10(trainning_errors))
    # testing_errors = np.asarray(np.log10(testing_errors))
    
    # plt.plot(degrees, trainning_errors, label='Training Errors')
    # plt.plot(degrees, testing_errors, label='Testing Errors')
    # plt.legend()
    # plt.ylabel('MSE in log10')
    # plt.xlabel('Degree of Polynomials')
    # plt.show() 

    # # # ! 2 scatter 
    # plt.scatter(splitA_data_train, splitA_labels_train)
    # minTrainErr = min(trainning_errors)
    # minTrainErr_idx = trainning_errors.index(minTrainErr)
    # minTestErr = min(testing_errors)
    # minTestErr_idx = testing_errors.index(minTestErr)
    # # print(trainning_errors, testing_errors)
    # # print(minTrainErr_idx, minTestErr_idx)
    
    # #for training error
    # p = PolynomialRegression(minTrainErr_idx)
    # p.fit(splitA_data_train, splitA_labels_train)
    # coeffs = p.model
    # x = np.linspace(-1, 1, 100)
    # plt.plot(x, PolyCoefficients(x, coeffs), label="Degree "+str(minTrainErr_idx)+' lowest training error')
    
    # #for testing error
    # p1 = PolynomialRegression(minTestErr_idx)
    # p1.fit(splitA_data_train, splitA_labels_train)
    # coeffs = p1.model
    # x = np.linspace(-1, 1, 100)
    # plt.plot(x, PolyCoefficients(x, coeffs), label="Degree "+str(minTestErr_idx)+' lowest testing error')
    # plt.legend()
    # plt.ylabel('y value')
    # plt.xlabel('x value')
    # plt.show()
    
    
    
    
    print("finished")