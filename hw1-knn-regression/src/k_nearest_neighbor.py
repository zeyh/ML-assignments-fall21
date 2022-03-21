import numpy as np 
# from .distances import euclidean_distances, manhattan_distances
# from distances import euclidean_distances, manhattan_distances


def mode(a, axis=0):
    """
    Copied from scipy.stats.mode. 
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L361-L451

    Return an array of the modal (most common) value in the passed array.
    If there is more than one such value, only the smallest is returned.
    The bin-count for the modal bins is also returned.
    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    """
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Hint: numpy has functions for computing the mean and median, but you can use the `mode`
              function for finding the mode. 

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator



    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.num = features.shape[0]
        self.features = features
        self.targets = targets


    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        # init distance matrix for all the testing pt
        result = []
        for i in range(len(features)):
            ptDist = []
            counter = 0
            for train in self.features:
                # currDist = euclidean_distances(train.reshape(train.shape[0],1), np.asarray(features[i]).reshape(features[i].shape[0],1))
                # print("pt: ", features[i], train)
                # print("Dist: ",currDist, euclidean_distances(features[i].reshape(features[i].shape[0],1), train.reshape(train.shape[0],1)))
                # ptDist.append([currDist[0][0], self.targets[counter][0]])
                # print("!!!",train.shape[0])
                # if train.shape[0] < 3:
                currDist = np.sum(np.square(train - features[i]))
                if self.distance_measure == "manhattan":
                    currDist = np.abs(train - features[i]).sum()
                tmp = [currDist]
                tmp.extend(list(self.targets[counter]))
                ptDist.append(tmp)
                counter += 1
            # ptDist = np.asarray(ptDist)
                
            # sort the distances for each point based on the first entry of distance measure
            ptDist.sort(key=lambda x: x[0], reverse=False)

            ptDist = np.asarray(ptDist)
            
            
            currNeighbor = ptDist[:self.n_neighbors][:,1] # * not used
             
            currNeighbor = np.asarray(ptDist[:self.n_neighbors])  # * used 
            #exclude the first column - the distance part
            currNeighbor = currNeighbor[:,1:]

            # #vote the label by mode
            # (label, counts) = np.unique(currNeighbor, return_counts=True)  
            # frequencies = np.asarray((label, counts)).T  
            # frequencies = list(frequencies)   
            # frequencies.sort(key=lambda x: x[1], reverse=True)  
            # result.append(frequencies[0][0])
            
            #by mean
            predicted = currNeighbor.mean(axis=0)
            #by median
            if self.aggregator == "median":
                predicted = np.median(currNeighbor, axis=0)
            #by mode
            if self.aggregator == "mode":
                predicted, oldcounts = mode(currNeighbor)
                predicted.flatten()
                predicted = list(predicted)[0]
                
            result.append(list(predicted))
        # print(self.aggregator, result)
        return np.asarray(result)

       

if __name__ == "__main__":
    import os
    from load_json_data import load_json_data

    datasets = [
            os.path.join('data', x)
            for x in os.listdir('../data')
            if os.path.splitext(x)[-1] == '.json'
    ]
    aggregators = ['mean', 'mode', 'median']
    distances = ['euclidean', 'manhattan']
    
    
    # # For debugging purposes only
    _features = np.array([
        [-1, 1, 1, -1, 2],
        [-1, 1, 1, -1, 1],
        [-1, 2, 2, -1, 1],
        [-1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 1]
    ])

    _predict = np.array([
        [-1, 1, 0, -1, 0],
        [-1, 1, 1, -1, 0],
        [-1, 0, 1, 0, 0],
        [-1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 0]
    ])
    _targets = np.array([
        [1, 0, 1],
        [1, 1, 5],
        [3, 1, 1],
        [1, 1, 2],
        [5, 1, 1]
    ])
    aggregators = ['mean', 'mode', 'median']
    answers = [
        np.repeat(np.mean(_targets, axis=0, keepdims=True), _targets.shape[0], axis=0),
        np.ones_like(_targets),
        np.repeat(np.median(_targets, axis=0, keepdims=True), _targets.shape[0], axis=0)
    ]
    _est = []
    for a in aggregators:
        knn = KNearestNeighbor(5, aggregator=a)
        knn.fit(_features, _targets)
        y = knn.predict(_predict)
        _est.append(y)
    print(aggregators)
    print("output:", _est)
    print("Sol: ",answers)
    print(np.allclose(_est, answers))
    
    # from sk//////learn.neighbors import KNeighborsClassifier
    # from sk/////learn.metrics import accuracy_score as accuracy
    # # datasets = [datasets[0]]
    # aggregators = [aggregators[0]]
    # # distances = [distances[1]]
    
    # for data_path in datasets:
    #     # Load data and make sure its shape is correct
    #     features, targets = load_json_data("../"+data_path)
    #     targets = targets[:, None]  # expand dims
    #     for d in distances:
    #         for a in aggregators:
    #             # make model and fit
    #             knn = KNearestNeighbor(1, distance_measure=d, aggregator=a)
    #             knn.fit(features, targets)

    #             # predict and calculate accuracy
    #             labels = knn.predict(features)
    #             acc = accuracy(targets, labels)
    #             print(acc == 1.0, acc)
                
    #             # # error if there's an issue
    #             # msg = 'Failure with dataset: {}. Settings: dist={}, agg={}.'.format(data_path, d, a)
    #             # assert (acc == 1.0), msg
                
                
                
    #             neigh = KNeighborsClassifier(n_neighbors=1)
    #             neigh.fit(features, targets.reshape(targets.shape[0],))
    #             # print(neigh.get_params())
    #             labels = neigh.predict(features)
    #             acc = accuracy(targets, labels)
    #             # print(acc == 1.0, acc)
                
