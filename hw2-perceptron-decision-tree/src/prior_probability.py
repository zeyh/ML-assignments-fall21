import numpy as np
import os, math
from collections import Counter, defaultdict, OrderedDict

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        Output:
            VOID: You should be updating self.most_common_class with the most common class
            found from the prior probability.
        """
        targets = list(map(str, targets.tolist()))
        self.classes = Counter(targets)
        self.featuresNum = features.shape[1]
        self.N = features.shape[0]
        
        print(self.classes, self.featuresNum, self.N)
        
        # a dict hashed by idx each feature's idx
        self.freq_features = {}

        for i in range(self.featuresNum):
            col = features[:, [i]]
            col_list = col.T.tolist()[0]
            self.freq_features[i] = Counter(col_list)

        #* extrating columns for each class
        sortByClass = {}
        for c in self.classes.keys():
            sortByClass[c] = []
            
        for rowIdx, row in enumerate(features):
            sortByClass[targets[rowIdx]].append(row.tolist())

        # a dict hashed by each class's name
        self.freq_features_cond = {}
        for i,c in enumerate(sortByClass.keys()):
            tmp = np.asarray(sortByClass[c])
            print(tmp.shape)
            tmp_feature_freq = {}
            
            for fidx in range(self.featuresNum):
                col = tmp[:, [i]]
                col_list = col.T.tolist()[0]
                tmp_feature_freq[fidx] = Counter(col_list)
            self.freq_features_cond[c] = tmp_feature_freq
        
        # print(sortByClass)
        # print(freq_features_cond)
        
        #update the dict by calculating the conditional prob
        for c in self.freq_features_cond.keys(): #for each class
            for f in self.freq_features_cond[c].keys():  #for each feature
                for val in self.freq_features_cond[c][f]:
                    self.freq_features_cond[c][f][val] = \
                        math.log(self.freq_features_cond[c][f][val]/ self.classes[c])
        # print(self.freq_features_cond)
        # print(self.freq_features)
  
            
            
    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        prediction = []
        for dataPt in data:
            scores = {}
            for eachClass in self.classes.keys():
                scores[eachClass] = self.score(dataPt, eachClass)
            scores = dict(OrderedDict(scores))
            predicted = next(iter(scores))
            prediction.append(float(predicted))
        return np.asarray(prediction)

    
    def score(self, dataPt, eachClass):
        prob = math.log(self.classes[eachClass]/self.N)
        for i, feature in enumerate(dataPt):
            prob += self.freq_features_cond[eachClass][i][feature]
        return prob

if __name__ == "__main__":
    print("local testing...")
    import os
    from experiment import run
    datasets = [
        os.path.join('data', x)
        for x in os.listdir('data')
        if x.endswith('.csv')
    ]
    #? test prior probability
    from data import load_data, train_test_split
    accuracies = {}
    for data_path in datasets:
        if 'candy' in data_path or 'ivy' in data_path:
            # features,targets,attribute_names = load_data(data_path)
            # classifier = PriorProbability()
            # train_features, train_targets, test_features, test_targets = train_test_split(features, targets, 1)
            # classifier.fit(train_features, train_targets)
            # targets_hat = classifier.predict(test_features)
            
            # confusion_matrix = compute_confusion_matrix(test_targets, targets_hat)
            # accuracy = compute_accuracy(test_targets, targets_hat)
            # precision, recall = compute_precision_and_recall(test_targets, targets_hat)
            # f1_measure = compute_f1_measure(test_targets, targets_hat)

            # =======
            learner_type = 'prior_probability'
            confusion_matrix, accuracy, precision, recall, f1_measure = (
                run(data_path, learner_type, 1.0)
            )
            accuracies[data_path] = accuracy
            print("RESULT: ", data_path, accuracy)