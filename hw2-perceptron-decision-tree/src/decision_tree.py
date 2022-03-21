import numpy as np
from collections import Counter, OrderedDict
import math
from operator import add
from functools import reduce
import random


class Node():
    def __init__(self, value=None, attribute_name="root", feature_name="", attribute_index=None, branches=None, currFreqCount=None, decision=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.feature_name = feature_name
        self.value = value
        # self.currFreqCount = currFreqCount
        self.decision = decision
        
        
    def addLeaf(self, node):
        self.branches.append(node)
    
    def addLeaves(self, nodelist):
        self.branches = nodelist
        
    def setData(self, restFeatures, restTargets, attribute_names):
        self.features = restFeatures
        self.targets = restTargets
        self.attribute_names = attribute_names

    
    def print(self):
        print("PRINTING the node: ", self.feature_name, " from ", self.attribute_name, "---------")
        print("LEAVES: ", len(self.branches))
        if self.branches == []:
            print("EMPTY")
        for node in self.branches:
            print(node.feature_name, " from ", node.attribute_name)
            # print("freq: ", node.currFreqCount)
        print("--------------")
            
    def findAllLeaves(self, classes):
        '''
        require setData() being called before this
        add all leaves to a currRootNode
        reduce one of feature's columns
        reduce one of attribute_names columns
        '''
        if self == None:
            return

        # self.print()
        # print(self.features, self.targets, self.attribute_names)
        selectedColumn = self.features[:, self.attribute_index]
        #update features without this column
        features = np.delete(self.features, self.attribute_index, 1)
        targets = self.targets
        restAttribute_names =  np.delete(self.attribute_names, self.attribute_index, 0).tolist()
        currAttrNames = set(selectedColumn) 
        currFreqCount = count_class_given_target(selectedColumn, self.targets, classes.keys())
        # print(currFreqCount)
        # print(currAttrNames, selectedColumn, self.targets)
        # print()
        
        #? FIND LEAVES: calculate which feature should be the next branch's selected Node
        currRootNodeTotalEnt = {}
        for attrName in currFreqCount:
            currRootNodeTotalEnt[attrName] = entropy_sub_idx(currFreqCount[attrName])
            if currRootNodeTotalEnt[attrName] == 0: #! make a decision
                currFreqCount[attrName] = dict(sorted(currFreqCount[attrName].items(), key=lambda item: len(item[1]), reverse=True))
                decision = list(currFreqCount[attrName].keys())[0]
                # print(attrName, " - END! ",currFreqCount[attrName], "==>",decision)
                currLeafNode = Node(
                        attribute_name=attrName,
                        branches=[],
                        decision=decision
                )
                self.addLeaf(currLeafNode)
            elif len(restAttribute_names) > 0: #! keep going down
                restIdx = currFreqCount[attrName].values() # extracting the rest feature array based on the index
                restIdx = sorted(reduce(add,restIdx))
                
                if len(restIdx) != len(set(restIdx)):
                    print("ERROR!!!! sanity check!!! current extracted index is wrong!!!")
                # print("!!!", features, targets)
                # print("************",attrName, " - ",restIdx)
                restFeatures = features[restIdx, :]  
                restTargets = np.take(targets, restIdx)   

                dict_entropy = {}
                for i in range(len(restAttribute_names)):
                    dict_entropy[restAttribute_names[i]] = information_gain(restFeatures, i, restTargets)
                dict_entropy = dict(sorted(dict_entropy.items(), key=lambda item: item[1], reverse=True))
                currLeafName = next(iter(dict_entropy))
                currLeafNode = Node(
                        attribute_name=attrName,
                        attribute_index=restAttribute_names.index(currLeafName),
                        value=dict_entropy[currLeafName],
                        branches=[],
                        feature_name=currLeafName,
                        currFreqCount=currFreqCount
                )
                # print(dict_entropy)
                # print(restFeatures) #shrink the rows #shrinked the columns before
                # print(restTargets) #shrink the rows
                # print(restAttribute_names) #shrinked the columns before
                currLeafNode.setData(restFeatures, restTargets, restAttribute_names)
                self.addLeaf(currLeafNode)


class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = self
        # self.root = None
        self.height = 0
        self.nodeNum = 0

    def setRoot(self, root):
        self.tree = root
        
    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            None: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        self.classes = Counter(targets)
        
        # ! set up the root
        dict_entropy = {}
        for i in range(len(self.attribute_names)):
            dict_entropy[self.attribute_names[i]] = information_gain(features, i, targets)
        dict_entropy = dict(sorted(dict_entropy.items(), key=lambda item: item[1], reverse=True))
        currRootName = next(iter(dict_entropy))
        currRootNode = Node(
                feature_name=currRootName,
                attribute_index=self.attribute_names.index(currRootName),
                value=dict_entropy[currRootName],
                branches=[],
        )
        currRootNode.setData(features, targets, self.attribute_names)
        self.setRoot(currRootNode)
        self.nodeNum = 1
        
        # ! BFS to build the tree
        queue = []
        queue.append(currRootNode)
        height = 0 
        while queue != [] and height < len(self.attribute_names):
            queue2 = []
            while queue != []:
                # print("QUEUE:",len(queue))
                currNode = queue.pop()
                self.nodeNum += 1
                currNode.findAllLeaves(self.classes)
                # currNode.print()
                for n in currNode.branches:
                    if n.decision == None:
                        queue2.append(n)
            # self.visualize()
            height += 1
            queue = queue2

        self.height = height
                        
    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predictions 
            for the input data.
        """
        self._check_input(features)
        # print(self.attribute_names)
        
        targets_hat = [-1]*features.shape[0]
        # targets_hat = [0.0]*int(features.shape[0]/2) + [1.0]*(features.shape[0] - int(features.shape[0]/2))
        # random.shuffle(targets_hat)
        count = 0
        for i, pt in enumerate(features):
            # print(pt)
            # traverse the tree for each pt
            queue = [self.tree]
            height = 0
            past = []
            while queue != []:
                queue2 = []
                while queue != []:
                    currNode = queue.pop()
                    if height == len(self.attribute_names)-1: #force to make a decision at the last level
                        # currNode.print()
                        if targets_hat[i] == -1 and currNode.decision == None:
                            # targets_hat[i] = 0 if sum(past) <= 0 else 1
                            print("@", sum(past), targets[i], past)
                            print("@", targets[i], pt, self.attribute_names.index(currNode.feature_name))
                            targets_hat[i] = pt[self.attribute_names.index(currNode.feature_name)]
                    if currNode.feature_name != '':
                        # print("At Level: ",currNode.feature_name, currNode.attribute_name)
                        test_feature_idx = self.attribute_names.index(currNode.feature_name)
                        test_attr = pt[test_feature_idx]
                        # print("Testing: ",test_feature_idx, test_attr)
                        if currNode.branches != []:
                            for n in currNode.branches:
                                # print(n.attribute_name)
                                # if n.attribute_name == "root":
                                #     queue2.append(n) 
                                if n.attribute_name == test_attr:
                                    if n.decision != None:
                                        targets_hat[i] = n.decision
                                        count += 1
                                    else:
                                        queue2.append(n)
                                        if n.attribute_name == 1:
                                            past.append(1)
                                        elif n.attribute_name == 0:
                                            past.append(-1)
                                        # print("@@@@",n.attribute_name, past)

                queue = queue2
                height +=1
           
            
        # print(count, features.shape)
        # print(len(targets_hat))
        # freqClass = Counter(targets_hat)
        # freqClass.pop(-1, None)
        # freqClass = dict(sorted(freqClass.items(), key=lambda item: item[1], reverse=True))
        # majority = list(freqClass.keys())[0]
        targets_hat = np.asarray(targets_hat)
        # targets_hat = np.where(targets_hat==-1, majority, targets_hat) 
        # print(targets)
        return targets_hat
                
        
    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A; we'll split S into two data points.

    For binary A: S(A == 0) and S(A == 1)
    For continuous A: S(A < m) and S(A >= m), where m is the median of A in S.

    Together, the two subsets make up S. If the attribute A were perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (groups in S)} -p(c) * log_2 p(c)

    To elaborate: for each group in S, you compute its prior probability p(c):

        (# of elements of group c in S) / (total # of elements in S)

    Then you compute the term for this group:

        -p(c) * log_2 p(c)

    Then compute the sum across all groups: either classes 0 and 1 for binary data, or
    for the above-median and below-median classes for continuous data. The final number
    is the entropy. To gain more intuition about entropy, consider the following - what
    does H(S) = 0 tell you about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    column = features[:,attribute_index].tolist()
    # column = ['s','s','o','r','r','r','o','s','s','r','s','o','o','r' ]
    N = len(column)
    targets = targets.tolist()
    classes = Counter(targets)
    
    freq_features_cond = {}
    for i,f in enumerate(column):
        tmp = {}
        for j, c in enumerate(classes.keys()):
            tmp[c] = 0
        freq_features_cond[f] = tmp
        
    for i,f in enumerate(column):
        freq_features_cond[f][targets[i]] += 1

    ent_total = entropy_sub(classes)
    ent_sum = 0.0
    for i,key in enumerate(freq_features_cond):
        ent_idv = entropy_sub(freq_features_cond[key])
        prob = sum(freq_features_cond[key].values())/N
        ent_sum = ent_sum - prob * ent_idv
        
    return ent_total+ent_sum

def count_class_given_target(column, targets, targetClasses):
    freq_features_cond = {}
    for i,f in enumerate(column):
        tmp = {}
        for j, c in enumerate(targetClasses):
            tmp[c] = []
        freq_features_cond[f] = tmp
        
    for i,f in enumerate(column):
        freq_features_cond[f][targets[i]].append(i)
    return freq_features_cond

def entropy_sub(dict):
    N = sum(dict.values())
    ent = 0.0
    for i,key in enumerate(dict):
        if dict[key] == 0:
            ent = ent
        else:
            ent = ent - dict[key]/N * math.log(dict[key]/N, 2)
    return ent

def entropy_sub_idx(raw):
    raw = list(raw.values()) #2d list of indices grouped by target TF
    total = sum(len(x) for x in raw)
    ent = 0.0
    for target in raw:
        prob = len(target)/total
        if len(target) == 0:
            ent = ent
        else:
            ent = ent - prob * math.log(prob, 2)
    return ent
    
if __name__ == '__main__':
    from data import load_data
    # #test information gain
    # _features, _targets, _attribute_names = load_data('data/PlayTennis.csv')
    # iGHumidity = information_gain(_features, 2, _targets)
    # realIGHumidity = 0.1515
    # iGWind = information_gain(_features, 3, _targets)
    # realIGWind = 0.048
    
    # print("------result-------")
    # print(np.abs(iGHumidity-realIGHumidity)< 1e-3, iGHumidity, realIGHumidity)
    # print(np.abs(iGWind - realIGWind) < 1e-3, iGWind, realIGWind)
    
    # construct a fake tree
    # attribute_names = ['larry', 'curly', 'moe']
    
    # attribute_names = ["outlook", "temp","humidity", "wind"]
    # decision_tree = DecisionTree(attribute_names=attribute_names)
    # features = np.asanyarray([['s','s','o','r','r','r','o','s','s','r','s','o','o','r'],
    #           ['h','h','h','m','c','c','c','m','c','m','m','m','h','m'],
    #           ['h','h','h','h','n','n','n','h','n','n','n','h','n','h'],
    #           ['w','s','w','w','w','s','s','w','w','w','s','s','w','s']]).T
    # targets = np.asarray([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
    # # print(features)
    # decision_tree.fit(features, targets)
    # decision_tree.visualize()
    # print("==========")
    # print(decision_tree.predict(features))
    
    # while len(attribute_names) > 0:
    #     attribute_name = attribute_names[0]
    #     if not decision_tree.tree:
    #         decision_tree.tree = Node(
    #             attribute_name=attribute_name,
    #             attribute_index=decision_tree.attribute_names.index(attribute_name),
    #             value=0,
    #             branches=[]
    #         )
    #     else:
    #         decision_tree.tree.branches.append(
    #             Node(
    #                 attribute_name=attribute_name,
    #                 attribute_index=decision_tree.attribute_names.index(attribute_name),
    #                 value=0,
    #                 branches=[]
    #             )
    #         )
    #     attribute_names.remove(attribute_name)
    # decision_tree.visualize()
    
    # import os
    # from data import train_test_split
    # from metrics import compute_accuracy
    # datasets = [
    #     os.path.join('data', x)
    #     for x in os.listdir('data')
    #     if x.endswith('.csv')
    # ]
    # accuracies = {}
    # for data_path in [datasets[9]]: #5 #9
    #     learner_type = 'decision_tree'
    #     features,targets,attribute_names = load_data(data_path)
    #     classifier = DecisionTree(attribute_names=attribute_names)
    #     train_features, train_targets, test_features, test_targets = train_test_split(features, targets, 1)
        
    #     print(attribute_names)
    #     print(train_features, train_targets)
    #     classifier.fit(train_features, train_targets)
    #     targets_hat = classifier.predict(test_features)
    #     accuracy = compute_accuracy(test_targets, targets_hat)
    #     print(data_path, accuracy)
    
    from data import load_data
    from visualize import plot_decision_regions
    from metrics import compute_accuracy
    path = 'data/parallel_lines.csv'
    features, targets, attribute_names = load_data(path)
    print("data size: ",features.shape, targets.shape)
    classifier = DecisionTree(attribute_names=attribute_names)
    classifier.fit(features, targets)
    targets_hat = classifier.predict(features)
    # print("input",features)
    # print("output",targets_hat)
    accuracy = compute_accuracy(targets, targets_hat)
    print(accuracy)

    # print("test: ",classifier.predict(np.asarray([[44.00347913, 78.02539841],
                                                #   [-29.18737575,  61.14292829]])))
    plot_decision_regions(features, targets, classifier, path)