# from .decision_tree import DecisionTree
# from .prior_probability import PriorProbability
# from .perceptron import Perceptron
# from .metrics import compute_precision_and_recall, compute_confusion_matrix
# from .metrics import compute_f1_measure, compute_accuracy
# from .data import load_data, train_test_split

from decision_tree import DecisionTree
from prior_probability import PriorProbability
from perceptron import Perceptron
from metrics import compute_precision_and_recall, compute_confusion_matrix
from metrics import compute_f1_measure, compute_accuracy
from data import load_data, train_test_split


def run(data_path, learner_type, fraction):
    """
    This function walks through an entire machine learning workflow as follows:

        1. takes in a path to a dataset
        2. loads it into a numpy array with `load_data`
        3. instantiates the class used for learning from the data using learner_type (e.g
           learner_type is 'decision_tree', 'prior_probability', or 'perceptron')
        4. splits the data into training and testing with `train_test_split` and `fraction`.
        5. trains a learner using the training split with `fit`
        6. tests the trained learner using the testing split with `predict`
        7. evaluates the trained learner with precision_and_recall, confusion_matrix, and
           f1_measure

    Each run of this function constitutes a trial. Your learner should be pretty
    robust across multiple runs, as long as `fraction` is sufficiently high. See how
    unstable your learner gets when less and less data is used for training by
    playing around with `fraction`.

    IMPORTANT:
    If fraction == 1.0, then your training and testing sets should be exactly the
    same. This is so that the test cases are deterministic. The test case checks if you
    are fitting the training data correctly, rather than checking for generalization to
    a testing set.

    Args:
        data_path (str): path to csv file containing the data
        learner_type (str): either 'decision_tree,' 'prior_probability', or 'perceptron'.
            For each of these, the associated learner is instantiated and used
            for the experiment.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns:
        confusion_matrix (np.array): Confusion matrix of learner on testing examples
        accuracy (np.float): Accuracy on testing examples using learner
        precision (np.float): Precision on testing examples using learner
        recall (np.float): Recall on testing examples using learner
        f1_measure (np.float): F1 Measure on testing examples using learner
    """
    #1 & 2. takes in a path to a dataset & loads it into a numpy array with `load_data`
    features,targets,attribute_names = load_data(data_path)
    
    #3. instantiates the class used for learning
    global classifier
    classifier = Perceptron(max_iterations=200)
    if learner_type == "decision_tree":
        classifier = DecisionTree(attribute_names=attribute_names)
    elif learner_type == "prior_probability":
        classifier = PriorProbability()
    
    #4. splits the data into training and testing
    train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)
    
    #5. trains a learner using the training split with `fit`
    classifier.fit(train_features, train_targets)
    
    #6. tests the trained learner using the testing split with `predict`
    targets_hat = classifier.predict(test_features)
    
    #7. evaluates the trained learner
    confusion_matrix = compute_confusion_matrix(test_targets, targets_hat)
    accuracy = compute_accuracy(test_targets, targets_hat)
    precision, recall = compute_precision_and_recall(test_targets, targets_hat)
    f1_measure = compute_f1_measure(test_targets, targets_hat)
    
    # Order of these returns must be maintained
    return confusion_matrix, accuracy, precision, recall, f1_measure

if __name__ == "__main__":
    import os
    from tqdm import tqdm
    datasets = [
        os.path.join('data', x)
        for x in os.listdir('data')
        if x.endswith('.csv')
    ]
    accuracies = {}
    for data_path in datasets:
        learner_type = 'perceptron'
        each_acc = []
        for i in tqdm(range(5)):
            confusion_matrix, accuracy, precision, recall, f1_measure = (
                run(data_path, learner_type, 0.8)
            )
            each_acc.append(classifier.converge)
        # accuracies[data_path] = sum(each_acc)/5
        accuracies[data_path] = each_acc

    print(accuracies)
        
    # accuracies = {}
    # for data_path in datasets:
    #     learner_type = 'decision_tree'
    #     each_acc = []
    #     for i in range(5):
    #         confusion_matrix, accuracy, precision, recall, f1_measure = (
    #             run(data_path, learner_type, 0.8)
    #         )
    #         each_acc.append(classifier.nodeNum)
    #     accuracies[data_path] = sum(each_acc)/5
    # print(accuracies)
        
        
    #? test prior probability
    # accuracies = {}
    # for data_path in datasets:
    #     learner_type = 'prior_probability'
    #     confusion_matrix, accuracy, precision, recall, f1_measure = (
    #         run(data_path, learner_type, 1.0)
    #     )
    #     accuracies[data_path] = accuracy
    #     print("RESULT: ", data_path, accuracy)
    # # dataset = xp_dataset_name('ivy-league.csv')
    # # assert (accuracies[dataset] > .2)

    #? test perceptron
    # accuracies = {}
    # for data_path in datasets:
    #     learner_type = 'perceptron'
    #     confusion_matrix, accuracy, precision, recall, f1_measure = (
    #         run(data_path, learner_type, 1.0)
    #     )
    #     accuracies[data_path] = accuracy
    #     print("RESULT: ", data_path, accuracy)

    # accuracy_goals = {
    #     xp_dataset_name('ivy-league.csv'): .85,
    #     xp_dataset_name('candy-data.csv'): .6,
    #     xp_dataset_name('majority-rule.csv'): 1.0,
    #     xp_dataset_name('blobs.csv'): 0.9,
    #     xp_dataset_name('circles.csv'): 0.54,
    #     xp_dataset_name('parallel_lines.csv'): 1.0,
    # }