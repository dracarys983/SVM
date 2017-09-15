"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM using scikit-learn.
Skeleton code for implementing SVM classifier for an
image classification dataset (cifar-10). This is the final
submission file.

Dataset is taken from: https://www.cs.toronto.edu/~kriz/cifar.html

Remember
--------
1) SVM algorithms are not scale invariant.
2) You might have to import more modules from sklearn.
"""

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

def unpickle(filename):
    """
    Function to read the binary pickled file for cifar-10

    Parameters
    ----------
    filename: The input pickled file

    Returns
    -------
    d: A dictionary of unpickled file content

    """
    import cPickle
    with open(filename, 'rb') as fo:
        d = cPickle.load(fo)
    return d

def get_input_data(filename):
    """
    Function to read the input data from the letter recognition data file.

    Parameters
    ----------
    filename: The path to input data file

    Returns
    -------
    X: The input for the SVM classifier of the shape [n_samples, n_features].
       n_samples is the number of data points (or samples) that are to be loaded.
       n_features is the length of feature vector for each data point (or sample).
    Y: The labels for each of the input data point (or sample). Shape is [n_samples,].

    """

    d = unpickle(filename)
    flattened_images = d['data']        # 10000 x 3072
    labels = d['labels']                # 10000

    X = []; Y = []
    for image, label in zip(flattened_images, labels):
        X.append([float(x) for x in image])
        Y.append(int(label))
    X = np.asarray(X); Y = np.asarray(Y)

    """
    An important part is missing here. Corresponding to point (1) in "Remember".
    Also, do you know how to bend space? Good, bend the feature space then. Do
    some engineering, if you need it ya know.
    ===========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ===========================================================================
    """

    return X, Y

def calculate_metrics(predictions, labels):
    """
    Function to calculate the precision, recall and F-1 score.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    precision: true_positives / (true_positives + false_positives)
    recall: true_positives / (true_positives + false_negatives)
    f1: 2 * (precision * recall) / (precision + recall)
    ===========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ===========================================================================
    """

    return precision, recall, f1

def calculate_accuracy(predictions, labels):
    """
    Function to calculate the accuracy for a given set of predictions and
    corresponding labels.

    Parameters
    ----------
    predictions: The predictions obtained as output from the SVM classifier
    labels: The true label values corresponding to the entries in predictions

    Returns
    -------
    accuracy: Fraction of total samples that have correct predictions (same as
    true label)

    """
    return accuracy_score(labels, predictions)

def SVM(train_data,
        train_labels,
        test_data,
        test_labels,
        kernel='linear'):
    """
    Function to create, train and test the one-vs-all SVM using scikit-learn.

    Parameters
    ----------
    train_data: Numpy ndarray of shape [n_train_samples, n_features]
    train_labels: Numpy ndarray of shape [n_train_samples,]
    test_data: Numpy ndarray of shape [n_test_samples, n_features]
    test_labels: Numpy ndarray of shape [n_test_samples,]
    kernel: linear (default)
            Which kernel to use for the SVM

    Returns
    -------
    accuracy: Accuracy of the model on the test data
    top_predictions: Top predictions for each test sample
    precision: The precision score for the test data
    recall: The recall score for the test data
    f1: The F1-score for the test data

    """

    """
    Create an SVM instance with the required parameters and train it.
    For details on how to do this in scikit-learn, refer:
        http://scikit-learn.org/stable/modules/svm.html
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """

    """
    Calculates training accuracy. Replace predictions and labels with your
    respective variable names.
    """
    train_accuracy = calculate_accuracy(train_predictions, train_labels)
    print "Training Accuracy: %.4f" % (train_accuracy)

    """
    Use the trained model to perform testing. Using the output of the testing
    prodecure, get the top prediction for each sample and calculate the accuracy
    on test data using the function given (as shown above for train accuracy).

    Also, complete the function given above for metrics using scikit-learn and
    return their values in this function.
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    # Set the value for svm_kernel as required.
    svm_kernel = 'linear'

    """
    Get the input data using the provided function. You need to get the data
    for all the five batches, so run a loop. Store the final X and Y as X_data
    and Y_data. All images in X_data and labels in Y_data.
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """

    Xtest_data, Ytest_data = get_input_data('test_batch')

    """
    We perform the testing on a single split as the test batch images were
    provided on the toronto.edu website for CIFAR-10.
    ==========================================================================
    """
    accumulated_metrics = []
    print "Number of training samples: %d | Number of testing "\
        "samples: %d" % (len(X_data), len(Xtest_data))
    train_data, test_data = X_data, Xtest_data
    train_labels, test_labels = Y_data, Ytest_data
    accumulated_metrics.append(
        SVM(train_data, train_labels, test_data, test_labels,
            svm_kernel))

    """
    Print out the accumulated metrics in a good format.
    ==========================================================================
    """

    # YOUR CODE GOES HERE

    """
    ==========================================================================
    """
