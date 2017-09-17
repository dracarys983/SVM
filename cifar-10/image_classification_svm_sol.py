"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM using scikit-learn.
Skeleton code for implementing SVM classifier for an
image classification dataset (cifar-10).

Dataset is taken from: https://www.cs.toronto.edu/~kriz/cifar.html

Remember
--------
1) SVM algorithms are not scale invariant.
2) You might have to import more modules from sklearn.
"""

from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

import argparse, os, sys, glob

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
    pca = PCA(n_components=48)
    X = pca.fit_transform(X)

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
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')

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
    clf = svm.SVC(kernel=kernel, C=1000.0, gamma=0.7, decision_function_shape='ovr')
    clf.fit(train_data, train_labels)
    train_predictions = clf.predict(train_data)

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
    test_predictions = clf.predict(test_data)
    accuracy = calculate_accuracy(test_predictions, test_labels)
    precision, recall, f1 = calculate_metrics(test_predictions, test_labels)

    """
    ==========================================================================
    """

    print "%.4f, %.4f, %.4f, %.4f" % (accuracy, precision, recall, f1)
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
            help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        print "Usage: python image_classification_svm.py --data_dir='<dataset dir path>'"
        sys.exit()
    else:
        filename = glob.glob(os.path.join(args.data_dir, 'data_batch_*'))
        for fn in filename:
            try:
                if os.path.exists(fn):
                    print "Using %s as the dataset file" % filename
            except:
                print "%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir)
                sys.exit()

    # Set the value for svm_kernel as required.
    svm_kernel = 'rbf'

    """
    Get the input data using the provided function. You need to get the data
    for all the five batches (paths to which are in the list contained in
    filename), so run a loop. Store the final X and Y as X_data and Y_data.
    All images in X_data and labels in Y_data.
    ==========================================================================
    """

    # YOUR CODE GOES HERE
    X_data = []; Y_data = []
    for fn in filename[:2]:
        X, Y = get_input_data(fn)
        X_data.append(X); Y_data.append(Y)
    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.concatenate(Y_data, axis=0)
    print X_data.shape, Y_data.shape

    """
    ==========================================================================
    """

    """
    We use 5-fold cross validation for reporting the final scores and metrics.
    Code to generate the 5-folds (You can change the split function to something
    else that you like as well) and then calling the SVM to classify the present
    split.
    ==========================================================================
    """
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        print "Fold%d -> Number of training samples: %d | Number of testing "\
            "samples: %d" % (fold, len(train_indices), len(test_indices))
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        accumulated_metrics.append(
            SVM(train_data, train_labels, test_data, test_labels,
                svm_kernel))
        fold += 1

    """
    Print out the accumulated metrics in a good format.
    ==========================================================================
    """

    # YOUR CODE GOES HERE
    N = len(accumulated_metrics)
    avg_acc = sum([x[0] for x in accumulated_metrics]) / N
    avg_pre = sum([x[1] for x in accumulated_metrics]) / N
    avg_rec = sum([x[2] for x in accumulated_metrics]) / N
    avg_f1 = sum([x[3] for x in accumulated_metrics]) / N
    print avg_acc, avg_pre, avg_rec, avg_f1

    """
    ==========================================================================
    """
