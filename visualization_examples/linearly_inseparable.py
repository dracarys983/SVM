"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM example using scikit-learn
Code to separate 2D and 3D linearly-separable data with SVMs using
the scikit-learn library. Examples for different kernels with
various parameter combinations are provided. Visualized using
matplotlib.
"""

from sklearn import svm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

# For reproducibility:
# Same seed produces similar randomness
np.random.seed(123)

def generate_training_data_2D():
    """
    Generate 100 random (x, y) coordinates for
    two different classes separable by a line.
    The points are plotted and saved in a PDF
    in the present directory.

    Returns
    -------
    c1, c2: Points belonging to the two classes
    """
    c11 = np.random.uniform(-0.50, 1.50, 100)
    c12 = np.random.uniform(-2.50, 1.50, 100)
    c21 = np.random.uniform(-1.50, 0.50, 100)
    c22 = np.random.uniform(-1.50, 2.50, 100)
    c1 = np.array([[i, j] for i, j in zip(c11, c12)])
    c2 = np.array([[i, j] for i, j in zip(c21, c22)])

    points = plt.figure()
    plt.plot(c1[:, 0], c1[:, 1], 'o', c2[:, 0], c2[:, 1], '*')
    plt.show()
    plt.close()

    return c1, c2

def generate_training_data_3D():
    """
    Generate 20 random (x, y, z) coordinates for
    two different classes separable by a line.
    The points are plotted and saved in a PDF
    in the present directory.

    Returns
    -------
    c1, c2: Points belonging to the two classes
    """
    c11 = np.random.uniform(0.05, 1.50, 20)
    c12 = np.random.uniform(-1.50, 1.50, 20)
    c13 = np.random.uniform(-2.50, -0.05, 20)
    c21 = np.random.uniform(-1.50, -0.05, 20)
    c22 = np.random.uniform(-1.50, 1.50, 20)
    c23 = np.random.uniform(0.05, 2.50, 20)
    c1 = np.array([[i, j, k] for i, j, k in zip(c11, c12, c13)])
    c2 = np.array([[i, j, k] for i, j, k in zip(c21, c22, c23)])

    points = plt.figure()
    ax = points.add_subplot(111, projection='3d')
    ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], c='r', marker='^')
    ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], c='b', marker='*')
    plt.show()
    plt.close()

    return c1, c2


def create_meshgrid(x, y, h=0.015):
    """
    Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """
    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional

    Returns
    -------
    out: The contours object
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def get_fitted_svm(X, Y):

    """
    Change the values of C in the range [1.0, 1500.0] to see how
    different kernels behave depending on C.
    Higher C results in less misclassification rate with a complex
    decision surface; lower C prefers a simple decision boundary
    and hence has chances of higher misclassification rate.
    """
    C = 1000.0             # SVM Regularization Parameter
    """
    Different classifiers using the various kernels available.
    -> Polynomial kernels with different degrees work for different
    types of problems.
    -> RBF and Sigmoid kernels change with the variation of gamma (margin).
    """
    linear_classifier = svm.SVC(kernel='linear', C=C)
    linear_classifier.fit(X, Y)
    poly_classifier_1 = svm.SVC(kernel='poly', degree=1, C=C)
    poly_classifier_1.fit(X, Y)
    poly_classifier_2 = svm.SVC(kernel='poly', degree=2, C=C)
    poly_classifier_2.fit(X, Y)
    poly_classifier_3 = svm.SVC(kernel='poly', degree=3, C=C)
    poly_classifier_3.fit(X, Y)
    rbf_classifier_g0 = svm.SVC(kernel='rbf', gamma=0.5, C=C)
    rbf_classifier_g0.fit(X, Y)
    rbf_classifier_g1 = svm.SVC(kernel='rbf', gamma=0.8, C=C)
    rbf_classifier_g1.fit(X, Y)
    sigmoid_classifier_g0 = svm.SVC(kernel='sigmoid', gamma=0.5, C=C)
    sigmoid_classifier_g0.fit(X, Y)
    sigmoid_classifier_g1 = svm.SVC(kernel='sigmoid', gamma=0.8, C=C)
    sigmoid_classifier_g1.fit(X, Y)

    models = (linear_classifier, poly_classifier_1, poly_classifier_2, poly_classifier_3,
            rbf_classifier_g0, rbf_classifier_g1, sigmoid_classifier_g0, sigmoid_classifier_g1)

    titles = ('SVC with Linear kernel',
              'SVC with Polynomial kernel (degree = 1)',
              'SVC with Polynomial kernel (degree = 2)',
              'SVC with Polynomial kernel (degree = 3)',
              'SVC with RBF kernel (gamma = 0.5)',
              'SVC with RBF kernel (gamma = 0.8)',
              'SVC with Sigmoid kernel (gamma = 0.5)',
              'SVC with Sigmoid kernel (gamma = 0.8)')

    return models, titles


def plot_decision_boundary(X, Y, models, titles):
    """
    Plotting decision boundaries logic for various kernels.
    This logic may be helpful in visualizing your final classifier
    in the assignment question.
    """
    fig, sub = plt.subplots(2, 4, figsize=(20, 8))
    plt.subplots_adjust(wspace=1.0, hspace=0.6)

    xx, yy = create_meshgrid(X[:, 0], X[:, 1])

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xlabel('Xvalues')
	ax.set_ylabel('Yvalues')
	ax.set_xticks(())
	ax.set_yticks(())
	ax.set_title(title)

    back = matplotlib.get_backend()
    manager = plt.get_current_fig_manager()
    if "QT" in back:
        manager.window.showMaximized()
    elif "Tk" in back:
        manager.resize(*manager.window.maxsize())
    else:
        manager.frame.Maximize(True)
    plt.show()
    plt.close()


def SVM():
    """
    Creates different sklearn.SVC objects using
    the various kernels available for SVM.
    The decision boundaries for various kernels are
    plotted as contours in a matplotlib Figure with
    subplots. This is helpful for visualization.

    However, these boundaries are for 2D data and hence
    should not be considered as Holy Grail for high
    dimensional data. The actual assignment question will
    have much more complex data.
    """
    x1, x2 = generate_training_data_2D()
    Y = np.concatenate([np.zeros(x1.shape[0], dtype=np.int32),
            np.ones(x2.shape[0], dtype=np.int32)])
    X = np.concatenate([x1, x2], axis=0)
    rng = np.random.get_state()
    np.random.shuffle(X)
    # Set the random state back to previous to shuffle X & Y similarly
    np.random.set_state(rng)
    np.random.shuffle(Y)

    models, titles = get_fitted_svm(X, Y)

    plot_decision_boundary(X, Y, models, titles)

if __name__ == '__main__':
    SVM()
