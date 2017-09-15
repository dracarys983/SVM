"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: tSNE embedding visualization for the dataset.

Dataset is taken from: https://archive.ics.uci.edu/ml/datasets/letter+recognition
"""

from sklearn import preprocessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE
import glob, time
from PIL import Image

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

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    X = preprocessing.scale(X)
    return X, Y

def plot_embedding(X, Y, X_old, images, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    colors = [ord(x)*1.0/12.0 for x in Y[:]]
    plt.scatter(X[:, 0], X[:, 1], c=colors, marker='o')

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X_old.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.bone),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

if __name__ == '__main__':
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    paths = sorted(glob.glob('stencils/*'))
    img_path_dict = dict(zip(letters, paths))

    images = []
    for letter in letters:
        img = Image.open(img_path_dict[letter])
        img = img.crop((80, 50, 400, 365))
        img.thumbnail((16, 16), Image.ANTIALIAS)
        images.append(img)

    images_dict = dict(zip(letters, images))

    X, Y = get_input_data('../letter-recognition-train.data')
    images = []
    for letter in Y:
        images.append(images_dict[letter])

    tsne = TSNE(n_components=2, init='pca',
            verbose=1, learning_rate=600, n_iter=5000)
    t0 = time.time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, Y, X, images,
                   "t-SNE embedding of the letters. Time: %.3f seconds" %
                   (time.time() - t0))
    plt.show()
