"""
Author: Tamir David Hay
Assignment: Intro to ML- exercise 1. KNN algorithm (Brute force edition).
"""
from sklearn.datasets import fetch_openml
import numpy.random
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


# %% functions
def knn_classifier(image_set: list, query_image: list, lables_vec: list, k_number: int) -> int:
    """
    Predicts the label of a given image (0-9 digit)
    :param image_set: Training images set (known labels)
    :param query_image: Image to be classified
    :param lables_vec: Training images corresponding labels
    :param k_number: How many neighbors to include in the check
    :return: a digit for 0-9
    """
    # neighbors = [(label, distance),..]
    neighbors = list()

    # calculate distance for image to any training image
    for index, image in enumerate(image_set):
        distance = euclidean(query_image, image)
        neighbors.append((index, distance))

    # sort based on distance
    sorted_neighbors = sorted(neighbors, key=lambda x: x[1])
    labels = [0] * 10
    # evaluation based only on the first K neighbors
    for number in range(0, k_number):
        neighbor_label = int(lables_vec[sorted_neighbors[number][0]])
        labels[neighbor_label] += 1
    return int(numpy.argmax(labels))


def run(test_images, test_labels, train_images, train_labels, k):
    """
    General function to run KNN on a set of images
    :param test_images: Images to be predicted
    :param test_labels: Known labels, used to check accuracy
    :param train_images: Set of known images
    :param train_labels: Set of known labes
    :param k: Number of neighbors
    :return: Accuracy rate (float)
    """
    correct = 0
    for index, query_image in enumerate(test_images):
        real_res = int(test_labels[index])
        result = knn_classifier(train_images, query_image, train_labels, k)
        if result == real_res:
            correct += 1

    # There were len(test_labels) images to be predicted. Correct was increased only when real == predicted.
    # Therefore Accuray = correct/len(test_labels)
    print("Accuracy: {0}".format(correct / len(test_labels)))
    return correct / len(test_labels)


if __name__ == '__main__':
    # %% import data
    mnist = fetch_openml('mnist_784')
    data = mnist["data"]
    labels = mnist["target"]

    # %% setup
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    # %% run KNN on test set using k=10 and the first 1000 training images
    run(test, test_labels, train[0:1000], train_labels[0:1000], 10)
    # %% run KNN on test set using k from [1,100] and the first 1000 training images. Plot result.
    lim = 101
    k_list = [i for i in range(1, lim)]
    acc_list = list()

    for k in range(1, lim):
        acc = run(test, test_labels, train[0:1000], train_labels[0:1000], k)
        acc_list.append(acc)
    plt.plot(k_list, acc_list)
    plt.ylabel("Accuracy")
    plt.xlabel("K")
    plt.show()

    # %% run KNN on test set using k=1 and n in [100,5000] (+100 per jump) first training images. Plot result.
    k = 1
    n_list = [i for i in range(100, 5001, 100)]
    acc_list = list()
    for n in range(100, 5001, 100):
        acc = run(test, test_labels, train[0:n], train_labels[0:n], k)
        acc_list.append(acc)
    # %% plot
    plt.plot(n_list, acc_list)
    plt.ylabel("Accuracy")
    plt.xlabel("n")
    plt.show()
