import matplotlib.pyplot as plt
from kmeans import k_means, euclidean_dist
import numpy as np


def k_means_test():
    X = np.concatenate((np.random.normal(5, 30, 100), np.random.normal(150, 44, 100), np.random.normal(300, 20, 100)))
    Y = np.concatenate((np.random.normal(5, 30, 100), np.random.normal(150, 44, 100), np.random.normal(300, 20, 100)))
    data = np.column_stack((X, Y))

    clusters, result, centroids = k_means(data, 3, euclidean_dist)

    cluster_points = []
    for cluster in clusters:
        Xaux = [X[i] for i in cluster]
        Yaux = [Y[i] for i in cluster]
        cluster_points.append((Xaux, Yaux))


    plt.figure()
    colors = ['green', 'red', 'blue', 'yellow', 'gray', 'lightblue', 'pink', 'black']
    i = 0
    for cluster in cluster_points:
        X, Y = cluster
        plt.scatter(X, Y, c=colors[i])
        i = (i + 1) % len(colors)

    plt.show()


k_means_test()