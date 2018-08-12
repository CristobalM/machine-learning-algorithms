import numpy as np


def init_clusters(k):
    clusters = []
    for i in range(0, k):
        clusters.append([])

    return clusters


def k_means(data, k, dist_fun):
    """
    :param data: List of points, where each point can be anything that dist_fun can handle
    :type data: list
    :param k: Number of centroids
    :type k: int
    :param dist_fun: Fun distance, will be called as dist_fun(pointA, pointB)
    :type dist_fun: FunctionType
    :return: clusters, result, centroids, Where clusters is a list of point indexes,
            result is a numpy array that maps index to centroid index and
            centroids gives centroids indexes
    :rtype: tuple
    """
    centroids = []
    random_permutation = np.random.permutation(len(data))
    shift = 0
    for i in range(0, k):
        r_idx = random_permutation[i]
        while len(data[r_idx]) <= 5:
            shift += 1
            r_idx = random_permutation[(i + shift) % len(data)]
        centroids.append(r_idx)

    result = np.zeros(len(data), dtype=int)

    previous_centroids = centroids[:]

    count = 0

    to_stop = 0

    while True:
        #print("in iteration number %d of k means" % count)
        count += 1
        clusters = init_clusters(k)
        for i in range(0, len(data)):
            current_element = data[i]
            min_dist = np.inf
            selected_centroid = -1
            for j in range(0, k):
                current_centroid = data[centroids[j]]
                dist = dist_fun(current_element, current_centroid)
                if dist < min_dist:
                    selected_centroid = j
                    min_dist = dist

            result[i] = selected_centroid
            clusters[selected_centroid].append(i)

        #print("initialized centroids")
        if to_stop >= 2:
            break

        for cluster_idx in range(0, k):
            cluster = clusters[cluster_idx]
            cluster_length = len(cluster)
            dist_matrix = np.zeros((cluster_length, cluster_length))
            for i in range(0, cluster_length):
                for l in range(i + 1, cluster_length):
                    dist_matrix[i, l] = dist_fun(data[cluster[i]], data[cluster[l]])
                    dist_matrix[l, i] = dist_matrix[i, l]

            min_sum = np.inf
            new_centroid = centroids[cluster_idx]
            for i in range(0, cluster_length):
                if len(data[cluster[i]]) == 0:
                    row_sum = np.inf
                else:
                    row_sum = np.sqrt(np.sum(dist_matrix[i, :] ** 2)) / len(data[cluster[i]])
                if row_sum < min_sum:
                    new_centroid = cluster[i]
                    min_sum = row_sum

            centroids[cluster_idx] = new_centroid

        #print("updated centroids")

        if centroids == previous_centroids:
            to_stop += 1
        else:
            previous_centroids = centroids[:]
            to_stop = 0

    return clusters, result, centroids
