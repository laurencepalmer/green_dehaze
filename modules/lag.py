# Laurence Palmer
# palmerla@usc.edu
# 2024.08

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy
import pdb
from typing import *


class LAG:
    """
    The LAG unit described in the following:

    https://arxiv.org/pdf/1909.08190

    :param num_clusters, the number of clusters to use in the KMeans algorithm
    :param alpha, parameter controlling decay, higher alpha -> faster probability decay with distance to centroid
    """

    def __init__(self, num_clusters: int, alpha: int):
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.centroids = None
        self.weights = None

    def cluster(
        self, images_features: np.array, targets: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Performs K-means clustering on the given images_features within each subclass.
        Assumes targets and images_features match up by index.

        :param images_features, the image features
        :param targets, the labels for the image features should be integers
        :param num_clusters, the number of clusters to use for each subclass
        :return new_targets, the targets based on the subclass clusters
        :return all_class_labels, the new labels of the clusters within each subclass (class number, subcluster)
        :return all_class_centroids, the centroids of each subclass cluster (class number, subcluster, centroid_dim)
        """
        classes = list(set(targets))
        classes.sort()  # want to be ascending
        num_classes = len(classes)

        # new arrays to store the outputs
        new_targets = np.ones(len(targets))
        all_class_labels = np.ones((num_classes, self.num_clusters))
        all_class_centroids = np.ones(
            (num_classes, self.num_clusters, images_features.shape[1])
        )

        for i in range(num_classes):
            # subset to the relevant class
            subclass_indices = np.array(
                [j for j in range(len(targets)) if targets[j] == classes[i]]
            )
            subset_images = images_features[subclass_indices]

            # cluster
            kmean = KMeans(n_clusters=self.num_clusters, random_state=0)
            kmean.fit(subset_images)
            labels = kmean.labels_
            centroids = kmean.cluster_centers_
            pdb.set_trace()

            # assign and record, labels, centroids
            all_class_labels[i] = np.unique(labels) + (i * num_classes)
            all_class_centroids[i] = centroids
            new_targets[subclass_indices] = labels + (i * num_classes)

        self.centroids = all_class_centroids

        return new_targets, all_class_labels, all_class_centroids

    def calculate_prob(self, image_features: np.array, target: int) -> np.array:
        """
        Calculates the probability vector which is the target for the regression task.

        :param image, single
        :target int, the target class (centroids will be indexed according to this)
        :param centroids, the centroids given by the cluster method (class number, subcluster, centroid_dim)
        :param alpha, parameter controlling decay, higher alpha -> faster probability decay with distance to centroid
        """
        # get the target subclass centroids
        subclass_centroids = self.centroids[target]
        prob_vector = np.zeros(subclass_centroids.shape[0])

        # get the distances of the centroids
        for i in range(subclass_centroids.shape[0]):
            dist = np.linalg.norm(image_features - subclass_centroids[i])
            dist *= -self.alpha
            prob_vector[i] = np.exp(dist)

        # pdb.set_trace()
        prob_vector = prob_vector / prob_vector.sum()
        return prob_vector

    def lsr(self, x, y):
        """
        Applies least squares regression to the input
        """
        weight, residues, r, s = scipy.linalg.lstsq(x, y)
        self.weight = weight
        return weight

    def fit(self, image_features: np.array, targets: np.array):
        """
        Fits the LAG Unit
        """

        new_targets, all_class_labels, all_class_centroids = self.cluster(
            image_features, targets
        )
        # print(all_class_labels)
        # print()
        # print(all_class_centroids)
        probs = []
        for i in range(image_features.shape[0]):
            probs.append(self.calculate_prob(image_features[i], targets[i]))
