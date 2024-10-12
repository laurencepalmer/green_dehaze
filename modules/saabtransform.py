# Laurence Palmer
# palmerla@usc.edu
# 2024.08

import numpy as np
import pdb
from typing import *


class SaabTransform:
    """
    Implementation of the Saab Transform for PixelHop described in the following:

    https://arxiv.org/pdf/1909.08190
    https://arxiv.org/pdf/1810.02786

    Specifically for use in the Saab transform

    :param num_kernels: number of kernels
    :param kernel_size: size of the window to apply the saab
    :param thresh: threshold for explained variance/energy
    :param use_bias: to use the bias term or not
    """

    def __init__(
        self,
        num_kernels: int = None,
        thresh: float = None,
        use_bias: bool = False,
        prev_bias: float = 0,
    ):
        self.num_kernels = num_kernels
        self.thresh = thresh
        self.use_bias = use_bias
        self.previous_bias = prev_bias
        self.bias = None
        self.kernels = None
        self.energy = None

    def subtract_mean(self, arr: np.array, axis: int) -> Tuple[np.array, np.array]:
        """Subtracts the mean of the array based on the axis"""
        mn = arr.mean(axis=axis, keepdims=True)
        return arr - mn, mn

    def pca(self, X: np.array) -> np.array:
        """Runs the PCA on the given X"""
        covariance = np.matmul(X.T, X)
        eigenvals, eigenvectors = np.linalg.eigh(covariance)
        indices = np.argsort(eigenvals)[::-1]
        eigenvalues = eigenvals[indices]
        eigenvectors = eigenvectors.T[indices]
        return eigenvectors, eigenvalues / (X.shape[0] - 1)

    def run_pca(self, image: np.array = None) -> Tuple[np.array, np.array, np.array]:
        """
        Run the PCA on the given array, removes features up to thresh for explained variance

        :param: image: the image
        :param: thresh: explained variance threshold to determine number of components to keep
        :return: (components, explained_var, explained_var_ratio): tuple of relevant info
        """
        num_kernels = self.num_kernels
        thresh = self.thresh
        components, energy = self.pca(image)

        # remove up to threshold or kernels, otherwise use all
        if thresh:
            greater_than = np.cumsum(energy) < thresh
            components = components[energy]
            energy = energy[greater_than]
        elif num_kernels:
            components = components[:num_kernels]
            energy = energy[:num_kernels]

        return components, energy

    def discard(self, threshold: float) -> int:
        """
        Retroactively remove kernels with insufficient energy based on threshold

        :param threshold: threshold to throw them away
        :return n_kernels: the current number of kept kernels
        """
        ind = np.argwhere(self.energy < threshold)
        self.kernels = np.delete(self.kernels, ind, axis=0)
        self.energy = np.delete(self.energy, ind)
        return len(self.energy) - len(ind)

    def fit(self, X: np.array):
        """
        Fits the saab transform, assigns attributes to fitted values

        :param X: array to apply saab to
        :param use_bias: whether to add the previous bias or not
        """
        X = X.astype("float32")
        X = X.reshape(-1, X.shape[-1])
        d1, d2 = X.shape

        if self.use_bias == True:
            X += self.previous_bias

        # get/remove dc, get AC
        X, dc = self.subtract_mean(X, axis=1)

        bias = np.max(np.linalg.norm(X, axis=1))

        X, feature_mean = self.subtract_mean(X, axis=0)

        # get the ac_kernels
        kernels, energy = self.run_pca(X)

        dc_kernel = np.ones((1, d2)) / np.sqrt(d2)
        kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)

        dc_en = np.var(dc * np.sqrt(d2))
        energy = np.concatenate((np.array([dc_en]), energy[:-1]), axis=0)
        energy = energy / np.sum(energy)
        self.kernels = kernels.astype("float32")
        self.energy = energy
        self.bias = bias
        self.mean = feature_mean

    def transform(self, X) -> np.array:
        """
        Performs the saab transform using the weights calculated in fit
        and adds bias term

        :param X: the original inputs
        :return transformed: transformed X with the bias term
        """
        X = X.astype("float32")
        if self.use_bias == True:
            X += self.previous_bias

        X = X - self.mean
        transformed = np.matmul(X, self.kernels.T)

        return transformed
