# Laurence Palmer
# palmerla@usc.edu
# 2024.09

import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
from typing import *


# couple of functions for the loss, written specifically for the RFT
def rmse(y_left: np.array, y_right: np.array) -> float:
    """Calculates the RMSE"""
    n_left = len(y_left)
    n_right = len(y_right)
    left_loss = (y_left - y_left.mean()) ** 2
    right_loss = (y_right - y_right.mean()) ** 2
    return np.sqrt((left_loss.sum() + right_loss.sum()) / (n_left + n_right))


def mse(y_left: np.array, y_right: np.array) -> float:
    """Calculates the MSE"""
    n_left = len(y_left)
    n_right = len(y_right)
    left_loss = (y_left - y_left.mean()) ** 2
    right_loss = (y_right - y_right.mean()) ** 2
    return (left_loss.sum() + right_loss.sum()) / (n_left + n_right)


class RFT:
    """
    Implementation of the RFT as specified in

    https://arxiv.org/abs/2203.11924

    :param loss: the loss function to use
    :param loss_name: the name of the loss function
    """

    def __init__(self, loss: Callable = rmse, loss_name: str = "RMSE"):
        self.loss = loss
        self.sorted_features = None
        self.dim_loss = {}
        self.sorted_features_v = None
        self.dim_loss_v = {}
        self.trained = False
        self.loss_name = loss_name

    def get_minimum_loss(
        self, X: np.array, y: np.array, n_bins: int, remove_outliers: bool = False
    ) -> float:
        """
        Gets the minimum loss for a given feature

        :param X: the 1d X values for the feature
        :param y: the 1d y values for the feature
        :param n_bins: the number of bins to use
        :param remove_outliers: whether to remove outliers from the given feature or not
        :param lowest: the lowest partition loss observed throughout the partitions
        """
        if remove_outliers:
            X, y = self.remove_outliers(X, y)

        lowest = np.inf
        minimum = X.min()
        maximum = X.max()
        bin_width = (maximum - minimum) / n_bins
        for b in range(1, n_bins):
            cut = minimum + bin_width * b
            y_left = y[X <= cut]
            y_right = y[X > cut]
            loss = self.loss(y_left, y_right)

            if loss < lowest:
                lowest = loss

        return lowest

    def remove_outliers(
        self, X: np.array, y: np.array, n_std: int = 3
    ) -> Tuple[np.array, np.array]:
        """
        Removes the outliers from the feature dimensions based on the X values

        :param X: the features should be 1D
        :param y: the targets should be 1D
        :param n_std: the number of stds a value has to be away from mean to be considered an outlier
        :return X, y: the features and the targets with the outliers removed and the corresponding index removed from y
        """
        mn, std = X.mean(), X.std()
        arr_inds = np.abs(X - mn) <= n_std * std
        return X[arr_inds], y[arr_inds]

    def plot_curve(self, save_path: str):
        """
        Plots the RFT curve
        """
        fig = plt.figure(figsize=(10, 10))
        x = list(self.dim_loss.keys())
        x = np.arange(len(self.dim_loss.keys()))
        y = list(self.dim_loss.values())
        plt.plot(x, y)
        plt.xlabel("Sorted Feature Index")
        plt.ylabel(f"RFT loss {self.loss_name}")
        plt.title("RFT Loss Curve")
        plt.savefig(save_path)

    def get_curve(self, type: str = "train") -> Tuple[np.array, np.array]:
        """Gives the RFT loss curve X and Y values"""
        if type == "train":
            x = np.arange(len(self.dim_loss.keys()))
            y = list(self.dim_loss.values())
        else:
            x = np.arange(len(self.dim_loss_v.keys()))
            y = list(self.dim_loss_v.values())
        return x, np.array(y)

    def fit(self, X_t: np.array, y_t: np.array, X_v: np.array, y_v: np.array, n_bins: int, remove_outliers: bool = False):
        """
        Fits the RFT based on the loss function

        :param X_t: the input feature array for training
        :param y_t: the corresponding targets for X_t
        :param X_v: the input feature array for testing 
        :param y_v: the corresponding targets for X_v
        :param n_bins: the number of bins to
        :param remove_outliers: to remove outliers or not
        """

        for i in range(X_t.shape[1]):
            min_loss = self.get_minimum_loss(X_t[:, i], y_t, n_bins, remove_outliers)
            self.dim_loss[i] = min_loss

        self.dim_loss = {
            k: v for k, v in sorted(self.dim_loss.items(), key=lambda item: item[1])
        }
        self.sorted_features = np.array(list(self.dim_loss.keys()))
        self.trained = True

        # do the validation set 
        for i in range(X_v.shape[1]):
            min_loss_val = self.get_minimum_loss(X_v[:, i], y_v, n_bins, remove_outliers)
            self.dim_loss_v[i] = min_loss_val
        self.sorted_features_v = np.array(list(self.dim_loss.keys()))

    def transform(self, X: np.array, n_selected: int) -> np.array:
        """
        Transforms the given input array using fitted values

        :param X: the input feature array
        :param n_selected: the number of features to select
        :return features: the features that are associated with the n_selected lowest losses
        """
        assert self.trained, "Need to train RFT"
        if n_selected > len(self.sorted_features) - 1:
            n_selected = len(self.sorted_features) - 1
        features = X[:, self.sorted_features[np.arange(n_selected)]]
        return features
