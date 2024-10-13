# Laurence Palmer
# palmerla@usc.edu
# 2024.09

import numpy as np
import pdb
import pickle
import cv2
import matplotlib.pyplot as plt
import xgboost as xgb
from .rft import RFT
from .saabtransform import SaabTransform
from .pixelhop import PixelHop
from typing import *
from .single_xgboost import SingleXGBoost
from sklearn.model_selection import train_test_split


class FeatureManager:
    """
    Class to manage the features from the PixelHop units in the GUSL pipeline
    and also the management of the XGBoosts.

    :param training_features: list of all of the training features from the PixelHop units
    :param targets: target values for the training data
    :param block_sizes: the block sizes to use, should be in reverse order compared to depth i.e.
    block_sizes[0] corresponds to the depth 0 training features
    :param n_bins: list of the bin sizes to use for the RFT procedure
    :param level_sizes: the size of the data at each level of the pipeline
    :param val_size: the size of the validation set to use
    """

    def __init__(
        self,
        training_features: List[dict],
        block_sizes: List[int],
        n_bins: List[int],
        level_sizes: List[int],
    ):
        self.training_features = training_features
        self.block_sizes = block_sizes
        self.n_bins = n_bins
        self.level_sizes = level_sizes
        self.predictions = {}
        self.rft_features = {}
        self.transformed_features = {}
        self.xgboosts = {}

    def fit_rft(
        self,
        targets: np.array,
        depth: int,
        val_size: float = 0.2,
        calculate_mean: bool = True,
    ):
        """
        Fits RFTs for a given depth

        :param depth: the depth to get the features from
        :param targets: the targets
        :param calculate_mean: whether to calculate the mean when we create the patches
        """
        # we make the blocks of the targets
        self.rft_features[depth] = []
        n_bins = self.n_bins[depth]
        reshaped_targets = self.make_blocks(
            targets, self.block_sizes[depth], calculate_mean=calculate_mean
        )

        print(f"{'='*50} Fitting RFTs for depth {depth}")
        # pdb.set_trace()
        for phop_features in self.training_features:
            feature = phop_features[f"hop_{depth}"]
            N, H, W, C = feature.shape
            X_t, X_v, y_t, y_v = train_test_split(
                feature, reshaped_targets, test_size=val_size
            )
            X_t = X_t.reshape(-1, C)
            X_v = X_v.reshape(-1, C)
            y_t = y_t.reshape(-1, 1)
            y_v = y_v.reshape(-1, 1)
            rft = RFT()
            rft.fit(X_t, y_t, X_v, y_v, n_bins, remove_outliers=True)
            self.rft_features[depth].append(rft)
        print(f"Done fitting RFTs for depth {depth}")

    def transform_rfts(self, n_selected: List[int], depth: int) -> np.array:
        """
        Transforms to get the RFT features based on the depth argument

        :param n_selected: the number of features to select for each rft, length should equal # of PU units
        :param depth: the depth to grab the features from
        :return features: the selected features
        """
        self.transformed_features[depth] = []
        print(f"{'='*50} Transforming RFT features for depth {depth}")
        for rft, phop_features, to_select in zip(
            self.rft_features[depth], self.training_features, n_selected
        ):
            feature = phop_features[f"hop_{depth}"]
            N, H, W, C = feature.shape
            feature = feature.reshape(-1, C)
            feature = rft.transform(feature, to_select)
            self.transformed_features[depth].append(feature)

        self.transformed_features[depth] = np.concatenate(
            self.transformed_features[depth], axis=-1
        )
        print(f"Done transforming RFTs for depth {depth}")

    def plot_rft_curves(self, depth: int, save_path: str, title: str, names: List[str]):
        """
        Plots the RFT features for a given depth

        :param depth: the depth to plot the features for
        :param save_path: the path to save the figures
        :param title: the title of the figure
        :param names: names to name the figures
        """
        rfts = self.rft_features[depth]
        fig = plt.figure(figsize=(10, 10))
        for i, rft in enumerate(rfts):
            x, y = rft.get_curve("train")
            plt.plot(x, y, label=names[i])
        plt.xlabel("Sorted Feature Index")
        plt.ylabel(f"RFT loss {rft.loss_name}")
        plt.title(title)
        plt.legend()
        plt.savefig(save_path)

    def plot_rft_train_val(self, depth: int, feat_ind: int, save_path: str, title: str):
        """
        Plots the RFT feature rankings in the training and validation set

        :param depth: the depth to plot the features
        :param feat_ind: the index at the depth to plot the rft for
        :param save_path: the path to save the figures
        :param title: the title of the figure
        """
        rfts = self.rft_features[depth][feat_ind]
        train_rank = [i for i in rfts.dim_loss.keys()]
        val_rank = [i for i in rfts.dim_loss_v.keys()]
        y_values = [val_rank.index(i) for i in train_rank]

        fig = plt.figure(figsize=(10, 10))
        plt.plot(np.arange(len(train_rank)), y_values, ".")
        plt.xlabel("Sorted Feature Index in Training")
        plt.ylabel("Ranking in Validation")
        plt.title(title)
        plt.savefig(save_path)

    def make_blocks(
        self, X: np.array, block: int, calculate_mean: bool = False
    ) -> np.array:
        """
        Makes blocks of the specified block sizes, this helps with creating the
        "coarse" representations of our target values to supervise the levels of the
        GUSL pipeline. Ensure that H//block and Y//block == the shape of the features
        at the relevant level of the RFT.

        :param X: the input array
        :param block: the block size to calculate
        :param calculate_mean: whether to compress down with mean
        :return blocks: the resulting blocks
        """
        N, H, W = X.shape
        blocks = X.reshape(-1, H // block, block, W // block, block)
        blocks = blocks.transpose((0, 1, 3, 2, 4))
        if calculate_mean:
            blocks = np.mean(blocks, axis=(3, 4))

        return blocks

    def train_xgboost(
        self, targets: np.array, depth: int, val_size: float, xgb_args: dict
    ):
        """
        Trains an XGboost and stores it. The data used is dependent on the depth argument.

        :param targets: the targets, un-reshaped
        :param depth: the depth of the GUSL model, decides what input features we use and the shape of y
        :param val_size: the size to use for th evalidation set
        :param xgb_args: the arguments to use for the SingleXGBoost object
        """
        features = self.transformed_features[depth]
        sxgb = SingleXGBoost(**xgb_args)

        # if we are at the lowest depth, we can train the xgboost directly
        if depth == len(self.level_sizes) - 1:
            y = self.make_blocks(targets, self.block_sizes[depth], calculate_mean=True)
            y = y.flatten()
            X_train, X_val, y_train, y_val = train_test_split(
                features, y.flatten(), test_size=val_size
            )
        else:
            # otherwise, we must train on the residuals from the previous layer
            residuals, y_prev_pred, y_prev_pred_resize = self.get_residuals(
                targets, depth + 1
            )
            X_train, X_val, y_train, y_val = train_test_split(
                features, residuals.flatten(), test_size=val_size
            )

        train = xgb.DMatrix(X_train, label=y_train)
        val = xgb.DMatrix(X_val, label=y_val)
        sxgb.fit(train, val)
        self.xgboosts[depth] = sxgb

    def get_residuals(self, targets: np.array, depth: int) -> Tuple[np.array]:
        """
        Get the residuals from the previously trained XGBoost, note, only
        applies for depth != lowest depth.

        :param targets: the targets to use
        :param depth: the depth of the current xgboost, depth + 1 is the previous xgboost
        :param (residuals, y_prev_pred, y_prev_pred_resize): the residuals from the previous model, the predictions from
        the previous model, and the resized (interpolated) values from that previous prediction
        """
        # resize the targets so they are at the appropriate size for the current depth
        block_size = self.block_sizes[depth - 1]
        targets = self.make_blocks(targets, block_size, calculate_mean=True)
        y_prev_pred = self.get_prediction(depth)
        y_prev_pred_resize = []
        # loop through the previous predictions, interpolate, and then size them up to the right level
        for i in range(len(y_prev_pred)):
            y_prev_pred_resize.append(
                cv2.resize(
                    y_prev_pred[i],
                    (self.level_sizes[depth - 1], self.level_sizes[depth - 1]),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            )

        y_prev_pred_resize = np.array(y_prev_pred_resize)
        residuals = targets - y_prev_pred_resize
        return residuals, y_prev_pred, y_prev_pred_resize

    def get_prediction(self, depth: int) -> np.array:
        """
        Get the prediction of the xgboost depending on the depth argument.

        :param depth: the depth of the xgboost
        :param prediction: the prediction of the xgboost
        """
        # TODO: Rewrite this to ensure that the we don't use as much memory
        # pdb.set_trace()
        deepest = len(self.level_sizes) - 1
        features = self.transformed_features[deepest]
        xgb_curr = self.xgboosts[deepest]
        X = xgb.DMatrix(features)
        predictions = xgb_curr.predict(X)
        predictions = predictions.reshape(
            -1, self.level_sizes[deepest], self.level_sizes[deepest]
        )

        # if we are not at the lowest level, then we must chain the previous predictions together
        if depth < deepest:
            last_prediction = predictions
            for i in range(deepest - 1, depth - 1, -1):
                features = self.transformed_features[i]
                xgb_curr = self.xgboosts[i]
                X = xgb.DMatrix(features)
                current_prediction = xgb_curr.predict(X)
                current_prediction = current_prediction.reshape(
                    -1, self.level_sizes[i], self.level_sizes[i]
                )
                last_prediction_resize = []
                for j in range(len(last_prediction)):
                    last_prediction_resize.append(
                        cv2.resize(
                            last_prediction[j],
                            (self.level_sizes[i], self.level_sizes[i]),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    )
                last_prediction_resize = np.array(last_prediction_resize)
                predictions = current_prediction + last_prediction_resize

        return predictions

    def save(self, path: str):
        """
        Saves the feature extractor object
        """
        self.training_features = None
        self.transformed_features = {}
        with open(path, "wb") as f:
            pickle.dump(self, f)
            f.close()
