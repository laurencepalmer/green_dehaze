# Laurence Palmer
# palmerla@usc.edu
# 2024.08

import numpy as np
import pdb
import gc
import pickle
import matplotlib.pyplot as plt
from typing import *
from sklearn import datasets
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from .saabtransform import SaabTransform


class PixelHop:
    """
    Implementation of PixelHop++ described in the following:

    https://arxiv.org/abs/2002.03141

    Note that we don't apply a reduce arg in the initial layer, thus
    the # of saab_args and hop_args should be equal to the depth
    while the # of reduce_args should be equal to depth - 1

    :param saab_args: arguments for the Saab transform, check the c/w Saab code
    :param hop_args: arguments for the PixelHop units, check the .hop() method
    :param reduce_args: arguments for the block_reduce procedure
    :param depth: the depth of the c/w Saab transform
    """

    def __init__(
        self,
        saab_args: List[dict] = None,
        hop_args: List[dict] = None,
        reduce_args: List[dict] = None,
        depth: int = 1,
        leaf_threshold: float = 0.002,
        discard_threshold: float = 0.0001,
    ):
        self.saab_args = saab_args
        self.hop_args = hop_args
        self.reduce_args = [{}] + reduce_args
        self.depth = depth
        self.leaf_threshold = leaf_threshold
        self.discard_threshold = discard_threshold
        self.cw_layers = {f"layer_{i}": [] for i in range(depth)}
        self.trained = False

    # @staticmethod
    def hop(
        self,
        X: np.array,
        pad: int = 1,
        stride: int = 1,
        window: int = 1,
        method: str = "reflect",
    ) -> np.array:
        """
        Perform the PixelHop on set of images, get the neighbors and reshape
        imgs should have dims (# samples, h, w, channels)

        :param: X: set of images
        :param pad: pad images, default with reflect
        :return: hopped: set of images with neighor pixels in channels

        Does the order that we include the elements matter? The set difference
        between this and the other hop method is none, but the order is simply permuted.
        """

        N, h, w, c = X.shape
        if pad:
            X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode=method)

        X = view_as_windows(X, (1, window, window, c), (1, stride, stride, c))

        return X.reshape(N, h, w, -1)

    def reduce(
        self, X: np.array, pool: int, method: Callable = np.max, use_abs: bool = True
    ) -> np.array:
        """
        Perform a block_reduce,

        :param X: the input array
        :param pool: the pooling method
        :param method: callable to perform the reduce, default is np.max
        :param use_abs: whether to apply absolute value prior to doing the callable
        :return out: the output of the specified block reduce procedure
        """
        if use_abs:
            X = np.absolute(X)

        if type(method) == str:
            if method == "np.max":
                method = np.max
            elif method == "np.min":
                method = np.min
            elif method == "np.mean":
                method = np.mean
            else:
                raise ValueError("Method is not a valid method, choose from np.min, np.max, np.mean")
        
        out = block_reduce(X, (1, pool, pool, 1), method)
        return out

    def cw_layer(
        self,
        X: np.array,
        saab_args: dict,
        reduce_args: dict,
        hop_args: dict,
        layer: int,
    ) -> np.array:
        """
        Single layer of the c/w Saab transform training, return the transformed X

        :param X: the input data
        :param saab_args: arguments to use for the Saab on the intermediate nodes
        :param reduce_args: arguments to use for the block reduce in the intermediate nodes
        :param hop_args: arguments to use for the hop in the intermediate nodes
        :param layer: the last layer of the cw saab transform
        :return transformed: the transformed data
        """
        N, h, w, c = X.shape
        X = np.moveaxis(X, -1, 0)
        last_layer = self.cw_layers[f"layer_{layer}"]
        output = []
        ct = -1

        # loop through all of the saab transforms from the previous layer
        for i in range(len(last_layer)):
            curr_saab = last_layer[i]
            prev_bias = curr_saab.bias
            energies = curr_saab.energy
            saab_args["prev_bias"] = prev_bias
            # print(f"Current Layer {i}: {curr_saab}, Previous Bias: {prev_bias}, Energies: {energies}")

            for j in range(len(energies)):
                curr_energy = energies[j]

                # leaf node, ordered in descending order so we may break once we encounter one less than
                if curr_energy < self.leaf_threshold:
                    break
                else:
                    ct += 1

                # non-leaf node
                tmp = X[ct].reshape(N, h, w, 1)
                tmp = self.reduce(tmp, **reduce_args)
                tmp = self.hop(tmp, **hop_args)
                saab = SaabTransform(**saab_args)
                saab.fit(tmp)
                saab.energy *= curr_energy
                _ = saab.discard(self.discard_threshold)
                tmp = saab.transform(tmp)
                self.cw_layers[f"layer_{layer+1}"].append(saab)
                tmp = tmp[:, :, :, saab.energy > self.leaf_threshold]
                output.append(tmp)

                # clean up
                tmp = None
                gc.collect()

        output = np.concatenate(output, axis=-1)
        return output

    def cw_layer_transform(
        self, X: np.array, reduce_args: dict, hop_args: dict, layer: int
    ) -> np.array:
        """
        Single layer of the c/w Saab transform, output is the layer of the trained Saab transforms

        :param X: the input data
        :param reduce_args: arguments to use for the block reduce in the intermediate nodes
        :param hop_args: arguments to use for the hop in the intermediate nodes
        :param layer: the last layer of the channel wise saab transform
        :return output: the transformed data
        """
        N, h, w, c = X.shape
        X = np.moveaxis(X, -1, 0)
        last_layer = self.cw_layers[f"layer_{layer}"]
        this_layer = self.cw_layers[f"layer_{layer + 1}"]
        output = []
        ct = -1
        cur_saab_ct = 0

        for i in range(len(last_layer)):
            curr_saab = last_layer[i]
            energies = curr_saab.energy
            # print(f"Current Layer {i}: {curr_saab}, Previous Bias: {prev_bias}, Energies: {energies}")

            for j in range(len(energies)):
                curr_energy = energies[j]

                # leaf node, ordered in descending order so we may break once we encounter one less than
                if curr_energy < self.leaf_threshold:
                    # update ct to ensure we are selecting the right channel
                    ct += len(energies) - j
                    break
                else:
                    ct += 1

                tmp = X[ct].reshape(N, h, w, 1)
                tmp = self.reduce(tmp, **reduce_args)
                tmp = self.hop(tmp, **hop_args)
                saab = this_layer[cur_saab_ct]
                tmp = saab.transform(tmp)
                output.append(tmp)

                tmp = None
                gc.collect()
                cur_saab_ct += 1

        output = np.concatenate(output, axis=-1)
        return output

    def fit(self, X: np.array):
        """
        Fits a c/w Saab transform

        :param X: output of hop method
        :return transformed: output of saab transform
        """
        print(f"{'='*50} Layer 0")

        init_saab_args = self.saab_args[0]
        init_hop_args = self.hop_args[0]

        X = self.hop(X, **init_hop_args)
        N, h, w, c = X.shape
        saab = SaabTransform(**init_saab_args)
        saab.fit(X)
        _ = saab.discard(self.discard_threshold)
        X = saab.transform(X)
        self.cw_layers["layer_0"].append(saab)
        print(f"Layer 0 output {X.shape}")

        for i in range(1, self.depth):
            reduce_args = self.reduce_args[i]
            saab_args = self.saab_args[i]
            hop_args = self.hop_args[i]

            print(f"{'='*50} Layer {i}")
            X = self.cw_layer(X, saab_args, reduce_args, hop_args, i - 1)
            print(f"Layer {i} output {X.shape}")

        print(f"Done fitting")
        self.trained = True

    def transform(self, X: np.array) -> np.array:
        """
        Perform the c/w saab transform on the kernels that were fit

        :param X: the input
        :return transformed: the transformed output
        """
        assert self.trained, "The PixelHop unit has not been trained yet"
        print(f"{'='*50} Transforming Input")
        init_saab_args = self.saab_args[0]
        init_hop_args = self.hop_args[0]

        out = {}
        transformed = self.hop(X, **init_hop_args)
        N, h, w, c = X.shape
        saab = self.cw_layers["layer_0"][0]
        transformed = saab.transform(transformed)
        out[0] = transformed
        print(f"{'='*50} Layer 0 output {transformed.shape}")

        for i in range(1, self.depth):
            reduce_args = self.reduce_args[i]
            hop_args = self.hop_args[i]

            transformed = self.cw_layer_transform(
                transformed, reduce_args, hop_args, i - 1
            )
            out[i] = transformed
            print(f"{'='*50} Layer {i} output {transformed.shape}")

        return out

    def save(self, path: str):
        """
        Saves the PixelHop unit to the requested path

        :param path: where to save the model
        """

        with open(path, "wb") as f:
            pickle.dump(self, f)
            f.close()
