# Laurence Palmer
# palmerla@usc.edu
# 2024.09

from helpers.helpers import *
from modules.pixelhop import PixelHop
from modules.single_xgboost import SingleXGBoost
from modules.rft import RFT
from modules.LNT import LNT
from argparse import ArgumentParser
from typing import *
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import cv2
import os
import yaml
import gc
import pdb
import datetime


DATADIR_PATH = "data/128x128_patches_RGB_indoor_outdoor/outdoor/"
HAZYDIR_NAME = "hazy"
CLEARDIR_NAME = "gt"
MODEL_ARCHS = "model_archs.yaml"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", default="cascaded_pixelhop_depth_7", type=str)
    parser.add_argument("--n_samples", default=50, type=int)
    parser.add_argument("--channel", default="Y", type=str)
    parser.add_argument(
        "--model_path", default=f"models/{datetime.datetime.now()}/", type=str
    )
    parser.add_argument("--color_space", default="YUV", type=str)
    parser.add_argument("--load_pu", default=False, type=bool)
    parser.add_argument("--load_rft", default=False, type=bool)
    parser.add_argument("--load_lnt", default = False, type = bool)
    parser.add_argument("--load_xgb", default=False, type=bool)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--make_plots", default=True, type=bool)
    parser.add_argument("--training_depth", default=-1, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--val_size", default=0.2, type=float)
    parser.add_argument("--pu_train_size", default = 0.5, type = float)
    parser.add_argument("--batch_size", default = 0.1, type = float)
    return vars(parser.parse_args())


def get_model_args(file_path: str, model_arch: str) -> dict:
    """
    Loads the specified yaml file and gets the arguments for that model architecture.

    :param file_path: the path to the yaml file with the model architectures
    :param model_arch: string value that corresponds to the name of the architecture
    :return: json object with the specified model arch/args otherwise empty dict
    """
    with open(file_path, "r") as f:
        archs = yaml.safe_load_all(f)
        for a in archs:
            if a["name"] == model_arch:
                return a
    return dict()


def process_data(
    color_space: str, n_samples: int
) -> Tuple[np.array, np.array]:
    """Just gets the path to the data"""
    X, y = get_data_batched(DATADIR_PATH, CLEARDIR_NAME, HAZYDIR_NAME)
    return X[:n_samples], y[:n_samples]


def apply_pixelhops_noncascaded(
    X: List[str],
    train_size: float, 
    model_depth: int,
    block_sizes: List[int],
    pixelhop_units: List[PixelHop],
    trained_pixelhop_units: List[Dict[int, PixelHop]],
    training_depth: int,
    save_path: str,
    model_args: Dict
) -> List[Dict[int, np.array]]:
    """
    Applies the pixelhop units in a non-cascaded fashion by
    resizing the input image.

    :param X: the input data
    :param train_size: how many samples to train the PUs on as a percentage of the total samples
    :param model_depth: how deep the model goes, determines the block sizes
    :param pixelhop_units: the pixelhop units to fit and transform
    :param trained_pixelhop_units: the pixelhop units that are already trained
    :param training_depth: a specific training depth to train the model, -1 means train all
    :param save_path: the path to save the PUs if we didn't load them up
    :return transformed: the input data X transformed by the Pixelhop units given in pixelhop_units
    """
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)
    N = round(len(X) * train_size)
    X_train = get_batch(X, indexes[:N], model_args)

    new_pixelhop_units = [{}]*len(pixelhop_units)
    for i in range(model_depth):
        # we train the pixelhop on the given block size if training_depth is given or its the -1 value
        if training_depth < 0 or (training_depth >= 0 and i == training_depth):
            resized = make_blocks(X_train, block_sizes[i], calculate_mean=True) # train on the subsample of training data
            resized = np.expand_dims(resized, axis=-1)
            for j in range(len(pixelhop_units)):
                pu = copy.deepcopy(pixelhop_units[j])
                pu.fit(resized)
                new_pixelhop_units[j][i] = pu
    
    # add the trained pixelhop units
    for i, trained_pu in enumerate(trained_pixelhop_units):
        for j in trained_pu.keys():
            new_pixelhop_units[i][j] = trained_pu[j]

    X_train = None
    gc.collect()
    X = get_batch(X, np.arange(len(X)), model_args)

    # tranform the input data
    transformed = [{}] * len(pixelhop_units)
    for i, pu in enumerate(new_pixelhop_units):
        for depth in pu.keys():
            resized = make_blocks(X, block_sizes[depth], calculate_mean = True)
            resized = np.expand_dims(resized, axis=-1)
            t = pu[depth].transform(resized)

            if model_args.get("feat_concat"):
                t = feature_concat(t, -1)
            
            # save so we don't have to keep it in memory
            path_to_file = save_path + f"tmp_{i}_{depth}_feat"
            with open(path_to_file, "wb") as f: 
                pickle.dump(t, f)
                f.close()
            transformed[i][depth] = path_to_file

    # save the pixelhop units
    for i in range(len(new_pixelhop_units)):
        for depth in new_pixelhop_units[i].keys():
            # i is the pixelhop unit number
            new_pixelhop_units[i][depth].save(
                save_path
                + f"PU_{i}_{new_pixelhop_units[i][depth].hop_args[0]['window']}_{depth}"
            ) 

    pixelhop_units = None
    gc.collect()
    return transformed


def apply_pixelhops_cascaded(
    X: np.array, y, train_size: float, batch_size: float, pixelhop_units: List[PixelHop], load_pu: bool, save_path: str, model_args: Dict
) -> List[Dict[int, np.array]]:
    """
    Applies the Pixelhops in cascaded manner in batches, i.e. no resizing done, but rather pooling operations. 
    Runs a RFT procedure in the process to help with storage blow up. 

    :param X: the input data
    :param y: the input targets
    :param train_size: how many samples to train the PUs on as a percentage of the total samples
    :param batch_size: the size of the batch to use when we make the transformations
    :param y: the target values Xs
    :param pixelhop_units: the Pixelhop units to apply to the input data
    :param load_pu: whether the Pixelhop units were loaded or not, controls whether we save the fitted PUs
    :param save_path: the path to save the PU units to if required
    :param model_args: the model arguments
    :return transformed: the data transformed by the PU units
    """
    # set up a random sample of train_size*len(X) to train the kernels on
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)
    N = round(len(X) * train_size)
    X_train = get_batch(X, indexes[:N], model_args)
    X_train = np.expand_dims(X_train, axis=-1)

    N_batch = round(len(X)*batch_size)
    transformed = []
    for i in range(len(pixelhop_units)):
        if not load_pu:
            pixelhop_units[i].fit(X_train)
        
        # now we run through each of the samples in a given batch and transform them
        path_to_files = DefaultDict(list)
        for j in range(0, len(X), N_batch):
            to_get = np.arange(j, j + N_batch) 
            Xb = get_batch(X, to_get, model_args)
            Xb = np.expand_dims(Xb, axis = -1)
            t = pixelhop_units[i].transform(Xb)

            if model_args.get("feat_concat"):        
                # save so we don't have to keep it in memory since feat_concat explodes mem
                for depth in t.keys():
                    concat = feature_concat_one_step(t[depth])
                    t[depth] = None

                    path_to_file = save_path + f"tmp_{i}_{depth}_feat_batch_{j}_{j+N_batch}"
                    with open(path_to_file, "wb") as f: 
                        pickle.dump(concat, f)
                        f.close()
                    path_to_files[depth].append(path_to_file)
                    gc.collect()
        
        transformed.append(path_to_files)

    for i in range(len(pixelhop_units)):
        pixelhop_units[i].save(
            save_path + f"PU_{pixelhop_units[i].hop_args[0]['window']}"
        )
    pixelhop_units = None
    gc.collect()
    return transformed


def apply_rfts(
    Xs: List[Dict[int, List[str]]],
    y: List[str],
    train_index: np.array,
    val_index: np.array,
    rft_units: List[Dict[int, RFT]],
    block_sizes: List[int],
    n_bins: List[int],
    n_selected: List[int],
    training_depth: int,
    save_path: str,
    make_plots: bool,
    model_args: Dict
) -> List[Dict[int, np.array]]:
    """
    Applies the RFTs to the given data

    :param Xs: the input data, list of dictionaries, where each entry of the list is the output of
    a Pixelhop and dictionary has (depth, output) pairs.
    :param y: the target valuss of the input data
    :param train_index: the indices for the training set
    :param val_index: the indices for the validation set
    :param rft_units: list of all the rft units we want to apply
    :param training_depth: whether to select a specific output depth to apply the rft to, -1 means we'll fit all of them
    :param log
    :param n_bins: the number of bins to use for the RFT
    :param save_path: the path to save the RFTs if they weren't loaded
    :return transformed: output of fitting and transforming based on the rft arguments given
    """
    y = get_batch(y, np.arange(len(y)), model_args)
    transformed = []
    new_rft_units = []
    print(f"{'='*50} Fitting RFTs")
    for i in range(len(Xs)):  # range over each pixelhop
        tmp = dict()
        for k in Xs[i].keys():  # range over the raw features of the given pixelhop, each k is a depth
            if training_depth < 0 or (training_depth >= 0 and training_depth <= k): # greater than or equal to since lower levels should be loaded 
                print(f"Fitting RFT for PU unit {i} and depth {k}")
                feature = []
                for f in Xs[i][k]:
                    feature.append(load_feature(f))
                feature = np.concatenate(feature, axis = 0)
                # feature = load_feature(Xs[i][k])[k if model_args.get("cascaded") else 0] # might be hacky
                reshaped_targets = make_blocks(y, block_sizes[k], calculate_mean=True)
                N, H, W, C = feature.shape
                X_t = feature[train_index]
                X_v = feature[val_index]
                y_t = reshaped_targets[train_index]
                y_v = reshaped_targets[val_index]
                X_t = X_t.reshape(-1, C)
                X_v = X_v.reshape(-1, C)
                y_t = y_t.reshape(-1, 1)
                y_v = y_v.reshape(-1, 1)

                rft = copy.deepcopy(rft_units[i][k])

                # if we are at the intended training depth then we have to fit the RFT otherwise we can transform
                if training_depth == k:
                    rft.fit(X_t, y_t, X_v, y_v, n_bins[k], remove_outliers=True)
                tmp[k] = rft.transform(
                    feature.reshape(-1, feature.shape[-1]), n_selected[k]
                )
                new_rft_units.append((rft, k, i))
    
                path_to_file = save_path + f"tmp_{i}_{k}_feat"
                with open(path_to_file, "wb") as f: 
                    pickle.dump(tmp[k], f)
                    f.close()
                tmp[k] = path_to_file
        
        transformed.append(tmp)

    if make_plots:
        generate_rft_plots(new_rft_units, save_path)

    for i in range(len(new_rft_units)):
        rft, depth, pu_unit = new_rft_units[i]
        with open(save_path + f"RFT_{depth}_{pu_unit}", "wb") as f:
            pickle.dump(rft, f)
            f.close()
    new_rft_units = None
    print(f"{'='*50} Done fitting RFTs")
    y = None
    gc.collect()
    return transformed

def generate_rft_plots(rfts: List[Tuple[RFT, int, int]], save_path: str):
    """
    Creates plots for an RFT procedure including joint validation and RFT curve

    :param rfts: List of tuples with the RFT as the first entry, the depth the RFT was applied to
    as the second argument, and the associated pixelhop unit as the last entry
    :param save_path: where to save the plots for the RFT
    """

    # organize everything
    grouped_rfts = DefaultDict(list)
    if len(rfts[0]) == 3:  # with three these are the RFTs that take in raw features
        name = "rft_curve"
        for rft, depth, pu_num in rfts:
            grouped_rfts[depth].append((rft, pu_num))
    else:  # with 4 these are the RFTs we use between successive LNTs
        name = "rft_curve_lnt"
        for rft, depth, pu_num, round in rfts:
            grouped_rfts[depth].append((rft, f"{pu_num}_{round}"))

    for k in grouped_rfts.keys():
        rft_curve_fig = plt.figure()
        for rft, pu_num in grouped_rfts[k]:
            x, y = rft.get_curve("train")
            plt.plot(x, y, label=f"{pu_num}")
            plot_rft_train_val(
                rft,
                save_path + f"jr_{k}_{pu_num}",
                f"Joint Ranking Curve For RFT at Depth {k}, PU {pu_num}",
            )
        plt.xlabel("Sorted Feature Index")
        plt.ylabel(f"RFT loss {rft.loss_name}")
        plt.title(f"RFT Curves for Depth {k}")
        plt.legend()
        plt.savefig(save_path + f"{name}_{k}")
        plt.close()


def plot_rft_train_val(rft: RFT, save_path: str, title: str):
    """
    Plots joint ranking curve for a given rft

    :param rft: the RFT to make the plot for
    :param save_path: where to save it
    :param title: the title of the RFT curve
    """
    train_rank = [i for i in rft.dim_loss.keys()]
    val_rank = [i for i in rft.dim_loss_v.keys()]
    y_values = [val_rank.index(i) for i in train_rank]

    fig = plt.figure(figsize=(10, 10))
    plt.plot(np.arange(len(train_rank)), y_values, ".")
    plt.xlabel("Sorted Feature Index in Training")
    plt.ylabel("Ranking in Validation")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def apply_lnts(
    Xs: List[Dict[int, np.array]],
    y: List[str],
    lnt: Dict[int, Dict[int, LNT]],
    rfts: Dict[int, Dict[int, RFT]],
    train_index: np.array,
    val_index: np.array,
    n_rounds: int,
    n_selected: Dict[int, List[int]],
    n_bins: Dict[int, List[int]],
    block_sizes: List[int],
    training_depth: int,
    save_path: str,
    make_plots: bool,
) -> List[Dict[int, np.array]]:
    """
    This module applies LNTs, can also run the compound LNT procedure if the relevant arguments are given

    :param Xs: the input features, the length of the list is the number of PU units and the keys in the dict are the depth of the features
    :param y: the target values
    :param lnt: the LNTs to apply, might have some that are loaded in there too, indexed by depth first and the round second
    :param rfts: the RFTs to use between successive LNTs, same indexing scheme as above
    :param train_index: the indices to use for training
    :param val_index: the indices to use for validation
    :param n_rounds: how many times to apply the LNT consequently, the number of rounds
    :param n_selected: the number of selected features that we want to keep between successive lnts/rfts, indexed by depth
    :param n_bins: the number of bins to use for the RFT procedures, indexed by depth
    :param block_sizes: the size of the dimension at a given depth
    :param training_depth: the specific depth to train on if given
    :param save_path: where to save the LNTs
    :param make_plots: whether to plot the compound rft procedures between the LNTs
    :return transformed: the transformed features for a given depth
    """
    y = get_batch(y, np.arange(len(y)), model_args)

    transformed = []
    fitted_lnts = []
    fitted_rfts = []
    for i in range(len(Xs)): # TODO: this might be an issue when we have different kernel sizes/multiple PU units, consider concatenating them together
        tmp = dict()
        for k in Xs[i].keys():
            if training_depth < 0 or (training_depth >= 0 and training_depth <= k):
                feature = load_feature(Xs[i][k])
                y_reshaped = make_blocks(y, block_sizes[k], calculate_mean=True)
                N, H, W = y_reshaped.shape
                y_train = y_reshaped[train_index].flatten()
                y_val = y_reshaped[val_index].flatten()
                y_reshaped = y_reshaped.flatten()
                feature_cp = feature # copy.deepcopy(feature)

                print(
                    f"{'='*50} Fitting LNTs for PU {i} depth {k} (n_rounds: {n_rounds})"
                )
                for j in range(n_rounds):  # compound lnt means n_rounds > 0
                    print(f"Round {j} input feature shape {feature_cp.shape}")
                    lnt_cp = copy.deepcopy(lnt[k][j])
                    if training_depth == k: # training_depth == k means we have to fit the LNT
                        lnt_cp.fit(feature_cp, y_reshaped)

                    fitted_lnts.append((lnt_cp, k, i, j))
                    lnt_out = lnt_cp.transform(feature_cp)

                    # we have to reshape the transformed back to the original shape to provide the correct train/val indices
                    _, C = lnt_out.shape
                    lnt_out = lnt_out.reshape(N, H, W, -1)
                    rft = copy.deepcopy(rfts[k][j])
                    lnt_out_t = lnt_out[train_index].reshape(-1, C)
                    lnt_out_v = lnt_out[val_index].reshape(-1, C)
                    lnt_out = lnt_out.reshape(-1, C)
                    
                    if training_depth == k: # training_depth == k means we need to fit the RFTs
                        rft.fit(
                            lnt_out_t,
                            y_train,
                            lnt_out_v,
                            y_val,
                            n_bins[k][j],
                            remove_outliers=True,
                        )
                    fitted_rfts.append((rft, k, i, j))
                    rft_out = rft.transform(lnt_out, n_selected[k][j])
                    feature_cp = np.concatenate([feature, rft_out], axis=-1)
                    lnt_out = None
                    gc.collect()

                tmp[k] = rft_out
                print(f"{'='*50} Done fitting LNTs")
            
        transformed.append(tmp)

    if make_plots:
        generate_rft_plots(fitted_rfts, save_path)

    for i in range(len(fitted_lnts)):
        lnt, depth, pu_unit, round = fitted_lnts[i]
        with open(save_path + f"LNT_{depth}_{pu_unit}_{round}", "wb") as f:
            pickle.dump(lnt, f)
            f.close()

    for i in range(len(fitted_rfts)):
        rft, depth, pu_unit, round = fitted_rfts[i]
        with open(save_path + f"RLFNT_{depth}_{pu_unit}_{round}", "wb") as f:
            pickle.dump(rft, f)
            f.close()

    fitted_lnts = None
    fitted_rfts = None

    gc.collect()
    return transformed

def generate_residuals(Xs: Dict[int, np.array], y: np.array, prev_xgbs: Dict[int, SingleXGBoost], block_sizes: List[int], level_sizes: List[int], max_depth: int, training_depth: int) -> np.array:
    """
    Gets residuals from the previously trained XGBoosts.
    
    :param features: the features 
    :param y: the targets 
    :param prev_xgbs: the previously trained XGBs indexed by their depth
    :param block_sizes: the block sizes for reshaping
    :param level_sizes: the shape of the inputs at each depth 
    :param max_depth: the maximum depth of the pipeline
    :param training_depth: the current training depth
    :return resid: the residuals
    """
    depths = list(prev_xgbs.keys())
    depths.sort(reverse = True)

    # pop off the maximum and get all the necessary args
    depths.pop(0)
    level_size = level_sizes[max_depth - 1]
    features = Xs[max_depth - 1]
    sxgb = prev_xgbs[max_depth - 1]
    X = xgb.DMatrix(features)
    
    # get the predictions and calculate the residual for the max depth
    curr_pred = sxgb.predict(X)
    curr_pred = curr_pred.reshape(-1, level_size, level_size)
    prev_pred = curr_pred
    pred = curr_pred

    # now we are predicting the residuals
    for d in depths:
        level_size = level_sizes[d]
        features = Xs[d]
        sxgb = prev_xgbs[d]
        X = xgb.DMatrix(features)
        curr_pred = sxgb.predict(X)
        curr_pred = curr_pred.reshape(-1, level_size, level_size)
        prev_pred_resize = size_up(prev_pred, (level_size, level_size), cv2.INTER_LANCZOS4)
        pred = curr_pred + prev_pred_resize
        prev_pred = curr_pred
        

    y_reshaped = make_blocks(y, block_sizes[training_depth], calculate_mean = True)
    pred = size_up(pred, (level_sizes[training_depth], level_sizes[training_depth]), cv2.INTER_LANCZOS4)

    resid = y_reshaped - pred
    return resid


def train_xgboosts(
    Xs: Dict[int, np.array],
    y: List[str],
    xgb_args: Dict,
    train_index: List[int],
    val_index: List[int],
    training_depth: int,
    max_depth: int,
    block_sizes: List[int],
    level_sizes: List[int],
    prev_xgbs: Dict[int, SingleXGBoost],
    save_xgb: bool,
    save_path: str,
    model_args: str
) -> SingleXGBoost:
    """
    Trains 1 XGBoost given the Xs and the ys

    :param Xs: the input data, list where the index corresponds to the depth of the input data
    :param y: the targets
    :param xgb_args: the arguments to create the xgb object
    :param train_index: the training indices
    :param val_index: the validation indices
    :param training_depth: the training depth, controls the input data we choose/block_sizes
    :param max_depth: the maximum depth of the pipeline
    :param block_sizes: the block sizes to use to shape the targets to the correct shape
    :param level_sizes: the size of the input for a given depth
    :param prev_xgbs: the previous xgbs, only used when we are training on residuals
    :param save_xgb: whether to save the new trained xgb or not
    :param save_path: where to save the xgb
    :param model_args: the model arguments
    :return
    """
    sxgb = SingleXGBoost(**xgb_args)
    y = get_batch(y, np.arange(len(y)), model_args)

    # we are training on the lowest depth
    if training_depth == max_depth - 1:
        y_reshaped = make_blocks(y, block_sizes[training_depth], calculate_mean=True)
        N, H, W = y_reshaped.shape
        y_train = y_reshaped[train_index].flatten()
        y_val = y_reshaped[val_index].flatten()
    else: 
        resids = generate_residuals(Xs, y, prev_xgbs, block_sizes, level_sizes, max_depth, training_depth)
        N, H, W = resids.shape
        y_train = resids[train_index].flatten()
        y_val = resids[val_index].flatten()
    
    feature = Xs[training_depth]
    _, C = feature.shape
    feature = feature.reshape(N, H, W, -1)
    X_train = feature[train_index].reshape(-1, C)
    X_val = feature[val_index].reshape(-1, C)

    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)
    sxgb.fit(train, val)

    if make_plots:
        sxgb.plot_learning_curve(
            eval_metric="rmse", path=save_path + f"lc_{training_depth}"
        )
    
    if save_xgb:
        with open(save_path + f"XGB_{training_depth}", "wb") as f: 
            pickle.dump(sxgb, f)
            f.close()
        for k, x in prev_xgbs.items():
            with open(save_path + f"XGB_{k}", "wb") as f: 
                pickle.dump(x, f)
                f.close()

    return sxgb

def model(
    X: List[str],
    y: List[str],
    train_index: np.array,
    val_index: np.array,
    model_args: dict,
    training_depth: int,
    save_path: str,
    load_pu: bool,
    load_rft: bool,
    load_lnt: bool,
    load_xgb: bool,
    load_path: str,
    make_plots: bool = True,
):
    """Run the GDH model"""

    # FEATURE EXTRACTION
    pixelhop_units = []
    pixel_args = model_args.get("pixelhops")

    # get the baseline pixelhop units we will use
    for unit in pixel_args.keys():
        pixelhop_units.append(PixelHop(**pixel_args[unit]))

    trained_pixelhop_units = [{}] *len(pixelhop_units)
    if load_pu: # if we're loading pixelhop units then we are training a higher level
        pu_files = [f for f in os.listdir(load_path) if "PU" in f]
        for pu in pu_files:
            _, pu_unit, window, depth = pu.split("_")
            with open(load_path + "/" + pu, "rb") as f:
                p = pickle.load(f)
                f.close()
            # first list index is the associated pixelhop unit number and key to the dict is the depth
            trained_pixelhop_units[int(pu_unit)][int(depth)] = p

    # cascaded or not
    cascaded = model_args.get("cascaded")
    model_depth = model_args.get("model_depth")
    block_sizes = model_args.get("block_sizes")
    pu_train_size = model_args.get("pu_train_size")
    batch_size = model_args.get("batch_size")
    if cascaded:
        transformed = apply_pixelhops_cascaded(X, pu_train_size, batch_size, pixelhop_units, load_pu, save_path, model_args)
    else:
        transformed = apply_pixelhops_noncascaded(
            X,
            pu_train_size, 
            model_depth,
            block_sizes,
            pixelhop_units,
            trained_pixelhop_units,
            training_depth,
            save_path,
            model_args
        )

    # SUPERVISED FEATURE REDUCTION

    # we have depth * n_pixelhop_units RFTs for the raw features
    rft_units = [{}] * len(pixelhop_units) 
    rfts = model_args.get("rfts")
    # TODO: put in functionality for RFT init arguments
    for i in range(len(pixelhop_units)):
        for depth in range(model_depth):
            rft_units[i][depth] = RFT()

    if load_rft:
        rft_files = [f for f in os.listdir(load_path) if "RFT" in f]
        for rft in rft_files:
            _, depth, pu_num = rft.split("_")
            with open(load_path + "/" + rft, "rb") as f:
                r = pickle.load(f)
                f.close()
            rft_units[int(pu_num)][int(depth)] = r

    n_selected = [
        model_args["rfts"][k]["n_selected"] for k in model_args["rfts"].keys()
    ]
    n_bins = [model_args["rfts"][k]["n_bins"] for k in model_args["rfts"].keys()]
    transformed = apply_rfts(
        transformed,
        y,
        train_index,
        val_index,
        rft_units,
        block_sizes,
        n_bins,
        n_selected,
        training_depth,
        save_path,
        make_plots,
        model_args
    )

    # get the LNT arguments
    lnt_procedure = model_args.get("lnts")
    lnt_args = lnt_procedure.get("lnt_args")
    n_rounds = lnt_procedure.get("n_rounds")
    lnts = DefaultDict(dict)
    lnt_rfts = DefaultDict(dict)

    # initialize LNTs and RFTs 
    for d in range(model_depth):
        for r in range(n_rounds):
            lnts[d][r] = LNT(**lnt_args)
            lnt_rfts[d][r] = RFT()

    # load saved LNT/RFTs if necessary
    if load_lnt: 
        lnt_files = [f for f in os.listdir(load_path) if "LNT" in f]
        for lnt in lnt_files:
            _, depth, pu_unit, round = lnt.split("_")
            with open(load_path + "/" + lnt, "rb") as f:
                l = pickle.load(f)
                f.close()
            lnts[int(depth)][int(round)] = l

        rft_files = [f for f in os.listdir(load_path) if "RLFNT" in f]
        for rft in rft_files:
            _, depth, pu_unit, round = rft.split("_")
            with open(load_path + "/" + rft, "rb") as f:
                r = pickle.load(f)
                f.close()
            lnt_rfts[int(depth)][int(round)] = r
            
    # get fitting arguments for the RFTs between successive LNTs
    rfts_params = lnt_procedure.get("rfts")
    n_selected = {depth: [rfts_params[depth][k].get("n_selected") for k in rfts_params[depth].keys()] for depth in rfts_params.keys()}
    n_bins = {depth: [rfts_params[depth][k].get("n_bins") for k in rfts_params[depth].keys()] for depth in rfts_params.keys()}

    transformed = apply_lnts(
        transformed,
        y,
        lnts,
        lnt_rfts,
        train_index,
        val_index,
        n_rounds,
        n_selected,
        n_bins,
        block_sizes,
        training_depth,
        save_path,
        make_plots,
    )

    # we put all the features at the same depth/spatial resolution together 
    features = DefaultDict(list)
    for d in transformed: 
        for k in d.keys():
            features[k].append(d[k])
    
    for k in features.keys():
        features[k] = np.concatenate(features[k], axis = -1)
        
    # SUPERVISED REGRESSION
    xgboost_args = model_args.get("xgboost")
    max_depth = model_args.get("model_depth")
    level_sizes = model_args.get("level_sizes")
    prev_xgbs = dict()

    if load_xgb: 
        xgb_files = [f for f in os.listdir(load_path) if "XGB" in f]
        for sxgb in xgb_files:
            _, depth = sxgb.split("_")
            with open(load_path + "/" + sxgb, "rb") as f:
                x = pickle.load(f)
                f.close()
            prev_xgbs[int(depth)] = x
            

    # first XGBoost
    sxgb = train_xgboosts(
        features, 
        y, 
        xgboost_args, 
        train_index, 
        val_index, 
        training_depth, 
        max_depth,
        block_sizes, 
        level_sizes,
        prev_xgbs, 
        True, 
        save_path,
        model_args
    )

    return None


if __name__ == "__main__":
    pdb.set_trace()
    cmd_args = parse_arguments()
    model_arch = cmd_args["model_arch"]
    n_samples = cmd_args["n_samples"]
    channel = cmd_args["channel"]
    model_path = cmd_args["model_path"]
    color_space = cmd_args["color_space"]
    load_path = cmd_args["load_path"]
    make_plots = cmd_args["make_plots"]
    training_depth = cmd_args["training_depth"]
    load_pu = cmd_args["load_pu"]
    load_rft = cmd_args["load_rft"]
    load_lnt = cmd_args["load_lnt"]
    load_xgb = cmd_args["load_xgb"]
    seed = cmd_args["seed"]
    val_size = cmd_args["val_size"]
    pu_train_size = cmd_args["pu_train_size"]
    batch_size = cmd_args["batch_size"]

    set_seed(seed)

    model_args = get_model_args(MODEL_ARCHS, model_arch)

    for arg in cmd_args.keys():
        model_args[arg] = cmd_args[arg]

    if color_space == "YUV":
        model_args["color_transform"]= cv2.COLOR_BGR2YUV
    elif color_space == "RGB":
        model_args["color_transform"] = cv2.COLOR_BGR2RGB


    x = os.mkdir(model_path)
    with open(f"{model_path}model_args", "wb") as f:
        pickle.dump(model_args, f)
        f.close()

    X_data, y_data = process_data(color_space, n_samples)
    N = len(X_data)
    N_val = round(N * val_size)
    indexes = np.arange(len(X_data))
    np.random.shuffle(indexes)
    val_indexes = indexes[:N_val]
    train_indexes = indexes[N_val:]

    model(
        X_data,
        y_data,
        train_indexes,
        val_indexes,
        model_args,
        training_depth,
        model_path,
        load_pu,
        load_rft,
        load_lnt,
        load_xgb,
        load_path,
        make_plots,
    )
