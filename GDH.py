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


DATADIR_PATH = "data/128x128_patches_RGB_indoor_outdoor/indoor/"
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
    parser.add_argument("--load_xgb", default=False, type=bool)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--make_plots", default=True, type=bool)
    parser.add_argument("--training_depth", default=-1, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--val_size", default=0.2, type=float)
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
    color_space: str, channel: str, n_samples: int
) -> Tuple[np.array, np.array]:
    """Processes data based on args given"""

    if color_space == "YUV":
        colorspace = cv2.COLOR_BGR2YUV
    elif color_space == "RGB":
        colorspace = cv2.COLOR_BGR2RGB

    # other options as well, look at helpers
    data_args = {
        "color_transform": {"colorspace": colorspace},
    }
    X, y = get_data(data_args, DATADIR_PATH, CLEARDIR_NAME, HAZYDIR_NAME)
    # y channel
    X_1, y_1 = split_channels(X, y, 0)
    # u channel
    X_2, y_2 = split_channels(X, y, 1)
    # v channel
    X_3, y_3 = split_channels(X, y, 2)

    if channel == "Y" or channel == "R":
        X_data = X_1
        y_data = y_1
    elif channel == "U" or channel == "G":
        X_data = X_2
        y_data = y_2
    elif channel == "V" or channel == "B":
        X_data = X_3
        y_data = y_3

    X_data = X_data[:n_samples]
    y_data = y_data[:n_samples].squeeze(-1)
    gc.collect()
    return X_data, y_data


def apply_pixelhops_noncascaded(
    X: np.array,
    model_depth: int,
    block_sizes: List[int],
    pixelhop_units: List[PixelHop],
    training_depth: int,
    load_pu: bool,
    save_path: str,
) -> List[Dict[int, np.array]]:
    """
    Applies the pixelhop units in a non-cascaded fashion by
    resizing the input image.

    :param X: the input data
    :param model_depth: how deep the model goes, determines the block sizes
    :param pixelhop_units: the pixelhop units to fit and transform
    :param training_depth: a specific training depth to train the model, -1 means train all
    :param load_pu: whether the pixelhops were loaded or not, controls whether we save them
    :param save_path: the path to save the PUs if we didn't load them up
    :return transformed: the input data X transformed by the Pixelhop units given in pixelhop_units
    """
    transformed = [{}] * len(pixelhop_units)
    X = X.squeeze(-1)

    # list to store the non-cascaded pixelhops, should end up with model_depth*len(pixelhop_units) if training all
    new_pixelhop_units = []
    for i in range(model_depth):
        # we train the pixelhop on the given block size if training_depth is given or its the -1 value
        if training_depth < 0 or (training_depth >= 0 and i == training_depth):
            resized = make_blocks(X, block_sizes[i], calculate_mean=True)
            resized = np.expand_dims(resized, axis=-1)
            tmp = []
            for j in range(len(pixelhop_units)):
                pu = copy.deepcopy(pixelhop_units[j])
                if not load_pu:
                    pu.fit(resized)
                t = pu.transform(resized)
                transformed[j].update({i: t[0]})
                tmp.append(pu)
            new_pixelhop_units.append(tmp)

    if not load_pu:
        for i in range(len(new_pixelhop_units)):
            for j in range(len(pixelhop_units)):
                new_pixelhop_units[i][j].save(
                    save_path
                    + f"PU_{i}_{new_pixelhop_units[i][j].hop_args[0]['window']}"
                )
        pixelhop_units = None
    gc.collect()
    return transformed


def apply_pixelhops_cascaded(
    X: np.array, pixelhop_units: List[PixelHop], load_pu: bool, save_path: str
) -> List[Dict[int, np.array]]:
    """
    Applies the Pixelhops in cascaded manner, i.e. no resizing done, but rather pooling operations

    :param X: the input data
    :param y: the target values Xs
    :param pixelhop_units: the Pixelhop units to apply to the input data
    :param load_pu: whether the Pixelhop units were loaded or not, controls whether we save the fitted PUs
    :param save_path: the path to save the PU units to if required
    :return transformed: the data transformed by the PU units
    """
    transformed = []
    for i in range(len(pixelhop_units)):
        if not load_pu:
            pixelhop_units[i].fit(X)
        transformed.append(pixelhop_units[i].transform(X))

    if not load_pu:
        for i in range(len(pixelhop_units)):
            pixelhop_units[i].save(
                save_path + f"PU_{pixelhop_units[i].hop_args[0]['window']}"
            )
        pixelhop_units = None
    gc.collect()
    return transformed


def apply_rfts(
    Xs: List[Dict[int, np.array]],
    y: np.array,
    train_index: List[int],
    val_index: List[int],
    rft_units: List[RFT],
    block_sizes: List[int],
    n_bins: List[int],
    n_selected: List[int],
    training_depth: int,
    load_rft: bool,
    save_path: str,
    make_plots: bool,
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
    :param load_rft: whether the RFTs were loaded from previously or not
    :param save_path: the path to save the RFTs if they weren't loaded
    :return transformed: output of fitting and transforming based on the rft arguments given
    """
    transformed = []
    new_rft_units = []
    print(f"{'='*50} Fitting RFTs")
    for i in range(len(Xs)):  # range over each pixelhop
        tmp = {}
        for k in Xs[
            i
        ].keys():  # range over the raw features of the given pixelhop, each k is a depth
            if training_depth < 0 or (training_depth >= 0 and training_depth == k):
                print(f"Fitting RFT for PU unit {i} and depth {k}")
                feature = Xs[i][k]
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

                rft = copy.deepcopy(rft_units[k])
                if not load_rft:
                    rft.fit(X_t, y_t, X_v, y_v, n_bins[k], remove_outliers=True)
                tmp[k] = rft.transform(
                    feature.reshape(-1, feature.shape[-1]), n_selected[k]
                )
                new_rft_units.append((rft, k, i))
        transformed.append(tmp)

    if make_plots:
        generate_rft_plots(new_rft_units, save_path)

    if not load_rft:
        for i in range(len(new_rft_units)):
            rft, depth, pu_unit = new_rft_units[i]
            with open(save_path + f"RFT_{depth}_{pu_unit}", "wb") as f:
                pickle.dump(rft, f)
                f.close()
        new_rft_units = None
    print(f"{'='*50} Done fitting RFTs")
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


# TODO: add the LNT module
def apply_lnts(
    Xs: List[Dict[int, np.array]],
    y: np.array,
    lnt: LNT,
    rfts: List[RFT],
    train_index: List[int],
    val_index: List[int],
    n_rounds: int,
    n_selected: List[int],
    n_bins: List[int],
    block_sizes: List[int],
    training_depth: int,
    load_lnt: bool,
    save_path: str,
    make_plots: bool,
):
    """
    This module applies LNTs, can also run the compound LNT procedure if the relevant arguments are given

    :param Xs: the input features, the length of the list is the number of PU units and the keys in the dict are the depth of the features
    :param y: the target values
    :param lnts: the LNT to apply
    :param rfts: the RFTs to use between successive LNTs
    :param train_index: the indices to use for training
    :param val_index: the indices to use for validation
    :param n_rounds: how many times to apply the LNT consequently, the number of rounds
    :param n_selected: the number of selected features that we want to keep between successive lnts/rfts
    :param n_bins: the number of bins to use for the RFT procedures
    :param block_sizes: the size of the dimension at a given depth
    :param training_depth: the specific depth to train on if given
    :param load_lnt: whether the LNTs were loaded or not
    :param save_path: where to save the LNTs
    :param make_plots: whether to plot the compound rft procedures between the LNTs
    :return transformed: the transformed features for a given depth
    """
    transformed = []
    fitted_lnts = []
    fitted_rfts = []
    for i in range(len(Xs)):
        tmp = {}
        for k in Xs[i].keys():
            if training_depth < 0 or (training_depth >= 0 and training_depth == k):
                feature = Xs[i][k]
                y_reshaped = make_blocks(y, block_sizes[k], calculate_mean=True)
                N, H, W = y_reshaped.shape
                y_train = y_reshaped[train_index].flatten()
                y_val = y_reshaped[val_index].flatten()
                y_reshaped = y_reshaped.flatten()
                feature_cp = copy.deepcopy(feature)

                print(
                    f"{'='*50} Fitting LNTs for PU {i} depth {k} (n_rounds: {n_rounds})"
                )
                for j in range(n_rounds):  # compound lnt means n_rounds > 0
                    print(f"Round {j} input feature shape {feature_cp.shape}")
                    lnt_cp = copy.deepcopy(lnt)
                    if not load_lnt:
                        lnt_cp.fit(feature_cp, y_reshaped)

                    fitted_lnts.append((lnt_cp, k, i, j))
                    lnt_out = lnt_cp.transform(feature_cp)

                    # we have to reshape the transformed back to the original shape to provide the correct train/val indices
                    _, C = lnt_out.shape
                    lnt_out = lnt_out.reshape(N, H, W, -1)
                    rft = copy.deepcopy(rfts[j])
                    lnt_out_t = lnt_out[train_index].reshape(-1, C)
                    lnt_out_v = lnt_out[val_index].reshape(-1, C)
                    lnt_out = lnt_out.reshape(-1, C)

                    rft.fit(
                        lnt_out_t,
                        y_train,
                        lnt_out_v,
                        y_val,
                        n_bins[j],
                        remove_outliers=True,
                    )
                    fitted_rfts.append((rft, k, i, j))
                    rft_out = rft.transform(lnt_out, n_selected[j])
                    feature_cp = np.concatenate([feature, rft_out], axis=-1)

                tmp[k] = rft_out
                print(f"{'='*50} Done fitting LNTs")
            
            transformed.append(tmp)

                # TODO: figure out how we would load this from already trained rfts
    if make_plots:
        generate_rft_plots(fitted_rfts, save_path)

    # TODO: save these RFTS
    if not load_lnt:
        for i in range(len(fitted_lnts)):
            lnt, depth, pu_unit, round = fitted_lnts[i]
            with open(save_path + f"LNT_{depth}_{pu_unit}_{round}", "wb") as f:
                pickle.dump(lnt, f)
                f.close()

        fitted_lnts = None

    gc.collect()
    return transformed


def train_xgboosts(
    Xs: List[np.array],
    y: np.array,
    xgb_args: Dict,
    train_index: List[int],
    val_index: List[int],
    training_depth: int,
    max_depth: int,
    block_sizes: List[int],
    prev_xgbs: List[SingleXGBoost],
    save_xgb: bool,
    save_path: str,
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
    :param prev_xgbs: the previous xgbs, only used when we are training on residuals
    :param save_xgb: whether to save the new trained xgb or not
    :param save_path: where to save the xgb
    :return
    """
    sxgb = SingleXGBoost(**xgb_args)

    # we are training on the lowest depth
    if training_depth == max_depth - 1:
        y_reshaped = make_blocks(y, block_sizes[training_depth], calculate_mean=True)
        N, H, W = y_reshaped.shape
        y_train = y_reshaped[train_index].flatten()
        y_val = y_reshaped[val_index].flatten()
        feature = Xs[training_depth]
        _, C = feature.shape
        feature = feature.reshape(N, H, W, -1)
        X_train = feature[train_index].reshape(-1, C)
        X_val = feature[val_index].reshape(-1, C)
    else:  # TODO: build in the functionality to train on residuals
        X_train = None
        y_train = None
        X_val = None
        y_val = None
        print("Not implemented yet")

    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)
    sxgb.fit(train, val)

    if make_plots:
        sxgb.plot_learning_curve(
            eval_metric="rmse", path=save_path + f"lc_{training_depth}"
        )
    
    if save_xgb:
        with open(save_path + f"xgb_{training_depth}", "wb") as f: 
            pickle.dump(sxgb, f)
            f.close()

    return sxgb

def model(
    X: np.array,
    y: np.array,
    train_index: List[int],
    val_index: List[int],
    model_args: dict,
    training_depth: int,
    save_path: str,
    load_pu: bool,
    load_rft: bool,
    load_xgb: bool,
    load_path: str,
    make_plots: bool = False,
):
    """Run the GDH model"""

    # FEATURE EXTRACTION
    pixelhop_units = []

    if load_pu:
        pu_files = [f for f in os.listdir(load_path) if "PU" in f]
        for pu in pu_files:
            with open(load_path + "/" + pu, "rb") as f:
                p = pickle.load(f)
                f.close()
            pixelhop_units.append(p)
    else:
        pixel_args = model_args.get("pixelhops")
        for unit in pixel_args.keys():
            pixelhop_units.append(PixelHop(**pixel_args[unit]))

    # cascaded or not
    cascaded = model_args.get("cascaded")
    model_depth = model_args.get("model_depth")
    block_sizes = model_args.get("block_sizes")
    if cascaded:
        transformed = apply_pixelhops_cascaded(X, pixelhop_units, load_pu, save_path)
    else:
        transformed = apply_pixelhops_noncascaded(
            X,
            model_depth,
            block_sizes,
            pixelhop_units,
            training_depth,
            load_pu,
            save_path,
        )

    feat_concat = model_args.get("feat_concat")
    if feat_concat:
        for i in range(len(transformed)):
            # default args concat 1-hop neighbors for feature_concat
            transformed[i] = feature_concat(transformed[i], training_depth)

    # SUPERVISED FEATURE REDUCTION
    rft_units = []
    # TODO: put in functionality for RFT init arguments
    if load_rft:
        rft_files = [f for f in os.listdir(load_path) if "RFT" in f]
        for rft in rft_files:
            with open(load_path + "/" + rft, "rb") as f:
                r = pickle.load(f)
                f.close()
            rft_units.append(r)
    else:
        rfts = model_args.get("rfts")
        for unit in rfts.keys():
            rft_units.append(RFT())

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
        load_rft,
        save_path,
        make_plots,
    )

    lnt_procedure = model_args.get("lnts")
    lnt_args = lnt_procedure.get("lnt_args")
    n_rounds = lnt_procedure.get("n_rounds")
    lnt = LNT(**lnt_args)
    rfts = lnt_procedure.get("rfts")
    n_selected = [rfts[k].get("n_selected") for k in rfts.keys()]
    n_bins = [rfts[k].get("n_bins") for k in rfts.keys()]
    rft_units = [RFT() for _ in rfts.keys()]

    transformed = apply_lnts(
        transformed,
        y,
        lnt,
        rft_units,
        train_index,
        val_index,
        n_rounds,
        n_selected,
        n_bins,
        block_sizes,
        training_depth,
        False,  # TODO: figure out how we'd load this
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
    prev_xgbs = []
    sxgb = train_xgboosts(
        features, 
        y, 
        xgboost_args, 
        train_index, 
        val_index, 
        training_depth, 
        max_depth,
        block_sizes, 
        prev_xgbs, 
        True, 
        save_path
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
    load_xgb = cmd_args["load_xgb"]
    seed = cmd_args["seed"]
    val_size = cmd_args["val_size"]

    set_seed(seed)

    model_args = get_model_args(MODEL_ARCHS, model_arch)

    for arg in cmd_args.keys():
        model_args[arg] = cmd_args[arg]

    x = os.mkdir(model_path)
    with open(f"{model_path}model_args", "wb") as f:
        pickle.dump(model_args, f)
        f.close()

    X_data, y_data = process_data(color_space, channel, n_samples)
    N = X_data.shape[0]
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
        load_xgb,
        load_path,
        make_plots,
    )
