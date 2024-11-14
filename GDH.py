# Laurence Palmer
# palmerla@usc.edu
# 2024.09

from helpers.helpers import *
from modules.pixelhop import PixelHop
from modules.featuremanager import FeatureManager
from modules_lab.pixelhop import Pixelhop as Pixelhop_lab
from modules.single_xgboost import SingleXGBoost
from argparse import ArgumentParser
from typing import *
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import gc
import pdb
import datetime


DATADIR_PATH = "data/128x128_patches_resize_400x400_RGB/"
HAZYDIR_NAME = "hazy"
CLEARDIR_NAME = "gt"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--which", default="", type=str)
    parser.add_argument("--n_samples", default=50, type=int)
    parser.add_argument("--n_iterations", default=1, type=int)
    parser.add_argument("--channel", default="Y", type=str)
    parser.add_argument("--model_name", default="model", type=str)
    parser.add_argument(
        "--model_path", default=f"models/{datetime.datetime.now()}/", type=str
    )
    parser.add_argument("--color_space", default="YUV", type=str)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--load_pu", default=False, type=bool)
    parser.add_argument("--load_features", default=False, type=bool)
    parser.add_argument("--make_rft_plots", default=False, type=bool)
    return vars(parser.parse_args())


# @profile
# @timer
def model(
    X: np.array,
    y: np.array,
    n_pixel_hops: int,
    training_depth: int,
    model_args: dict,
    save_path: str,
    load_path: str,
    load_pu: bool = False,
    load_feature: bool = False,
    make_rft_plots: bool = False,
):
    """Run the GDH model"""
    # pdb.set_trace()
    pixelhop_units = []
    names = []
    for i in range(n_pixel_hops):
        if not load_pu:
            PU = PixelHop(
                depth=model_args.get("depth")[i],
                saab_args=model_args.get(f"saab_args_{i}"),
                hop_args=model_args.get(f"hop_args_{i}"),
                reduce_args=model_args.get(f"reduce_args_{i}"),
            )
        else:
            with open(load_path + f"PU_{i}", "rb") as f:
                PU = pickle.load(f)

        pixelhop_units.append(PU)

    # fit the pixelhop and c/w Saab transforms
    if not load_pu:
        for i in range(n_pixel_hops):
            pixelhop_units[i].fit(X)

    # transform the inputs to get the features
    features_list = []
    for i in range(n_pixel_hops):
        feature = pixelhop_units[i].transform(X)
        feature = feature_concat(feature)

        # concatenate 1 hop features pixel by pixel

        features_list.append(feature)

        if not load_pu:
            pixelhop_units[i].save(save_path + f"PU_{i}")

    num_selected_features = model_args.get("n_selected_args")
    if not load_feature:
        feature_manager = FeatureManager(
            features_list, **model_args.get("feature_manager_args")
        )

        # fit and transform with all of the RFTs
        for i in range(training_depth, -1, -1):
            feature_manager.fit_rft(targets=y, depth=i, calculate_mean=True)
            feature_manager.transform_rfts(n_selected=num_selected_features[i], depth=i)
            gc.collect()
    else:
        with open(load_path + "feature_manager", "rb") as f:
            feature_manager = pickle.load(f)
            f.close()
        feature_manager.training_features = features_list
        feature_manager.transformed_features = {}
        # set the training data
        for i in range(training_depth, -1, -1):
            feature_manager.transform_rfts(n_selected = num_selected_features[i], depth = i)
        

    if not load_feature:
        feature_manager.save(save_path + f"feature_manager")

    if make_rft_plots:
        name_map = {
            "0x0": "128x128 feature maps (7x7 kernel)",
            "0x1": "128x128 feature maps (5x5 kernel)",
            "0x2": "128x128 feature maps (3x3 kernel)",
            "1x0": "64x64 feature maps (7x7 kernel)",
            "1x1": "64x64 feature maps (5x5 kernel)",
            "1x2": "64x64 feature maps (3x3 kernel)",
            "2x0": "32x32 feature maps (7x7 kernel)",
            "2x1": "32x32 feature maps (5x5 kernel)",
            "2x2": "32x32 feature maps (3x3 kernel)",
            "3x0": "16x16 feature maps (7x7 kernel)",
            "3x1": "16x16 feature maps (5x5 kernel)",
            "3x2": "16x16 feature maps (3x3 kernel)"
        }
        for i in range(training_depth + 1):
            for j in range(n_pixel_hops):
                # pdb.set_trace()
                feature_manager.plot_rft_train_val(i, j, save_path+name_map[f"{i}x{j}"], name_map[f"{i}x{j}"])

    # single round of RFT
    # feature_manager.train_xgboost(
    #     targets=y,
    #     depth=training_depth,
    #     val_size=model_args.get("val_size"),
    #     xgb_args=model_args.get("xgboost_args"),
    # )

    # for i in range(training_depth - 1, -1, -1):
    #     feature_manager.train_xgboost(
    #         targets = y,
    #         depth = i,
    #         val_size = model_args.get("val_size"),
    #         xgb_args = model_args.get("xgboost_args")
    #     )

    # for i in feature_manager.xgboosts.keys():
    #     feature_manager.xgboosts[i].plot_learning_curve(eval_metric = "rmse", path = save_path + f"lc_{i}")

    # with open(save_path + "feature_manager_final", "wb") as f:
    #     pickle.dump(feature_manager, f)
    #     f.close()
    return None


if __name__ == "__main__":
    set_seed(10)

    cmd_args = parse_arguments()
    which = cmd_args["which"]
    n_samples = cmd_args["n_samples"]
    n_iterations = cmd_args["n_iterations"]
    channel = cmd_args["channel"]
    model_name = cmd_args["model_name"]
    model_path = cmd_args["model_path"]
    color_space = cmd_args["color_space"]
    load_path = cmd_args["load_path"]
    load_pu = cmd_args["load_pu"]
    load_features = cmd_args["load_features"]
    make_rft_plots = cmd_args["make_rft_plots"]

    if color_space == "YUV":
        colorspace = cv2.COLOR_BGR2YUV
    elif color_space == "RGB":
        colorspace = cv2.COLOR_BGR2RGB

    data_args = {
        # "resize": {"width": 400, "height": 400, "method": cv2.INTER_AREA},
        "color_transform": {"colorspace": colorspace},
        # "patches": {"patch_size": 128}
    }

    x = os.mkdir(model_path)
    with open(f"{model_path}{model_name}_args", "wb") as f:
        pickle.dump(model_args, f)
        f.close()

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

    model(
        X_data,
        y_data,
        n_pixel_hops,
        training_depth,
        model_args,
        model_path,
        load_path,
        load_pu,
        load_features,
        make_rft_plots,
    )
