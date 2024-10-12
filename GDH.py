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
    parser.add_argument("--load_pu", default = False, type = bool)
    parser.add_argument("--load_features", default = False, type = bool)
    return vars(parser.parse_args())

# prev_bias will be passed by the PixelHop unit explicitly
saab_args = [
    {"num_kernels": None, "thresh": None, "use_bias": False},
    {"num_kernels": None, "thresh": None, "use_bias": True},
    {"num_kernels": None, "thresh": None, "use_bias": True},
    {"num_kernels": None, "thresh": None, "use_bias": True},
]

# hop arguments for each layer, essentially dictates our padding and window sizes
hop_args_1 = [
    {"pad": 3, "window": 7, "stride": 1, "method": "reflect"},
    {"pad": 3, "window": 7, "stride": 1, "method": "reflect"},
    {"pad": 3, "window": 7, "stride": 1, "method": "reflect"},
    {"pad": 3, "window": 7, "stride": 1, "method": "reflect"},
]

hop_args_2 = [
    {"pad": 2, "window": 5, "stride": 1, "method": "reflect"},
    {"pad": 2, "window": 5, "stride": 1, "method": "reflect"},
    {"pad": 2, "window": 5, "stride": 1, "method": "reflect"},
    {"pad": 2, "window": 5, "stride": 1, "method": "reflect"},
]

hop_args_3 = [
    {"pad": 1, "window": 3, "stride": 1, "method": "reflect"},
    {"pad": 1, "window": 3, "stride": 1, "method": "reflect"},
    {"pad": 1, "window": 3, "stride": 1, "method": "reflect"},
    {"pad": 1, "window": 3, "stride": 1, "method": "reflect"},
]

feature_manager_args = {
    "block_sizes": [1, 2, 4, 8],
    "n_bins": [64, 64, 64, 64],
    "level_sizes": [128, 64, 32, 16],
}

# reduce args, first layer of c/w we never reduce
reduce_args = [
    {"pool": 2, "method": np.max, "use_abs": True},
    {"pool": 2, "method": np.max, "use_abs": True},
    {"pool": 2, "method": np.max, "use_abs": True},
]

depth = [4, 4, 4]

n_pixel_hops = 3

# training_depth starts at 0
training_depth = 3

# numbers of features to select from the RFT procedure
n_selected_args = {0: [20, 20, 20], 1: [20, 20, 20], 2: [20, 20, 20], 3: [20, 20, 20]}

xgboost_args = {
    "params": {
        "tree_method": "hist",
        "gpu_id": None,
        "objective": "reg:squarederror",
        "max_depth": 4,
        "learning_rate": 0.5,
        "eval_metric": "rmse",
    },
    "num_boost_round": 2000,
    "early_stopping_rounds": 200,
}

val_size = 0.2

model_args = {
    "hop_args_0": hop_args_1,
    "hop_args_1": hop_args_2,
    "hop_args_2": hop_args_3,
    "saab_args_0": saab_args,
    "saab_args_1": saab_args,
    "saab_args_2": saab_args,
    "reduce_args_0": reduce_args,
    "reduce_args_1": reduce_args,
    "reduce_args_2": reduce_args,
    "feature_manager_args": feature_manager_args,
    "depth": depth,
    "training_depth": training_depth,
    "n_selected_args": n_selected_args,
    "xgboost_args": xgboost_args,
    "val_size": val_size,
}


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
    load_feature: bool = False
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
        
    if not load_feature:
        feature_manager = FeatureManager(
            features_list, **model_args.get("feature_manager_args")
        )

        # fit and transform with all of the RFTs
        training_depth = model_args.get("training_depth")
        num_selected_features = model_args.get("n_selected_args")
        for i in range(training_depth, -1, -1):
            feature_manager.fit_rft(targets=y, depth=i, calculate_mean=True)
            feature_manager.transform_rfts(n_selected=num_selected_features[i], depth=i)
    else:
        with open(load_path + "feature_manager", "rb") as f:
            feature_manager = pickle.load(f)
            f.close()

    if not load_feature:
        feature_manager.save(save_path + f"feature_manager")

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

    # m = model(X_data, y_data, n_pixel_hops, model_args)
    # with open(f"{model_path}{model_name}", "wb") as f:
    #     pickle.dump(m, f)
    #     f.close()

    # code for some tests
    model(X_data, y_data, n_pixel_hops, training_depth, model_args, model_path, load_path, load_pu, load_features)
