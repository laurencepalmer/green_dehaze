# Laurence Palmer
# palmerla@usc.edu
# 2024.09

from helpers.helpers import *
from modules.pixelhop import PixelHop
from modules.single_xgboost import SingleXGBoost
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
    parser.add_argument("--model_arch", default = "cascaded_pixelhop_depth_7", type = str)
    parser.add_argument("--n_samples", default=50, type=int)
    parser.add_argument("--channel", default="Y", type=str)
    parser.add_argument(
        "--model_path", default=f"models/{datetime.datetime.now()}/", type=str
    )
    parser.add_argument("--color_space", default="YUV", type=str)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--make_plots", default=False, type=bool)
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

def process_data(color_space: str, channel: str, n_samples: int) -> Tuple[np.array, np.array]:
    """Processes data based on args given"""

    if color_space == "YUV":
        colorspace = cv2.COLOR_BGR2YUV
    elif color_space == "RGB":
        colorspace = cv2.COLOR_BGR2RGB

    # other options as well, look at helpers
    data_args = {
        "color_transform": {"colorspace": colorspace},
    }
    X, y = get_data(data_args, DATADIR_PATH, CLEARDIR_NAME, HAZYDIR_NAME, n_samples)
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
    return X_data, y_data

def apply_pixelhops_noncascaded(X: np.array, model_depth: int, block_sizes: List[int], pixelhop_units: List[PixelHop], load_path: str, save_path: str) -> List[np.array]:
    """
    Applies the pixelhop units in a non-cascaded fashion by 
    resizing the input image. 

    :param X: the input data
    :param depth: 
    :param pixelhop_units: 
    :return 
    """
    transformed = []
    for i in range(model_depth):
        resized = make_blocks(X, block_sizes[i], calculate_mean = True)
        resized = np.expand_dims(resized, axis = -1)
        for j in range(len(pixelhop_units)):
            pixelhop_units[j].fit(resized)
            transformed.append(pixelhop_units[j].transform(resized))
        
    if not load_path: 
        for i in range(len(pixelhop_units)):
            pixelhop_units[i].save(save_path + f"PU_{i}")

    return transformed

def apply_pixelhops_cascaded(X: np.array, pixelhop_units: List[PixelHop], load_path: str, save_path: str) -> List[np.array]:
    transformed = []
    for i in range(len(pixelhop_units)):
        pixelhop_units[i].fit(X)
        transformed.append(pixelhop_units[i].transform(X))

    if not load_path: 
        for i in range(len(pixelhop_units)):
            pixelhop_units[i].save(save_path + f"PU_{i}")

    return transformed

def model(
    X: np.array,
    y: np.array,
    model_args: dict,
    save_path: str,
    load_path: str,
    make_rft_plots: bool = False,
):
    """Run the GDH model"""

    # feature extraction phase
    pdb.set_trace()
    pixelhop_units = []

    if not load_path: 
        pixel_args = model_args.get("pixelhops")
        for unit in pixel_args.keys():
            pixelhop_units.append(PixelHop(
                **pixel_args[unit]
            ))    
    else:
        pu_files = [f for f in os.listdir(load_path) if "PU" in f]
        for pu in pu_files:
            with open(load_path + "/" + pu, "rb") as f:
                p = pickle.load(f)
                f.close()
            pixelhop_units.append(p)

    # cascaded or not 
    cascaded = model_args.get("cascaded")
    model_depth = model_args.get("model_depth")
    block_sizes = model_args.get("block_sizes")
    if not cascaded: 
        transformed = apply_pixelhops_noncascaded(X, model_depth, block_sizes, pixelhop_units, load_path)
    else:
        transformed = apply_pixelhops_cascaded(X, pixelhop_units, load_path, save_path)
    
    feat_concat = model_args.get("feat_concat")
    if feat_concat: 
        for i in range(len(transformed)): 
            transformed[i] = feature_concat()
    

    
    return None


if __name__ == "__main__":
    set_seed(10)

    cmd_args = parse_arguments()
    model_arch = cmd_args["model_arch"]
    n_samples = cmd_args["n_samples"]
    channel = cmd_args["channel"]
    model_path = cmd_args["model_path"]
    color_space = cmd_args["color_space"]
    load_path = cmd_args["load_path"]
    make_plots = cmd_args["make_plots"]
    
    model_args = get_model_args(MODEL_ARCHS, model_arch)

    x = os.mkdir(model_path)
    with open(f"{model_path}model_args", "wb") as f:
        pickle.dump(model_args, f)
        f.close()

    X_data, y_data = process_data(color_space, channel, n_samples)

    model(
        X_data,
        y_data,
        model_args,
        model_path,
        load_path,
        make_plots,
    )
