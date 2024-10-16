# Laurence Palmer
# palmerla@usc.edu
# 2024.09
import cv2
import numpy as np
import random
import os
import time
import pdb
from typing import *
from skimage.util import view_as_windows
from skimage.measure import block_reduce


class HazeDataset:
    """
    Class to load/manipulate hazy image datasets.

    :param dataset_folder: folder where the hazy and clear datasets are stored
    :param clear_dataset: name of the clear dataset
    :param hazy_dataset: name of the hazy dataset
    """

    def __init__(self, dataset_folder: str, clear_dataset: str, hazy_dataset: str):

        assert clear_dataset in os.listdir(
            dataset_folder
        ), "Clear dataset not in the folder"
        assert hazy_dataset in os.listdir(
            dataset_folder
        ), "Hazy dataset not in the folder"

        dataset_folder = (
            dataset_folder + "/" if dataset_folder[-1] != "/" else dataset_folder
        )

        self.dataset_folder = dataset_folder
        self.clear_dataset_name = clear_dataset
        self.hazy_dataset_name = hazy_dataset

    def resize(
        self,
        img: np.array,
        width: int,
        height: int,
        method=cv2.INTER_AREA,
    ) -> Tuple[np.array, np.array]:
        """
        Resizes a single image

        :param imgs: the image
        :param dim1: first dimension of resize i.e. width
        :param dim2: second dimension of resize second dimension of resize i.e. height
        :param method: the method to resize the image
        :return resized: resized image
        """
        resized = cv2.resize(img, (width, height), method)
        return resized

    def pad(
        self,
        img: np.array,
        num_pad: int = 1,
        pad_method: str = "reflect",
        constant: Optional[int] = None,
    ) -> np.array:
        """
        Pads a single image

        :param img: a single image
        :param num_pad: how many in the padding
        :param pad_method: the padding method
        """
        if pad_method == "constant":
            padded = np.pad(
                img, ((num_pad, num_pad), (num_pad, num_pad), (0,0)), mode=pad_method, constant_values=constant
            )
        else:
            padded = np.pad(img, ((num_pad, num_pad), (num_pad, num_pad), (0,0)), mode=pad_method)
        return padded

    def patch(
        self, imgs: Tuple[np.array, np.array], center_h: int = None, center_w: int = None, patch_size: int = 64
    ) -> Tuple[np.array, np.array]:
        """
        Returns a patch of the image pair, same patch in the hazy vs non-hazy, decides center based on 
        patch size if center not provided

        :param imgs: (clear, hazy) the image pair, assuming dims order of (height, width, channels), should be same shape
        :param center_h: center pixel to capture from for height
        :param center_w: center pixel to capture from for the width
        :param patch_size: size of the patch, will always be square sized patches.
        """
        if not (center_h or center_w):
            center_h = random.randint(patch_size//2, imgs[0].shape[1] - patch_size//2)
            center_w = random.randint(patch_size//2, imgs[0].shape[1] - patch_size//2)

        left_patch = patch_size // 2
        right_patch = patch_size // 2
        clear_patch = imgs[0][
            center_h - left_patch : center_h + right_patch,
            center_w - left_patch : center_w + right_patch,
            :,
        ]
        hazy_patch = imgs[1][
            center_h - left_patch : center_h + right_patch,
            center_w - left_patch : center_w + right_patch,
            :,
        ]
        return clear_patch, hazy_patch

    def color_transform(self, img: np.array, colorspace) -> np.array:
        """
        Transform color space
        """
        return cv2.cvtColor(img, colorspace)

    def import_images(self) -> List[Tuple[np.array, np.array]]:
        """
        Imports the images and returns them in pairs

        :return images: (clear, hazy) pairs
        """
        full_clear_path = self.dataset_folder + self.clear_dataset_name
        full_haze_path = self.dataset_folder + self.hazy_dataset_name

        all_clear = os.listdir(full_clear_path)
        all_haze = os.listdir(full_haze_path)
        
        # sort them so we can pair them correctly
        all_clear.sort()
        all_haze.sort()

        haze_index = 0
        data = []
        for clear in all_clear:
            clear_prefix = clear.split(".")[0]

            # keep pairing the hazy images with the clear
            while haze_index < len(all_haze) and clear_prefix in all_haze[haze_index]:
                haze = all_haze[haze_index]
                clear_path = f"{full_clear_path}/{clear}"
                haze_path = f"{full_haze_path}/{haze}"
                clear_image = cv2.imread(clear_path)
                hazy_image = cv2.imread(haze_path)
                data.append((clear_image, hazy_image))
                haze_index += 1

        return data
    
    def import_images_custom(self) -> List[Tuple[np.array, np.array]]:
        """Helper to import images that have already been modified"""
        full_clear_path = self.dataset_folder + self.clear_dataset_name
        full_haze_path = self.dataset_folder + self.hazy_dataset_name

        all_clear = os.listdir(full_clear_path)
        all_haze = os.listdir(full_haze_path)
        data = []
        for clear, hazy in zip(all_clear, all_haze):
            clear_path = f"{full_clear_path}/{clear}"
            haze_path = f"{full_haze_path}/{hazy}"
            clear_image = cv2.imread(clear_path)
            hazy_image = cv2.imread(haze_path)
            data.append((clear_image, hazy_image))
        
        return data

    def train_test_split(
        self, data: List[Tuple[np.array, np.array]], n_train: int
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Splits into data into training and testing

        :param data: image pairs
        :param n_train: number to put in the training sample
        :return train, test: training and testing splits
        """
        shuffled = random.shuffle(data)
        train = shuffled[:n_train]
        test = shuffled[n_train:]

        return train, test
    
def timer(f: Callable):
    def timer(*arg, **kwargs):
        t0 = time.time()
        result = f(*arg, **kwargs)
        t1 = time.time()
        print(f"Function {f} took: {t1 - t0}s")
        return result

    return timer

def get_data(args: Dict[str, dict], datadir_path: str, cleardir_name: str, hazydir_name: str) -> Tuple[np.array, np.array]:
    """Loads data, use args to pass kwargs depending on how to augment"""
    ds = HazeDataset(datadir_path, cleardir_name, hazydir_name)
    data = ds.import_images_custom()

    data_augmented = {"clear": [], "hazy": []}
    
    # resize and pad if needed
    for clear, hazy in data:

        # color transform
        if args.get("color_transform"):
            clear = ds.color_transform(clear, **args["color_transform"])
            hazy = ds.color_transform(hazy, **args["color_transform"])

        # resizing
        if args.get("resize"):
            clear = ds.resize(clear, **args["resize"])
            hazy = ds.resize(hazy, **args["resize"])

        # patches 
        if args.get("patches"):
            clear, hazy = ds.patch((clear, hazy), **args["patches"])

        # only pad the hazy images
        if args.get("pad"):
            hazy = ds.pad(hazy, **args["pad"])

        data_augmented["clear"].append(clear)
        data_augmented["hazy"].append(hazy)

    X = np.array(data_augmented["hazy"])
    y = np.array(data_augmented["clear"])
    return X, y

def set_seed(seed: int):
    """Sets the random seed for the experiment"""
    np.random.seed(seed)
    random.seed(seed)

def split_channels(X: np.array, y: np.array, c: int = 0) -> np.array:
    """Splits the channels of the input image (N, h, w, c)"""
    X_c = np.expand_dims(X[:, :, :, c], axis = 3)
    y_c = np.expand_dims(y[:, :, :, c], axis = 3)
    return X_c, y_c

def feature_concat(
        Xs: dict, 
        pad: int = 1,
        stride: int = 1,
        window: int = 3,
        method: str = "reflect"
    ) -> dict:
    """Feature concatenation"""
    concated = {}
    for i in Xs.keys():
        feature = Xs[i]
        print(f"Concatenating feature with dimension {feature.shape}")
        N, h, w, c = feature.shape
        feature = np.pad(feature, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode=method)
        feature = view_as_windows(feature, (1, window, window, c), (1, stride, stride, c))
        feature = feature.reshape(N, h, w, -1)
        print(f"Resulting feature shape {feature.shape}")
        concated[i] = feature
    return concated
