# Laurence Palmer
# palmerla@usc.edu
# 2024.09
import cv2
import numpy as np
import random
import os
import time
import pdb
import torch
import pickle
from typing import *
from skimage.util import view_as_windows
from sklearn.preprocessing import MinMaxScaler
import gc

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

    @staticmethod
    def resize(
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

    @staticmethod
    def pad(
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

    @staticmethod
    def patch(
        imgs: Tuple[np.array, np.array], center_h: int = None, center_w: int = None, patch_size: int = 64
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

    @staticmethod
    def color_transform(img: np.array, colorspace) -> np.array:
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
    
    def import_images_custom(self, batch: bool = False) -> List[Tuple[np.array, np.array]]:
        """Helper to import images that have already been modified"""
        full_clear_path = self.dataset_folder + self.clear_dataset_name
        full_haze_path = self.dataset_folder + self.hazy_dataset_name

        all_clear = os.listdir(full_clear_path)
        all_haze = os.listdir(full_haze_path)
        all_clear.sort()
        all_haze.sort()
        pairs = [(i, j) for i, j in zip(all_clear, all_haze)]
        data = []
        for clear, hazy in pairs:
            clear_path = f"{full_clear_path}/{clear}"
            haze_path = f"{full_haze_path}/{hazy}"

            if batch:
                data.append((clear_path, haze_path))
            else:
                data.append((cv2.imread(clear_path), cv2.imread(haze_path)))
        
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
        clear = augment_image(clear, args)
        hazy = augment_image(hazy, args)
        
        data_augmented["clear"].append(clear)
        data_augmented["hazy"].append(hazy)

    X = np.array(data_augmented["hazy"])
    y = np.array(data_augmented["clear"])
    return X, y

def get_data_batched(datadir_path: str, cleardir_name: str, hazydir_name: str) -> Tuple[List[str], List[str]]:
    """Returns the data but just the path to find them"""
    ds = HazeDataset(datadir_path, cleardir_name, hazydir_name)
    data = ds.import_images_custom(batch = True)
    clear = []
    hazy = []

    for clear_path, hazy_path in data:
        clear.append(clear_path)
        hazy.append(hazy_path)
    return hazy, clear

def get_batch(X: List[str], to_get: np.array, args: Dict[str, dict]) -> Tuple[np.array, np.array]:
    """Gets the images in a given batch"""
    data = []
    for ind in to_get: 
        img = cv2.imread(X[ind])
        img = augment_image(img, args)
        data.append(img)

    data = np.array(data)
    
    channel = args.get("channel")

    if channel == "Y" or channel == "R":
        data = split_channels(data, 0)
    elif channel == "U" or channel == "G":
        data = split_channels(data, 1)
    elif channel == "V" or channel == "B":
        data = split_channels(data, 2)
    
    data = data.squeeze(-1)
    gc.collect()
    return data
    
def augment_image(img: np.array, args: Dict[str, str]) -> Tuple[np.array, np.array]:
    """Augment a single image"""
    # color transform
    if args.get("color_transform"):
        img = HazeDataset.color_transform(img, args["color_transform"])

    # resizing
    if args.get("resize"):
        img = HazeDataset.resize(img, **args["resize"])

    # patches TODO: Make the pair an optional argument
    if args.get("patches"):
        img = HazeDataset.patch((img, img), **args["patches"])

    # only pad the hazy images
    if args.get("pad"):
        img = HazeDataset.pad(img, **args["pad"])

    return img

def set_seed(seed: int):
    """Sets the random seed for the experiment"""
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 


def split_channels(X: np.array, c: int = 0) -> np.array:
    """Splits the channels of the input image (N, h, w, c)"""
    X_c = np.expand_dims(X[:, :, :, c], axis = 3)
    return X_c

def feature_concat(
        Xs: Dict[int, np.array], 
        training_depth: int, 
        pad: int = 1,
        stride: int = 1,
        window: int = 3,
        method: str = "reflect",
    ) -> dict:
    """
    Feature concatenation, decides how many neighboring pixels features to concatentate to the spectral dimension of a pixel.
    Default params concatenates features for every 1-hop neighbor for every pixel
    
    :param Xs: the input data 
    :param training_depth: the depth of the data to concatenate for, -1 if we want to train all
    :param pad: how much padding to use
    :param stride: step size for the window
    :param window: size of the window, window x window = # of neighbors for feature concatentation 
    :param method: what method to use for the padding
    :return concated: all of the concatenated features
    """
    concated = {}
    for i in Xs.keys():
        if training_depth < 0 or (training_depth >= 0 and training_depth == i):
            feature = Xs[i]
            print(f"Concatenating feature with dimension {feature.shape}")
            N, h, w, c = feature.shape
            feature = np.pad(feature, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode=method)
            feature = view_as_windows(feature, (1, window, window, c), (1, stride, stride, c))
            feature = feature.reshape(N, h, w, -1)
            print(f"Resulting feature shape {feature.shape}")

            concated[i] = feature

    return concated

def feature_concat_one_step(
    X: np.array, 
    pad: int = 1,
    stride: int = 1, 
    window: int = 3, 
    method: str = "reflect"
) -> dict:
    """
    Feature concatenation, but one step rather than for the full set

    :param X: the input data
    :param pad: how much padding to use 
    :param stride: step size for the window 
    :param window: size of the window
    :param method: what method to use for the padding
    """
    N, h, w, c = X.shape
    print(f"Concatenating feature with dimension {X.shape}")
    N, h, w, c = X.shape
    X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode=method)
    X = view_as_windows(X, (1, window, window, c), (1, stride, stride, c))
    X = X.reshape(N, h, w, -1)
    print(f"Resulting feature shape {X.shape}")
    return X

def make_blocks(
        X: np.array, block: int, calculate_mean: bool = False, scale: bool = False
    ) -> np.array:
        """
        Makes blocks of the specified block sizes, this helps with creating the
        "coarse" representations of our target values to supervise the levels of the
        GUSL pipeline. Ensure that H//block and Y//block == the shape of the features
        at the relevant level of the RFT.

        :param X: the input array, should be size (N, H, W)
        :param block: the block size to calculate
        :param calculate_mean: whether to compress down with mean
        :param scale
        :return blocks: the resulting blocks
        """

        N, H, W = X.shape
        blocks = X.reshape(-1, H // block, block, W // block, block)
        blocks = blocks.transpose((0, 1, 3, 2, 4))

        if scale: # TODO: when block sizes are 1 then we have to take it across the entire image
            # need floats now if we scale
            blocks = blocks.astype("float32")
            scaler = MinMaxScaler()
            for i in range(blocks.shape[0]): 
                for j in range(blocks.shape[1]):
                    for k in range(blocks.shape[2]):
                        blocks[i][j][k] = scaler.fit_transform(blocks[i][j][k].reshape(-1, 1)).reshape(block, block)

        if calculate_mean:
            blocks = np.mean(blocks, axis=(3, 4))

        return blocks

def size_up(X: np.array, size: Tuple[int, int], scheme: int = cv2.INTER_LANCZOS4) -> np.array:
    """Sizes up an image using cv2"""
    X_resize = []
    for i in range(len(X)):
        X_resize.append(
            cv2.resize(X[i], size, interpolation = scheme)
        )
    return np.array(X_resize)


def load_feature(path: str) -> np.array:
    with open(path, "rb") as f:
        r = pickle.load(f)
        f.close()
    return r