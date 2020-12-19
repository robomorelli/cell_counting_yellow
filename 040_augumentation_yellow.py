import glob
import sys
import numpy as np
import imageio
import cv2
import random
from skimage import transform
import os
from tqdm import tqdm
from subprocess import check_output
import matplotlib.pyplot as plt
from importlib import import_module
from skimage import io
import skimage.io
from skimage.transform import rotate
from skimage.transform import resize
from subprocess import check_output
import albumentations as alb
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology import watershed, remove_small_holes, remove_small_objects, label, erosion
from skimage.feature import peak_local_max
import argparse
import logging
import shutil

from albumentations import (RandomCrop,CenterCrop,ElasticTransform,RGBShift,Rotate,
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,Transpose,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip, HorizontalFlip,

    HueSaturationValue,
)

from augmentation_utils import *
from config import *
from utils import *

image_ids = os.listdir(CropImages)
split_num = 5
split_num_new_images = 11
shift = len(image_ids)
id_edges = [492, 969, 1116, 2001, 2322, 2325, 2326, 2327, 2328, 2330, 2333, 2336]

if __name__ == "__main__":

    with open('id_new_images.pickle', 'rb') as handle:
        dic = pickle.load(handle)
    id_new_image = dic['id_new_images']

    print(id_new_image)

    parser = argparse.ArgumentParser(description='Define parameters for crop.')
    parser.add_argument('--start_from_zero', nargs="?", default = False, help='remove previous file in the destination folder')

    args = parser.parse_args()

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        shutil.rmtree(AugCropImages)
        os.makedirs(AugCropImages)

        shutil.rmtree(AugCropMasks)
        os.makedirs(AugCropMasks)
        print('starting augmentation')

    src_files = os.listdir(CropImages)

    print('copying images')
    for file_name in src_files:
        full_file_name = os.path.join(CropImages, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, AugCropImages)

    print('copying masks')
    src_files = os.listdir(CropWeightedMasks)
    for file_name in src_files:
        full_file_name = os.path.join(CropWeightedMasks, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, AugCropMasks)


    make_data_augmentation(image_ids, CropImages,  CropWeightedMasks, split_num, id_new_images,
                           split_num_new_images, id_edges, AugCropImages, AugCropMasks,  ix = shift)
