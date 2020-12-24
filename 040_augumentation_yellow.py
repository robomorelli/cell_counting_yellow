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
image_ids.sort()
shift = len(image_ids)
id_edges = [300, 302, 1161, 1380, 1908, 206] #These numbers are valid if use our test

if __name__ == "__main__":

    try:
        with open('id_new_images.pickle', 'rb') as handle:
            dic = pickle.load(handle)
        id_new_images = dic['id_new_images']
    except:
        id_new_images = int(0.8*shift)

    print(id_new_images)

    parser = argparse.ArgumentParser(description='Define augmentation setting....default running follow paper requirements')
    parser.add_argument('--start_from_zero',action='store_true', help='remove previous file in the destination folder')

    parser.add_argument('--split_num', nargs="?", type=int, default=4, help='width of the crop')
    parser.add_argument('--split_num_new_images', nargs="?", type=int, default=10, help='height of the crop')

    parser.add_argument('--no_copy_images', action='store_const', const=True, default=False,
                        help='copy cropped in crop_aug images')
    parser.add_argument('--no_copy_masks', action='store_const', const=True, default=False,
                        help='copy cropped in crop_aug masks')

    parser.add_argument('--unique_split', type=int, default=0,
                        help='default is 0, define a different number and wil be the unique split for all the images')
    parser.add_argument('--no_artifact_aug', action='store_const', const=True, default=False,
                        help='run basic augmentation')

    args = parser.parse_args()

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        if (args.no_artifact_aug) | (args.unique_split):
            try:
                shutil.rmtree(AugCropImagesBasic)
            except:
                pass
            os.makedirs(AugCropImagesBasic,exist_ok=True)
            try:
                shutil.rmtree(AugCropMasksBasic)
            except:
                pass
            os.makedirs(AugCropMasksBasic,exist_ok=True)
        else:
            try:
                shutil.rmtree(AugCropImages)
            except:
                pass
            os.makedirs(AugCropImages,exist_ok=True)
            try:
                shutil.rmtree(AugCropMasks)
            except:
                pass
            os.makedirs(AugCropMasks,exist_ok=True)

    src_files = os.listdir(CropImages)
    # src_files_i.sort()
    # src_files_m = os.listdir(CropWeightedMasks)
    # src_files_m.sort()

    # limits = min((len(src_files_i), len(src_files_m)))

    if not args.no_copy_images:
        print('copying images')
        for file_name in src_files:
            full_file_name = os.path.join(CropImages, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, AugCropImages)

    if not args.no_copy_masks:
        print('copying masks')
        for file_name in src_files:
            full_file_name = os.path.join(CropWeightedMasks, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, AugCropMasks)

    make_data_augmentation(image_ids, CropImages,  CropWeightedMasks, args.split_num, id_new_images,
                           args.split_num_new_images, id_edges, AugCropImages, AugCropMasks,  shift
                           , args.unique_split, args.no_artifact_aug
                           )
