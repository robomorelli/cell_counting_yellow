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
from skimage.io import imread
from skimage.transform import rotate
from skimage.transform import resize
from subprocess import check_output
# import albumentations as alb
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology import watershed, remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize, binary_erosion, remove_small_holes
from skimage.feature import peak_local_max
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.segmentation import find_boundaries
import pickle
import argparse
import logging

from config import *
from utils import *

image_ids = os.listdir(CropMasks)
ix = [int(x.split('.')[0]) for x in image_ids]
ix.sort()
image_ids = [str(x)+'.tiff' for x in ix]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='weighting masks')

    parser.add_argument('--CropMasks', nargs="?", default = CropMasks, help='path including mask to crop')
    parser.add_argument('--CropWeightedMasks', nargs="?", default = CropWeightedMasks, help='path where save weighted mask')
    parser.add_argument('--save_images_path', nargs="?", default = CropImages, help='save images path')

    parser.add_argument('--normalize', action='store_const', const=True, default=False, help='find the maximum for normalization')
    parser.add_argument('--start_from_zero', action='store_const', const=True, default=False, help='delete all file in destination folder')
    parser.add_argument('--continue_after_normalization', action='store_const', const=True, default=False,  help='find the maximum for normalization')
    parser.add_argument('--resume_after_normalization', action='store_const', const=True, default=False, help='find the maximum for normalization')

    parser.add_argument('--maximum', nargs="?", type = int, default = 3.8177538,  help='Maximum value for normalization')
    parser.add_argument('--sigma', nargs="?", type = int, default = 25,  help='kernel for cumpling cell deacaying influence')

    args = parser.parse_args()

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        try:
            shutil.rmtree(CropWeightedMasks)
        except:
            pass
        os.makedirs(CropWeightedMasks)

        print('start new weighting mask')

    if args.normalize:
        make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=False)
        if args.continue_after_normalization:
            with open('max_weight_{}.pickle'.format(sigma), 'rb') as handle:
                dic = pickle.load(handle)
            maximum = dic['max_weight']
            make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=args.maximum)
        elif args.resume_after_normalization:
            with open('max_weight_{}.pickle'.format(sigma), 'rb') as handle:
                dic = pickle.load(handle)
            maximum = dic['max_weight']
            make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=args.maximum)
    else:
        make_weights(image_ids,  args.CropMasks, args.CropWeightedMasks, sigma = args.sigma, maximum=args.maximum)
