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
import pickle
# from numba import jit
from skimage.morphology import erosion
import argparse
import logging

from utils import *
from config import *


image_ids = os.listdir(TrainValImages)
Number = [int(num.split('.')[0]) for num in image_ids]
Number.sort()
image_ids = [str(num) + '.tiff' for num in Number]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for crop.')

    parser.add_argument('--images_path', nargs="?", type=path, default = TrainValImages, help='the folder including the images to crop')
    parser.add_argument('--masks_path', nargs="?", type=path, default = TrainValMasks, help='the folder including the masks to crop')
    parser.add_argument('--save_images_path', nargs="?", type=path, default = CropImages, help='save images path')
    parser.add_argument('--save_masks_path', nargs="?", type=path, default = CropMasks, help='save masks path')

    parser.add_argument('--x_size', nargs="?", default = 512,  help='width of the crop')
    parser.add_argument('--y_size', nargs="?", default = 512,  help='height of the crop')
    parser.add_argument('--x_distance', nargs="?", default = 400,  help='distance beetwen cuts points on the x axis')
    parser.add_argument('--y_distance', nargs="?", default = 400,  help='distance beetwen cuts points on the y axis')

    parser.add_argument('--img_width', nargs="?", default = IMG_WIDTH,  help='width of images to crop')
    parser.add_argument('--img_height', nargs="?", default = IMG_HEIGHT,  help='height of images to crop')

    args = parser.parse_args()

    make_cropper(image_ids = image_ids, images_path = args.images_path, mask_path = args.mask_path
                 ,SaveCropImages = args.save_images_path, SaveCropMasks = args.save_masks_path,
                 XCropSize = args.x_size, YCropSize = args.y_size, XCropCoord = args.x_distance, YCropCoord = args.y_distance,
                 img_width = args.img_width, img_height = args.img_height, shift = 0)
