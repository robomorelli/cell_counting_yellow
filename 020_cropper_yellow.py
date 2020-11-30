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
from utils import *
from config import *


IMG_WIDTH = 1600
IMG_HEIGTH = 1200
# Crop size 
XCropSize = 512
YCropSize = 512
# if the coord are lesser than the crop size
# overlapping between crop is allowed
# if XCropCoord = XCropSize and same for Y coord, no overlapping beetween crop
XCropCoord = 400
YCropCoord = 400

XCropNum = int(IMG_WIDTH/XCropCoord)
YCropNum = int(IMG_HEIGTH/YCropCoord)

NumCropped = int(IMG_WIDTH/XCropCoord * IMG_HEIGTH/YCropCoord)

YShift = YCropSize - YCropCoord
XShift = XCropSize - XCropCoord

x_coord = [XCropCoord*i for i in range(0, XCropNum+1)]
y_coord = [YCropCoord*i for i in range(0, YCropNum+1)]

image_ids = os.listdir(LoadImagesForCrop)
Number = [int(num.split('.')[0]) for num in image_ids]
Number.sort()

image_ids = [str(num) + '.tiff' for num in Number]

if __name__ == "__main__":
    
    make_cropper(image_ids, SaveCropImages,SaveCropMasks)
