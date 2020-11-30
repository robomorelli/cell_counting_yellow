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


image_ids = os.listdir(LoadImagesForCrop)
Number = [int(num.split('.')[0]) for num in image_ids]
Number.sort()
image_ids = [str(num) + '.tiff' for num in Number]

if __name__ == "__main__":
    
    make_cropper(image_ids, images_path , masks_path,
                 XCropSize=512, YCropSize=512, XCropCoord=400, YCropCoord = 400,
                 SaveCropImages, SaveCropMasks, shift = shift)
