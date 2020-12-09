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

from config import *
from utils import *

image_ids = os.listdir(CropMasks)
ix = [int(x.split('.')[0]) for x in image_ids]
ix.sort()
image_ids = [str(x)+'.tiff' for x in ix]


if __name__ == "__main__":

    Normalize = False
    maximum = 3.8177538
    sigma = 25

    if Normalize:
        make_weights(image_ids,  CropMasks, CropWeightedMasks, sigma = sigma, maximum=False)
        with open('max_weight_{}.pickle'.format(sigma), 'rb') as handle:
            dic = pickle.load(handle)
        maximum = dic['max_weight']
        make_weights(image_ids,  CropMasks, CropWeightedMasks, sigma = sigma, maximum=maximum)
    else:
        make_weights(image_ids,  CropMasks, CropWeightedMasks, sigma = sigma, maximum=maximum)
