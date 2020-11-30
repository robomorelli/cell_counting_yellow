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

from albumentations import (RandomCrop,CenterCrop,ElasticTransform,RGBShift,Rotate,
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,Transpose,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip, HorizontalFlip,
    
    HueSaturationValue,
)

from augmentation_utils import *
from config import *
from utils import *

image_ids = os.listdir('./DATA/TRAIN_VAL/all_cropped_images/images/')
IMG_WIDTH = 1600
IMG_HEIGTH = 1200

split_num = 5
split_num_new_images = 11
shift = len(image_ids)

if __name__ == "__main__":
    
    make_data_augmentation(image_ids, LoadImagesForAug,  LoadMasksForAug, split_num, id_new_images,
                           split_num_new_images, ix, SaveAugImages, SaveAugMasks)


