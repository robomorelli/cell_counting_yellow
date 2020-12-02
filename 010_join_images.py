import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects


IMG_WIDTH = 1600
IMG_HEIGHT = 1200
IMG_CHANNELS = 3

NumberTest = 60
tot_num = len(os.listdir('DATASET/original_images/images/'))

idx = 0


PATH = 'DATASET/original_images/images/'
PATH_Y = 'DATASET/original_masks/masks/'

PATH_TRAIN = 'DATASET/all_images/images/'
PATH_TRAIN_Y = 'DATASET/all_masks/masks/'

names = os.listdir(PATH)

for ix, name in tqdm(enumerate(names), total=len(names)):

    img_x = cv2.imread(PATH + name)
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

    if img_x is None:
        name_x = name.replace('TIF', 'tif')
        img_x = cv2.imread(PATH + name_x)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

    try:

        img_y = cv2.imread(PATH_Y + name)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

    except:

        name_y = name.replace('TIF', 'tif')
        img_y = cv2.imread(PATH_Y + name_y)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

    if len(np.unique(img_y)) > 2:
        print('prima più di due valori{} {} {}'.format(ix, name, np.unique(img_y)))

        ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

        print('dopo invece {} {} {}'.format(ix, name, np.unique(img_y)))

    img_y = img_y.astype(bool)
    img_y = remove_small_objects(img_y, min_size=15)
    img_y = img_y.astype(np.uint8) * 255

    img_dir = PATH_TRAIN + '{}.tiff'.format(ix)
    mask_dir = PATH_TRAIN_Y + '{}.tiff'.format(ix)
    plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
    plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

NEW_IMAGES = Path('DATASET/new_images/images/')
NEW_MASKS = Path('DATASET/new_masks/masks/')

new_images = os.listdir(NEW_IMAGES)
new_masks = os.listdir(NEW_MASKS)

idx = 0
shift = 252
for im_name, mask_name in tqdm(zip(new_images, new_masks), total=len(new_images)):

    print(im_name)
    img_x = cv2.imread(str(NEW_IMAGES) + '/' + im_name)
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
    img_y = cv2.imread(str(NEW_MASKS) + '/' + mask_name)
    img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

    if len(np.unique(img_y)) > 2:
        ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

    img_y = img_y.astype(bool)
    img_y = remove_small_objects(img_y, min_size=15)
    img_y = img_y.astype(np.uint8) * 255

    img_dir = PATH_TRAIN + '{}.tiff'.format(shift + idx)
    mask_dir = PATH_TRAIN_Y + '{}.tiff'.format(shift + idx)
    plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
    plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

    idx += 1