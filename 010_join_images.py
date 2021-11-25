# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 Luca Clissa, Marco Dalla, Roberto Morelli
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on Tue May  7 10:42:13 2019
@author: Roberto Morelli
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import remove_small_objects, erosion
from tqdm import tqdm
import logging
from config import *

IMG_CHANNELS = 3


def main():
    # logging.basicConfig(filename='example.log',level=logging.INFO)
    names = os.listdir(OriginalImages)
    names.sort()
    # logging.info(names)
    for ix, name in tqdm(enumerate(names), total=len(names)):

        img_x = cv2.imread(OriginalImages + name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        if img_x is None:
            name_x = name.replace('TIF', 'tif')
            img_x = cv2.imread(OriginalImages + name_x)
            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        try:

            img_y = cv2.imread(OriginalMasks + name)
            img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        except:

            name_y = name.replace('TIF', 'tif')
            img_y = cv2.imread(OriginalMasks + name_y)
            img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        if len(np.unique(img_y)) > 2:
            print('before the threshold we have more than 2 grayscale values {} {} {}'.format(ix, name, np.unique(img_y)))

            ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

            print('after the threshold we have {} {} {}'.format(ix, name, np.unique(img_y)))

        img_y = img_y.astype(bool)
        img_y = remove_small_objects(img_y, min_size=25)
        # img_y = erosion(np.squeeze(img_y), selem=np.ones([2, 2]))
        img_y = img_y.astype(np.uint8) * 255

        img_dir = AllImages + '{}.tiff'.format(ix)
        mask_dir = AllMasks + '{}.tiff'.format(ix)
        plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
        plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

    new_images = os.listdir(NewImages)
    new_images.sort()
    new_masks = os.listdir(NewMasks)
    new_masks.sort()

    idx = 0
    shift = 252 #pay attention to set oslistdir:imagine to run again >>> it will be 272 and not 252
    for im_name, mask_name in tqdm(zip(new_images, new_masks), total=len(new_images)):

        print(im_name)
        img_x = cv2.imread(str(NewImages) + '/' + im_name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_y = cv2.imread(str(NewMasks) + '/' + mask_name)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        if len(np.unique(img_y)) > 2:
            ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

        img_y = img_y.astype(bool)
        img_y = remove_small_objects(img_y, min_size=25)
        img_y = img_y.astype(np.uint8) * 255

        img_dir = AllImages + '{}.tiff'.format(shift + idx)
        mask_dir = AllMasks + '{}.tiff'.format(shift + idx)
        plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
        plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

        idx += 1


if __name__ == "__main__":
    main()
