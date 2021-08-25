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

import argparse
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import *

IMG_CHANNELS = 3


# Split images it train_val and test folder
# during the reading the images if the binary mask has more than two values:
# 0 for black and 255 for white this could be related to spurious saving effect from
# GIMP and imageJ software (interpolation???). We need to reset only 0 and 255 value
# We did in the #############Processing############### section below
def main():
    for ix, im_name in enumerate(images_name):
        #############Processing###############
        print(im_name)
        img_x = cv2.imread(str(AllImages) + im_name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_y = cv2.imread(str(AllMasks) + im_name)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        #############Processing###############
        if len(np.unique(img_y)) > 2:
            print(' restoring {}'.format(im_name))

            ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

        img_y = img_y.astype(bool)
        img_y = img_y.astype(np.uint8) * 255

        #############Saving in new folder###############
        if int(im_name.split('.')[0]) in test_names:
            print('test')
            img_dir = TestImages + '{}'.format(im_name)
            mask_dir = TestMasks + '{}'.format(im_name)
            plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
            plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

        else:
            img_dir = TrainValImages + '{}'.format(im_name)
            mask_dir = TrainValMasks + '{}'.format(im_name)
            plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
            plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')

    # NEW TEST IMAGES
    test_images_name = os.listdir(NewTestImages)
    # test_images_name = [x.replace('TIF', 'tif') for x in test_images_name]

    test_masks_name = ['_'.join(x.split('_')[:2])+'_mask_'+'_'.join(x.split('_')[2:]) for x in test_images_name]
    test_masks_name = [x.replace('TIF', 'tif') for x in test_masks_name]

    for ix, im_name in enumerate(test_images_name):
        #############Processing###############
        print(im_name)
        img_x = cv2.imread(str(NewTestImages) + im_name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_y = cv2.imread(str(NewTestsMasks) + test_masks_name[ix])
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        #############Processing###############
        if len(np.unique(img_y)) > 2:
            print(' restoring {}'.format(im_name))

            ret, img_y = cv2.threshold(img_y, 75, 255, cv2.THRESH_BINARY)

        img_y = img_y.astype(bool)
        img_y = img_y.astype(np.uint8) * 255

        #############Saving in new folder###############
        img_dir = TestImages + '{}'.format(ix + tot_num) + '.tiff'
        mask_dir = TestMasks + '{}'.format(ix + tot_num) + '.tiff'
        plt.imsave(fname=img_dir, arr=np.squeeze(img_x))
        plt.imsave(fname=mask_dir, arr=np.squeeze(img_y), cmap='gray')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--start_from_zero', action='store_const', const=True, default=False,
                        help='remove previous file in the destination folder')
    parser.add_argument('--random_test_set', action='store_const', const=True, default=False,
                        help='use our test set instead of a random generated one')
    args = parser.parse_args()

    tot_num = len(os.listdir(AllImages))
    NumberTest = 58
    test_names = []
    UpperLimit = tot_num - len(os.listdir(NewImages))

    if args.random_test_set:
        random.seed(a=NumberTest, version=2)  # FIX the SEED#

        while len(test_names) < NumberTest:
            x = random.randint(0, UpperLimit)
            if x not in test_names:
                test_names.append(x)
            else:
                continue

        test_names.append(254)  # maccheroni images needed in the test
        test_names.append(81)  # yellow strip artifact

    else:
        test_names = [148, 50, 52, 189, 164, 251, 242, 51, 10, 49, 115, 103, 90, 241, 73, 206, 224, 66, 247, 205,
                      157, 107, 72, 223, 26, 3, 125, 54, 120, 193, 18, 141, 168, 96, 94, 15, 25, 200, 170, 199,
                      34, 77, 8, 47, 222, 75, 79, 44, 156, 154, 185, 62, 194, 174, 233, 19, 40, 114]

        test_names.append(254)
        test_names.append(81)
        print('test_names {}'.format(test_names))

    test_names.sort()

    # Our images name are 0.tiff, 1.tiff and so on
    # the format is .tiff and no more .TIF as the firts experiment run
    # I failed to save in .TIF format when reading from original dataset
    # If we want to fix this, we need to change 010_load_file_join_all_images
    images_name = os.listdir(AllImages)
    images_name.sort()
    # select only the number
    images_name = [int(x.split('.')[0]) for x in images_name]
    # sort the images
    images_name.sort()
    # restore the original name
    images_name = [str(x) + '.tiff' for x in images_name]

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        shutil.rmtree(TrainValImages)
        os.makedirs(TrainValImages,exist_ok=True)
        shutil.rmtree(TrainValMasks)
        os.makedirs(TrainValMasks,exist_ok=True)

        shutil.rmtree(TestImages)
        os.makedirs(TestImages,exist_ok=True)
        shutil.rmtree(TestMasks)
        os.makedirs(TestMasks,exist_ok=True)
        print('Splitting image in train_val and test')

    main()
