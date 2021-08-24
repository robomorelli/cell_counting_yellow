# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 Luca Clissa
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
@author: Luca Clissa
"""
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import *

test_img_names = ['Mar21bS1C2R2_VLPAGl_200x_y.png',
                  'Mar20bS1C4R3_DMl_200x_y.png',
                  'Mar19bS1C5R3_DMr_200x_y.png',
                  'Mar27bS1C2R1_LHl_200x_y.png',
                  'Mar19bS1C4R1_VLPAGr_200x_y.png',
                  'Mar19bS1C3R2_VLPAGl_200x_y.png',
                  'Mar26bS1C4R3_DMl_200x_y.png',
                  'Mar21bS1C1R3_VLPAGr_200x_y.png',
                  'Mar20bS1C3R2_VLPAGl_200x_y.png',
                  'Mar22bS1C4R1_LHl_200x_y.png',
                  'Mar24bS1C3R2_LHr_200x_y.png',
                  'Mar20bS1C4R3_DMr_200x_y.png',
                  'Mar20bS1C2R3_VLPAGr_200x_y.png',
                  'Mar20bS1C2R2_VLPAGl_200x_y.png',
                  'Mar26bS1C4R2_LHl_200x_y.png',
                  'Mar19bS1C4R3_DMr_200x_y.png',
                  'Mar24bS1C2R2_LHl_200x_y.png',
                  'Mar20bS2C1R1_LHl_200x_y.png',
                  'Mar19bS1C1R2_VLPAGr_200x_y.png',
                  'Mar24bS2C4R3_DMr_200x_y.png',
                  'Mar23bS1C6R1_DMr_200x_y.png',
                  'Mar19bS1C3R2_VLPAGr_200x_y.png',
                  'Mar24bS1C1R1_LHl_200x_y.png',
                  'Mar21bS2C1R2_LHl_200x_y.png',
                  'Mar21bS1C2R3_VLPAGr_200x_y.png',
                  'Mar19bS1C2R3_VLPAGr_200x_y.png',
                  'Mar20bS2C1R3_DMl_200x_y.png',
                  'Mar26bS2C2R2_DMr_200x_y.png',
                  'Mar21bS1C2R1_VLPAGl_200x_y.png',
                  'Mar22bS1C4R2_LHr_200x_y.png',
                  'Mar22bS1C3R2_DMr_200x_y.png',
                  'Mar24bS2C2R3_VLPAGl_200x_y.png',
                  'Mar24bS1C2R1_DMl_200x_y.png',
                  'Mar20bS1C4R1_DMr_200x_y.png',
                  'Mar19bS1C5R2_DMr_200x_y.png',
                  'Mar20bS2C2R3_LHr_200x_y.png',
                  'Mar21bS2C2R2_LHr_200x_y.png',
                  'Mar20bS1C1R3_VLPAGr_200x_y.png',
                  'Mar22bS2C1R1_LHr_200x_y.png',
                  'Mar20bS1C2R1_VLPAGl_200x_y.png',
                  'Mar19bS1C5R3_LHl_200x_y.png',
                  'Mar26bS2C2R2_LHr_200x_y.png',
                  'Mar19bS1C5R2_DMl_200x_y.png',
                  'Mar26bS2C2R1_DMl_200x_y.png',
                  'Mar22bS1C2R1_VLPAGl_200x_y.png',
                  'Mar20bS2C2R3_LHl_200x_y.png',
                  'Mar19bS1C1R3_VLPAGl_200x_y.png',
                  'Mar27bS1C3R1_LHr_200x_y.png',
                  'Mar24bS1C3R1_LHr_200x_y.png',
                  'Mar24bS1C1R2_DMr_200x_y.png',
                  'Mar19bS1C4R2_LHr_200x_y.png',
                  'Mar21bS2C1R1_LHr_200x_y.png',
                  'Mar20bS1C4R1_LHl_200x_y.png',
                  'Mar26bS2C1R1_DMr_200x_y.png',
                  'Mar19bS1C5R2_LHl_200x_y.png',
                  'Mar21bS2C2R3_LHl_200x_y.png',
                  'Mar19bS1C4R3_LHr_200x_y.png',
                  'Mar23bS2C1R1_LHl_200x_y.png',
                  'Mar23bS1C1R4_VLPAGl_200x_y.png',
                  '39_y.png',
                  'Mar31bS2C1R2_VLPAGr_200x_y.png',
                  'Mar32bS2C2R2_DMl_200x_y.png',
                  'Mar33bS2C1R1_DMl_200x_y.png',
                  'MAR38S1C3R1_LHR_20_o.png',
                  'Mar42S2C4R2_VLPAGr_200x_o.png',
                  'Mar31bS2C3R4_DMr_200x_y.png',
                  'Mar33bS1C4R2_DMl_200x_y.png',
                  'Mar36bS1C6R2_DMr_200x_y.png',
                  'Mar42S2C2R2_DMr_200x_o.png',
                  'MAR55S1C5R3_DMR_20_o.png']


def main():
    # logging.basicConfig(filename='example.log',level=logging.INFO)

    # root = '/home/luca/workspace/'
    # AllImages = root + 'DATASET/all_images/images/'
    # AllMasks = root + 'DATASET/all_masks/masks/'
    #
    # # create folder for train/validation data
    # TrainValImages = root + 'DATASET/train_val/full_size/all_images/images/'
    # TrainValMasks = root + 'DATASET/train_val/full_size/all_masks/masks/'
    #
    # if not os.path.exists(TrainValImages):
    #     os.makedirs(TrainValImages)
    #
    # if not os.path.exists(TrainValMasks):
    #     os.makedirs(TrainValMasks)
    #
    # # create folder for test data
    # TestImages = root + 'DATASET/test/all_images/images/'
    # TestMasks = root + 'DATASET/test/all_masks/masks/'
    #
    # if not os.path.exists(TestImages):
    #     os.makedirs(TestImages)
    #
    # if not os.path.exists(TestMasks):
    #     os.makedirs(TestMasks)

    names = os.listdir(AllImages)
    names.sort()
    # logging.info(names)

    number_names = []
    test_bool = []
    for ix, name in tqdm(enumerate(names), total=len(names)):

        # name = names[0]
        img_x = cv2.imread(AllImages + name)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        if img_x is None:
            name_x = name.replace('TIF', 'tif')
            img_x = cv2.imread(OriginalImages + name_x)
            img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        img_y = cv2.imread(AllMasks + name)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)[:, :, 0:1]

        number_name = f'{ix}.png'
        number_names.append(number_name)
        if name in test_img_names:
            outpath_img = TestImages + number_name
            outpath_mask = TestMasks + number_name
            test_bool.append(True)
        else:
            outpath_img = TrainValImages + number_name
            outpath_mask = TrainValMasks + number_name
            test_bool.append(False)
        plt.imsave(fname=outpath_img, arr=np.squeeze(img_x))
        plt.imsave(fname=outpath_mask, arr=np.squeeze(img_y), cmap='gray')

    split_map_df = pd.DataFrame({'orig_name': names, 'number_name': number_names, 'is_test': test_bool})
    split_map_df.to_csv(root + 'DATASET/test_split_map.csv', index=False)

if __name__ == "__main__":
    main()
