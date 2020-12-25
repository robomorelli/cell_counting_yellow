#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Luca Clissa, Marco Dalla, Roberto Morelli
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
Created on Wed Jan  9 19:45:22 2019

@author: Roberto Morelli
"""

from tqdm import tqdm
from shutil import move, copy
from pathlib import Path
import argparse
import shutil

from config import *


def split_images_in_folder(folder, fol_type):
    file_per_folder = 1000

    images = os.listdir(str(folder))
    images.sort()

    if len(images) % file_per_folder != 0:
        n_folder = len(images) // file_per_folder + 1
    else:
        n_folder = len(images) // file_per_folder

    for i in range(n_folder):
        path = str(folder.parent) + '_splitted_{}/{}/{}{}'.format(fol_type, fol_type, fol_type, str(i))
        os.makedirs(path , exist_ok=True)

    for i, name in tqdm(enumerate(images)):
        fol = i // file_per_folder
        path = str(folder.parent) + '_splitted_{}/{}/{}{}'.format(fol_type, fol_type, fol_type, str(fol))
        dest_name = path + '/' + name
        copy(str(folder / name), str(dest_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='weighting masks')

    parser.add_argument('--images', nargs="?", default = AugCropImages, help='path including images for training')
    parser.add_argument('--start_from_zero', action='store_const', const=True, default=True, help='remove previous file in the destination folder')

    args = parser.parse_args()
    if args.images == 'AugCropImagesBasic':
        images_path = AugCropImagesBasic
        masks_path = AugCropMasksBasic
    else:
        images_path = AugCropImages
        masks_path = AugCropMasks

    if args.start_from_zero:
        print('deleting existing files in destination folder')

        try:
            shutil.rmtree(str(Path(images_path).parent) + '_splitted_images')
        except:
            pass
        os.makedirs(str(Path(images_path).parent) + '_splitted_images',exist_ok=True)

        try:
            shutil.rmtree(str(Path(masks_path).parent) + '_splitted_images')
        except:
            pass
        os.makedirs(str(Path(masks_path).parent) + '_splitted_images',exist_ok=True)

        print('Splitting image in train_val and test')

    split_images_in_folder(Path(images_path), 'images')
    split_images_in_folder(Path(masks_path), 'masks')
