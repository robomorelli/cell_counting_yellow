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
import shutil

from config import *
from utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for crop.')

    parser.add_argument('--start_from_zero', action='store_const', const=True, default=False,
                        help='remove previous file in the destination folder')

    parser.add_argument('--images_path', nargs="?", default=TrainValImages,
                        help='the folder including the images to crop')
    parser.add_argument('--masks_path', nargs="?", default=TrainValMasks, help='the folder including the masks to crop')
    parser.add_argument('--save_images_path', nargs="?", default=CropImages, help='save images path')
    parser.add_argument('--save_masks_path', nargs="?", default=CropMasks, help='save masks path')

    parser.add_argument('--x_size', nargs="?", type=int, default=512, help='width of the crop')
    parser.add_argument('--y_size', nargs="?", type=int, default=512, help='height of the crop')
    parser.add_argument('--x_distance', nargs="?", type=int, default=400,
                        help='distance beetwen cuts points on the x axis')
    parser.add_argument('--y_distance', nargs="?", type=int, default=400,
                        help='distance beetwen cuts points on the y axis')

    parser.add_argument('--img_width', nargs="?", type=int, default=IMG_WIDTH, help='width of images to crop')
    parser.add_argument('--img_height', nargs="?", type=int, default=IMG_HEIGHT, help='height of images to crop')

    args = parser.parse_args()

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        try:
            shutil.rmtree(CropImages)
        except:
            pass
        os.makedirs(CropImages,exist_ok=True)

        try:
            shutil.rmtree(CropMasks)
        except:
            pass
        os.makedirs(CropMasks,exist_ok=True)

        print('start to crop')

    image_ids = os.listdir(args.images_path)
    image_ids.sort()
    Number = [int(num.split('.')[0]) for num in image_ids]
    Number.sort()
    image_ids = [str(num) + '.tiff' for num in Number]

    make_cropper(image_ids=image_ids, images_path=args.images_path, masks_path=args.masks_path
                 , SaveCropImages=args.save_images_path, SaveCropMasks=args.save_masks_path,
                 XCropSize=args.x_size, YCropSize=args.y_size, XCropCoord=args.x_distance, YCropCoord=args.y_distance,
                 img_width=args.img_width, img_height=args.img_height, shift=0)
