#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Copyright 2018 Luca Clissa, Marco Dalla, Roberto Morelli
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""
Created on Wed Jan  9 19:45:22 2019

@author: Roberto Morelli
"""

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
from shutil import move

from config_script import *


def SplitImagesInFolders(folder, fol_type):

    file_per_folder = 1000

    images = os.listdir(str(folder))
    images.sort()

    if len(images) % file_per_folder != 0:
        nfolder = len(images) // file_per_folder + 1
    else:
        nfolder = len(images) // file_per_folder

    for i in range(nfolder):
        os.makedirs(str(folder.parent / fol_type)+str(i),exist_ok=True)

    for i,name in tqdm(enumerate(images)):
        fol = i // file_per_folder
        dest_name = fol_type+str(fol)+'/'+name
        move(str(folder / name), str(folder.parent / dest_name))

if __name__ == "__main__":

    SplitImagesInFolders((ALL_IMAGES), 'images')
    SplitImagesInFolders((ALL_MASKS), 'masks')
