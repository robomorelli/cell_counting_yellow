#!/usr/bin/python3
import os
import subprocess
from subprocess import PIPE

IMG_WIDTH = 1600
IMG_HEIGHT = 1200

try:
    o = subprocess.run(["hostname"],  capture_output=True)
    o = o.stdout
    o = o.strip().decode('utf-8')
except:
    o = subprocess.run(["hostname"],  stdout=PIPE, stderr=PIPE)
    o = o.stdout
    o = o.strip().decode('utf-8')

if 'cnaf' in o:
    root = '/storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/'
else:
    root = ''
OriginalImages = root + 'DATASET/original_images/images/'
OriginalMasks = root + 'DATASET/original_masks/masks/'

if not os.path.exists(OriginalImages):
    os.makedirs(OriginalImages)

if not os.path.exists(OriginalMasks):
    os.makedirs(OriginalMasks)

NewImages = root + 'DATASET/new_images/images/'
NewMasks = root + 'DATASET/new_masks/masks/'

if not os.path.exists(NewImages):
    os.makedirs(NewImages)

if not os.path.exists(NewMasks):
    os.makedirs(NewMasks)

# Temporary folder for the union of original and new images, final dataset that we are going to share is going to be in
# DATASET/train_val/{images, masks}_before_crop and DATASET/test/all_{images,masks}
AllImages = root + 'DATASET/all_images/images/'
AllMasks = root + 'DATASET/all_masks/masks/'

if not os.path.exists(AllImages):
    os.makedirs(AllImages)

if not os.path.exists(AllMasks):
    os.makedirs(AllMasks)

#I'd like to change in this way
TrainValImages = root + 'DATASET/train_val/full_size/images/'
TrainValMasks = root + 'DATASET/train_val/full_size/masks/'

if not os.path.exists(TrainValImages):
    os.makedirs(TrainValImages)

if not os.path.exists(TrainValMasks):
    os.makedirs(TrainValMasks)

TestImages = root + 'DATASET/test/all_images/images/'
TestMasks = root + 'DATASET/test/all_masks/masks/'

if not os.path.exists(TestImages):
    os.makedirs(TestImages)

if not os.path.exists(TestMasks):
    os.makedirs(TestMasks)

#This should be only for train_val
CropImages = root + 'DATASET/train_val/cropped/images/'
CropMasks = root + 'DATASET/train_val/cropped/masks/'
CropWeightedMasks = root + 'DATASET/train_val/cropped/weighted_masks/'

if not os.path.exists(CropImages):
    os.makedirs(CropImages)

if not os.path.exists(CropMasks):
    os.makedirs(CropMasks)
if not os.path.exists(CropWeightedMasks):
    os.makedirs(CropWeightedMasks)

# Final folder where the images for train reside, already cropped and augmented and weighted
# Cropped and augmented images are going to be read from the same folder
AugCropImages = root + 'DATASET/train_val/crop_augmented/images/'
AugCropMasks = root + 'DATASET/train_val/crop_augmented/masks/'

AugCropImagesSplitted = root + 'DATASET/train_val/crop_augmented_splitted/images/'
AugCropMasksSplitted = root + 'DATASET/train_val/crop_augmented_splitted/masks/'

AugCropImagesBasic = root + 'DATASET/train_val/crop_augmented_basic/images/'
AugCropMasksBasic = root + 'DATASET/train_val/crop_augmented_basic/masks/'

AugCropImagesBasicSplitted = root + 'DATASET/train_val/crop_augmented_basic_splitted/images/'
AugCropMasksBasicSplitted = root + 'DATASET/train_val/crop_augmented_basic_splitted/masks/'

if not os.path.exists(AugCropImages):
    os.makedirs(AugCropImages)

if not os.path.exists(AugCropMasks):
    os.makedirs(AugCropMasks)

if not os.path.exists(AugCropImagesBasic):
    os.makedirs(AugCropImagesBasic)

if not os.path.exists(AugCropMasksBasic):
    os.makedirs(AugCropMasksBasic)

ModelResults = root + 'model_results/'

if not os.path.exists(ModelResults):
    os.makedirs(ModelResults)
