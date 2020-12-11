#!/usr/bin/python3
import os

IMG_WIDTH = 1600
IMG_HEIGHT = 1200

OriginalImages = 'DATASET/original_images/images/'
OriginalMasks = 'DATASET/original_masks/masks/'

if not os.path.exists(OriginalImages):
    os.makedirs(OriginalImages)

if not os.path.exists(OriginalMasks):
    os.makedirs(OriginalMasks)

NewImages = 'DATASET/new_images/images/'
NewMasks = 'DATASET/new_masks/masks/'

if not os.path.exists(NewImages):
    os.makedirs(NewImages)

if not os.path.exists(NewMasks):
    os.makedirs(NewMasks)

# Temporary folder for the union of original and new images, final dataset that we are going to share is going to be in
# DATASET/train_val/{images, masks}_before_crop and DATASET/test/all_{images,masks}
AllImages = 'DATASET/all_images/images/'
AllMasks = 'DATASET/all_masks/masks/'

if not os.path.exists(AllImages):
    os.makedirs(AllImages)

if not os.path.exists(AllMasks):
    os.makedirs(AllMasks)

#I'd like to change in this way
TrainValImages = 'DATASET/train_val/full_size/images/'
TrainValMasks = 'DATASET/train_val/full_size/masks/'

if not os.path.exists(TrainValImages):
    os.makedirs(TrainValImages)

if not os.path.exists(TrainValMasks):
    os.makedirs(TrainValMasks)

TestImages = 'DATASET/test/all_images/images/'
TestMasks = 'DATASET/test/all_masks/masks/'

if not os.path.exists(TestImages):
    os.makedirs(TestImages)

if not os.path.exists(TestMasks):
    os.makedirs(TestMasks)

#This should be only for train_val
CropImages = 'DATASET/train_val/cropped/images/'
CropMasks = 'DATASET/train_val/cropped/masks/'
CropWeightedMasks = 'DATASET/train_val/cropped/weighted_masks/'

if not os.path.exists(CropImages):
    os.makedirs(CropImages)

if not os.path.exists(CropMasks):
    os.makedirs(CropMasks)
if not os.path.exists(CropWeightedMasks):
    os.makedirs(CropWeightedMasks)

# Final folder where the images for train reside, already cropped and augmented and weighted
# Cropped and augmented images are going to be read from the same folder
AugCropImages = 'DATASET/train_val/crop_augmented/images/'
AugCropMasks = 'DATASET/train_val/crop_augmented/masks/'

if not os.path.exists(AugCropImages):
    os.makedirs(AugCropImages)

if not os.path.exists(AugCropMasks):
    os.makedirs(AugCropMasks)
