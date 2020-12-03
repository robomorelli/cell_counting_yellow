#!/usr/bin/python3

IMG_WIDTH = 1600
IMG_HEIGHT = 1200

OriginalImages = 'DATASET/original_images/images/'
OriginalMasks = 'DATASET/original_masks/masks/'

NewImages = 'DATASET/new_images/images/'
NewMasks = 'DATASET/new_masks/masks/'

# Temporary folder for the union of original and new images, final dataset that we are going to share is going to be in
# DATASET/train_val/{images, masks}_before_crop and DATASET/test/all_{images,masks}
AllImages = 'DATASET/all_images/images/'
AllMasks = 'DATASET/all_masks/masks/'

TrainValImages = 'DATASET/train_val/images_before_crop/images/'
TrainValMasks = 'DATASET/train_val/masks_before_crop/masks/'

TestImages = 'DATASET/test/all_images/images/'
TestMasks = 'DATASET/test/all_masks/masks/'

# Folder for all cropped images and masks
CropImages = 'DATASET/all_cropped_images/images/'
CropMasks = 'DATASET/all_cropped_masks/masks/'

# Final folder where the images for train reside, already cropped and augmented and weighted
# Cropped and augmented images are going to be read from the same folder
AugImages = 'DATASET/train_val/all_images/images/'
AugMasks = 'DATASET/train_val/all_masks/masks/'
