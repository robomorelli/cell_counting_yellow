#!/usr/bin/python3

LoadImagesForCrop =  'DATA/TRAIN_VAL/all_images/images/'
LoadLabelsForCrop = 'DATA/TRAIN_VAL/all_masks/masks/'
SaveCropImages =  'DATA/TRAIN_VAL/all_cropped_images/images/' 
SaveCropMasks = 'DATA/TRAIN_VAL/all_cropped_masks/masks/'

LoadImagesForAug =  './DATA/TRAIN_VAL/all_cropped_images/images/'
LoadLabelsForAug = './DATA/TRAIN_VAL/all_weighted_masks/masks/'
SaveAugImages =  './DATA/TRAIN_VAL/all_cropped_images/images/'
SaveAugMasks = './DATA/TRAIN_VAL/all_weighted_masks/masks/'

LoadMasksForWeight = './DATA/TRAIN_VAL/all_cropped_masks/masks/'
LoadImgsForWeight = './DATA/TRAIN_VAL/all_cropped_images/images/'
SaveWeightMasks = './DATA/TRAIN_VAL/all_weighted_masks/masks/'