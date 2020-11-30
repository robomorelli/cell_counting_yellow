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
from skimage import io
from skimage.io import imread
from skimage.transform import rotate
from skimage.transform import resize
from subprocess import check_output
# import albumentations as alb
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology import watershed, remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize, binary_erosion, remove_small_holes
from skimage.feature import peak_local_max
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.segmentation import find_boundaries
import pickle

from config import *

image_ids = os.listdir(LoadMasksForWeight)

ix = [int(x.split('.')[0]) for x in image_ids]
ix.sort()

image_ids = [str(x)+'.tiff' for x in ix]

def make_weights(image_ids, sigma = 25, maximum=False):
    
    if not maximum:
        total = np.zeros((len(image_ids), 512, 512), dtype=np.float32) 
    
    for ax_index, name in tqdm(enumerate(image_ids),total=len(image_ids)):

        target = read_masks(name)[:,:,0:1]
        target = target.astype(bool)
        target = remove_small_objects(target,min_size = 100)  
        target = remove_small_holes(target,200)
        target = target.astype(np.uint8)*255

        tar_inv = cv2.bitwise_not(target)
        tar_dil = dilation(np.squeeze(target), selem=np.ones([100, 100]))

        mask_sum = cv2.bitwise_and(tar_dil, tar_inv)
        mask_sum1 = cv2.bitwise_or(mask_sum, target)

        null = np.zeros((target.shape[0], target.shape[1]), dtype = np.float32)
        weighted_mask = np.zeros((target.shape[0], target.shape[1]), dtype = np.float32)

        mask, nlabels_mask = ndimage.label(target)

        if nlabels_mask < 1:

            weighted_maskk = np.ones((target.shape[0], target.shape[1]), dtype = np.float32)

        else:

            mask = remove_small_objects(mask, min_size=25, connectivity=1, in_place=False)
            mask, nlabels_mask = ndimage.label(mask)
            mask_objs = ndimage.find_objects(mask)

            for idx,obj in enumerate(mask_objs):
                new_image = np.zeros_like(mask)
                new_image[obj[0].start:obj[0].stop,obj[1].start:obj[1].stop] = mask[obj]  

                new_image = np.clip(new_image, 0, 1).astype(np.uint8)
                new_image *= 255

                inverted = cv2.bitwise_not(new_image)

                distance = ndimage.distance_transform_edt(inverted)
                w = np.zeros((distance.shape[0],distance.shape[1]), dtype=np.float32)
                w1 = np.zeros((distance.shape[0],distance.shape[1]), dtype=np.float32)

                for i in range(distance.shape[0]):

                    for j in range(distance.shape[1]):

                        if distance[i, j] != 0:

                            w[i, j] = 1.*np.exp((-1 * (distance[i,j]) ** 2) / (2 * (sigma ** 2)))

                        else:

                            w[i, j] = 1

                weighted_mask = cv2.add(weighted_mask, w, mask = mask_sum)

            # Complete from inner to edge with 1.5 as weight 
            weighted_mask = np.clip(weighted_mask, 1, weighted_mask.max())

            mul = target*1.5/255
            mul = mul.astype(np.float32)
            mul = np.clip(mul,1,mul.max())

            weighted_maskk = cv2.multiply(weighted_mask, mul)
            
        target = np.clip(target, 0 , 1)
        final_target = np.dstack((target, weighted_maskkk, null))
    
        if not maximum:
            total[ax_index] = weighted_maskk
            
        else:     
            if (weighted_maskk.max()/(maximum+0.0001))> 1:
                break
            weighted_maskk = weighted_maskk*1/maximum
            mask_dir = SaveWeightMasks + '{}'.format(name)
            print('saving {}'.format(name))
            plt.imsave(fname=mask_dir,arr = final_target)
            
    if not maximum:   
        np.save('total.npy', total)
        dic = {}
        dic['max_weight'] = max(total)
        with open('max_weight_{}.pickle'.format(sigma), 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
            
if __name__ == "__main__":
    
    Normalize = False
    maximum = 
    sigma = 25
    
    if Normalize:
        make_cropper(image_ids, sigma = sigma, maximum = False, SaveCropImages, SaveCropMasks)
        with open('max_weight_{}.pickle'.format(sigma), 'rb') as handle:
            dic = pickle.load(handle)
        maximum = dic['max_weight']
        make_cropper(image_ids, sigma = sigma, maximum = maximum, SaveCropImages, SaveCropMasks)
    else:    
        make_cropper(image_ids, sigma = sigma, maximum = 3.8177538, SaveCropImages, SaveCropMasks)