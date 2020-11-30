import numpy as np
import imageio
import cv2
from config import *

def read_masks(path, image_id):

        mask = cv2.imread(path + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return mask
    
def read_images(path, image_id):
     
        img = cv2.imread(path + image_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
def read_image_masks(image_id, images_path,  masks_path):
     
        x = cv2.imread(images_path + image_id)
        image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(masks_path, + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return image, mask

def cropper(image, mask):
    
    CroppedImgs = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
    CroppedMasks = np.zeros((NumCropped, YCropSize, XCropSize), np.uint8)
    idx = 0
    
    for i in range(0, 4):
        for j in range(0, 3):
                
                if (i == 0) & (j == 0):
                    CroppedImgs[idx] = image[y_coord[j]:y_coord[j+1] + YShift, x_coord[i]:x_coord[i+1] + XShift]
                    CroppedMasks[idx] = mask[y_coord[j]:y_coord[j+1] + YShift, x_coord[i]:x_coord[i+1] + XShift]
                    idx +=1 

                if (i == 0) & (j != 0):
                    CroppedImgs[idx] = image[y_coord[j] - YShift : y_coord[j+1], x_coord[i]:x_coord[i+1] + XShift]
                    CroppedMasks[idx] = mask[y_coord[j] - YShift : y_coord[j+1], x_coord[i]:x_coord[i+1] + XShift]
                    idx +=1 

                if (i != 0) &  (j == 0):
                    CroppedImgs[idx] = image[y_coord[j]:y_coord[j+1] + YShift, x_coord[i] - XShift :x_coord[i+1]]
                    CroppedMasks[idx] = mask[y_coord[j]:y_coord[j+1] + YShift, x_coord[i] - XShift :x_coord[i+1]]
                    idx +=1 

                if (i != 0) &  (j != 0):
                    CroppedImgs[idx] = image[y_coord[j] - YShift : y_coord[j+1], x_coord[i] - XShift :x_coord[i+1]]
                    CroppedMasks[idx] = mask[y_coord[j] - YShift : y_coord[j+1], x_coord[i] - XShift :x_coord[i+1]]
                    idx +=1   
            
    return CroppedImgs, CroppedMasks

def make_cropper(image_ids, images_path , masks_path,
                 XCropSize=512, YCropSize=512, XCropCoord=400, YCropCoord = 400,
                 SaveCropImages, SaveCropMasks, shift = 0):
    ix = shift
    # Crop size 
#     XCropSize = 512
#     YCropSize = 512
    # if the coord are lesser than the crop size
    # overlapping between crop is allowed
    # if XCropCoord = XCropSize and same for Y coord, no overlapping beetween crop
#     XCropCoord = 400
#     YCropCoord = 400

    XCropNum = int(IMG_WIDTH/XCropCoord)
    YCropNum = int(IMG_HEIGTH/YCropCoord)

    NumCropped = int(IMG_WIDTH/XCropCoord * IMG_HEIGTH/YCropCoord)

    YShift = YCropSize - YCropCoord
    XShift = XCropSize - XCropCoord

    x_coord = [XCropCoord*i for i in range(0, XCropNum+1)]
    y_coord = [YCropCoord*i for i in range(0, YCropNum+1)]

    
    for ax_index, name in tqdm(enumerate(image_ids),total=len(image_ids)):
        
        image, mask = read_image_masks(name, images_path,  masks_path) 
        if int(name.split('.')[0]) <= 252:
            mask = erosion(np.squeeze(mask[:,:,0:1]), selem=np.ones([2,2]))
        else:
             mask = np.squeeze(mask[:,:,0:1])
            
        CroppedImages, CroppedMasks = cropper(image, mask)                                     
        
        for i in range(0,NumCropped):
            
            crop_imgs_dir = SaveCropImages + '{}.tiff'.format(ix)
            crop_masks_dir = SaveCropMasks + '{}.tiff'.format(ix) 

            plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
            plt.imsave(fname= crop_masks_dir,arr = CroppedMasks[i], cmap='gray')

            ix +=1
    return


    
def make_weights(image_ids,  LoadMasksForWeight, sigma = 25
                 , maximum=False, SaveWeightMasks):
    
    if not maximum:
        total = np.zeros((len(image_ids), 512, 512), dtype=np.float32) 
    
    for ax_index, name in tqdm(enumerate(image_ids),total=len(image_ids)):

        target = read_masks(LoadMasksForWeight, name)[:,:,0:1]
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
            
    return 