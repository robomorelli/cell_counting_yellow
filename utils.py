import numpy as np
import imageio
import cv2


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

def read_image_labels(image_id):
     
        x = cv2.imread(LoadImagesForCrop + image_id)
        image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(LoadLabelsForCrop + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return image, mask

def make_cropper(image_ids, SaveCropImages, SaveCropMasks, shift = 0):
    ix = shift

    for ax_index, name in tqdm(enumerate(image_ids),total=len(image_ids)):
        
#         if name == '252.tiff':
#             print(ix)    
        image, mask = read_image_labels(name) 
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


def read_masks(image_id):
     
        mask = cv2.imread(LoadMasksForWeight + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return mask
    
def read_images(image_id):
     
        mask = cv2.imread(LoadImgsForWeight + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return mask
    
def read_masks_done(image_id):
     
        mask = cv2.imread(SaveWeightMasks + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return mask
    
