# Cell counting
This is the code implementation of the work described here: [link to paper: *Automatic Cell Counting in Flourescent Microscopy Using Deep Learning*](https://arxiv.org/abs/2103.01141)]

# Data Overview
The dataset used in the paper ...

# How to use

## Dependencies

Check the requirements.txt file for the dependencies

A docker recipe is provided to build an enviroment to run this project.

## Dataset

You should copy zip of dataset (Fluorescent Neuronal Cells) inside the **main directory** of the repository

Once the zip file is here:

1. Create a new folder named DATASET
2. Move the Fluorescent_Neuronal_Cells.zip into the DATASET folder
3. Unzip the Fluorescent_Neuronal_Cells.zip

## Code

### Preliminary remark
1. All the scripts are provided with an **help description** to invoke as follow:
python <script name> -h
2. To reproduce the paper results, all the scripts should be launched without additional arguments (default mode)
3. Each time you run again one of the preprocessing scripts going from 011 to 040, take care to remove the images-masks produced at the previous run setting the **--start_from_zero True** flag
  

### Scripts description
#### Configuration 
  
1. **config.py**: Define all the default path where images and mask are saved and preprocessed. Also, some pre-defined variables like the height and the widht of the images contained in the Fluorescent Neuronal Cells dataset.
  
#### Preprocessing utils
2. **utils.py**: Contain all the function imported during the different preprocessing steps. These functions are used mainly to define the following utilities:
  - I/O of the images
  - Images cropping
  - Mask weighting
  - Images Augmentation
  
3. **augmentation_utils.py**: Further utils function used to manipulate the images during the augumentation process.

#### Preprocessing 
  
4. **010_join_images.py**: Join the images and the relative masks coming from the automatic and the manual segmenation (see data description in the paper [link to paper:](https://arxiv.org/abs/2103.01141)]) into one unique folder: /DATASET/all_images/images for the images and /DATASET/all_masks/masks for the masks. This script also correct the masks images with more than two grayscales values (0 for the background and 255 for the pixels of the cells).

5. **011_load_images.py**: Split the dataset into train-validation and test. Without any additional arguments the split will be the same used in the paper. For example, if you want to use a different split run whit the additional arguments: 
python 011_load_images.py --random_test_set True
  
If you already run this script previusly take care to add  **--start_from_zero True** flag to remove the previous images in the destination folder

6. **020_cropper_yellow.py**: Crop the images and masks to make smaller picture for the training of the network. The default size of the crops will be the same used in the referenced paper. To make crops of different features check the help description of the script

If you already run this script previusly take care to add  **--start_from_zero True** flag to remove the previous images in the destination folder
  
7. **030_weights_maker.py**: Desing the weighted maps as described in the paper. No additional arguments to reproduce the same  
masks of the paper. 
  
**NOTE**: If random split was used running the **011_load_images.py** script, it is needed to find a new maximum value to normalize the weighted maps using the following instruction:
  
  python 030_weight_maker.py --normalize True --continue_after_normalization True
  
  or in a 2-step process:
  
  1. python 030_weight_maker.py --normalize True
  2. python 030_weight_maker.py --resume_after_normalization True
  
If you already run this script previusly take care to add  **--start_from_zero True** flag to remove the previous images in the destination folder

8. **040_augumentation_yellow.py**: Augmentation process to increase the number of training-validation images. Use the help to define the augmentation factor both for the images segmented automatically and those segmented manually. Also define to adopt or not a strategy for the artifact augmentation. No additional argument aim to reproduce the augumentation pipeline described in the paper. It is worth to remember that the augumentation will produce different images from those used in the paper due to the random nature of the augmentation process.
  
If you already run this script previusly take care to add  **--start_from_zero True** flag to remove the previous images in the destination folder
  
### Training

9. **050_dev_model.py**: Train a convolutional neural network on the Fluorescent Neuronal Cells dataset. Without additional arguments the ResUnet described in the paper as c-Resunet is trained. To train another architecture use the **--model_name** argument. Options are: [ResUnet, ResUnetBasic, Unet, UnetOriginal]. Check the help to modify other training parameters like the classes' loss weight. Default are 1.5 for white pixels (cell class) and 1 for black pixel (no cell class).

DOCKER instruction

-Build the image from dockerfile
docker build -t .

-Run the image:
docker run --rm -it -p 8888:8888 -v ${PWD}/DATASET:app/DATASET cell

Run initialize a shell session inside the container where it is possible to run the scripts and a jupyter session with the command:
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

connect following the instruction displayed on the terminal

run jupyter directly from docker:
docker run -it -p 8888:8888 -v ${PWD}/DATASET:/app/DATASET cell -c "jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"

connect following the instruction displayed on the terminal
  
