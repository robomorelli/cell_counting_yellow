# cell_counting_yellow
This is the code implementation of the work described here [link to paper: *Automatic Cell Counting in Flourescent Microscopy Using Deep Learning*](https://arxiv.org/abs/2103.01141)]

# Data Overview
The dataset used in the paper ...

# How to use

### Dependencies

Check the requirements.txt file for the dependencies

### Dataset

You should copy zip of dataset (Fluorescent Neuronal Cells) inside the **main directory** of the repository

Once the zip file is here:

1. Create a folder named DATASET
2. Move the Fluorescent_Neuronal_Cells.zip into the DATASET folder
3. Unzip the Fluorescent_Neuronal_Cells.zip

### Code

1. 010_join_images: Join the images coming from the automatic and the manual segmenation (see description in the paper [link to paper:](https://arxiv.org/abs/2103.01141)]) into one unique folder (/DATASET/all_images/images for the images and /DATASET/all_masks/masks for the masks). This script also correct the masks with more than two grayscales values (0 for the background and 255 for the pixels of the cells).

Run in this way: python 010_join_images.py

2. 011_load_images.py: Split the dataset into train-validation and test.
- To run with the same splits used in the paper: python 011_load_images.py
- If you want to use a different split run whit the additional arguments: python 011_load_images.py --random_test_set True
If you already run this script previusoly take care to add  **--start_from_zero True** flag to remove the previous images in the destination folder





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
