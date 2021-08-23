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
