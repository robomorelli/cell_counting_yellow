# Cell counting

This repository is the code implementation of the work [*Automatic Cell Counting in Flourescent Microscopy Using Deep
Learning*](https://arxiv.org/abs/2103.01141)

## Installation

### pip

```commandline
git clone git@github.com:robomorelli/cell_counting_yellow.git
cd cell_counting_yellow
pip install requirements.txt
```

### docker (TO DO)

-Build the image from dockerfile docker build -t .

-Run the image:
docker run --rm -it -p 8888:8888 -v ${PWD}/DATASET:app/DATASET cell

Run initialize a shell session inside the container where it is possible to run the scripts and a jupyter session with
the command:
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

connect following the instruction displayed on the terminal

run jupyter directly from docker:
docker run -it -p 8888:8888 -v ${PWD}/DATASET:/app/DATASET cell -c "jupyter notebook --port=8888 --no-browser
--ip=0.0.0.0 --allow-root"

connect following the instruction displayed on the terminal

### Dataset

The data consist of 283 high-resolution pictures (1600x1200 pixels) of mice brain slices acquired through a fluorescence
microscope. The final goal is to individuate and count neurons highlighted in the pictures by means of a marker, so to
assess the result of a biological experiment. For more information, please refer to [1]. The corresponding ground-truth
labels were generated through a hybrid approach involving semi-automatic and manual semantic segmentation. The result
consists of black (0) and white (255) images having pixel-level annotations of where the neurons are located. Possible
applications include but are not limited to *semantic segmentation*, *object detection*
and *object counting*.

After cloning the repository, you can download the data from the **Fluorescent Neuronal Cells** [archive]().

Once the zip file of the dataset is downloaded:

1. Create a new folder named `DATASET` inside the root of the repository
2. Unzip its content into the `DATASET` folder. Final structure should be:

```commandline
cell_counting_yellow/
├── DATASET
│   ├── all_images
│   ├── all_masks
│   └── README.md
├── model_results
├── notebooks
└── results
```

3. Run the script `010_split_data.py` to get the exact same train/val VS test split used in the paper. Final structure
   should be:

```commandline
cell_counting_yellow/
├── DATASET
│   ├── all_images
│   ├── all_masks
│   ├── README.md
│   ├── test
│   ├── train_val
│   └── test_split_map.csv
├── model_results
├── notebooks
└── results
```
Note: `010_join_images.py` and `011_load_images.py` refer to a previous version of the dataset and are replaced by `010_split_data.py`.

### Customization

All the relative paths and macro-variables are defined in the `config.py` module. Edit this file for custom
configurations.

## Usage

### Preliminary remark
1. All the scripts are provided with an **help description** to invoke as follow:
python <script name> -h
2. To reproduce the paper results, all the scripts should be launched without additional arguments (default mode)
3. Each time you run again one of the preprocessing scripts going from 011 to 040, take care to remove the images-masks produced at the previous run setting the **--start_from_zero True** flag

Utilities are defined in python scripts used as modules. These are imported in the pipeline scripts (those having names
starting with numbers) in order to perform the various steps of the analysis (described in the following).

### Pre-processing

- `020_cropper_yellow.py`: Crop the images and masks into smaller patches for the training of the network. The default
  size of the crops will be the same used in the referenced paper (512x512). To make crops with different features check
  the help description of the script.

- `030_weights_maker.py`: Get the weight maps as described in the paper. No additional arguments are required to
  reproduce the same masks of the paper. Some default parameters (like the sigma factor) can be changed following the
  help decription.

```commandline
  python 030_weight_maker.py --normalize True --continue_after_normalization True
```

- `040_augumentation_yellow.py`: Run the augmentation to increase the number of the training-validation images. Use the
  help to define the augmentation factor both for the images segmented automatically and those segmented manually. Also,
  it is possible to select or not a strategy for the artifact augmentation. As usual, running without additional
  arguments reproduce the same augumentation pipeline adopted in the paper. It is worth to remember that the
  augumentation will produce different images from those used in the paper due to the random nature of the augmentation
  process.

### Training

- `050_dev_model.py`: Train a convolutional neural network on the Fluorescent Neuronal Cells dataset. Without additional
  arguments the ResUnet described in the paper as c-Resunet is trained. To train another architecture use
  the `--model_name` argument. Options are: [c_resunet, resunet, unet, small_unet]. Check the help to modify other
  training parameters like the classes' loss weight: defaults are 1.5 for white pixels (cell class) and 1 for black
  pixel (background).

- `060_split_file.py`: Utility to split the train-val set in subfolder containing 1000 images each. This procedure is
  *suggested only for training on a cluster* because the I/O operations on smaller files are preferred in this
  case.

### Evaluation

In order to assess the performance on a model it is possible to exploit the script `evaluate_model_test.py`

```commandline
python evaluate_model.py --mode [eval|test|test_code] --threshold [knee|best|grid] --out_folder results/ <model_name>
```

**Note:** the model name must be specified without file extension, e.g. *c-ResUnet* for the weights saved in 
*c-ResUnet.h5*


To reproduce the results of the paper run:

```commandline
# run grid search on the train/val split (full-size images only)
python evaluate_model.py --mode eval --threshold grid --out_folder results/ <model_name>

# evaluate the performance on the test set with the optimal threshold according to the kneedle method
python evaluate_model.py --mode test --threshold knee --out_folder results/ <model_name>
```

### Notebooks

Some useful notebooks are provided to visualize the output of the training and compare different models:

- `Compare_models.ipynb`: compares the heatmaps and the object detection abilities of multiple models side-by-side
- `Visualize_results.ipynb`: shows the effect of the post-processing pipeline for a given model

### References

    [[1]](https://www.nature.com/articles/s41598-019-51841-2) Hitrec, T., Luppi, M., Bastianini, S., Squarcio, F.,
    Berteotti, C., Martire, V.L., Martelli, D., Occhinegro, A., Tupone, D., Zoccoli, G. and Amici, R., 2019. Neural control
    of fasting-induced torpor in mice. Scientific reports, 9(1), pp.1-12.
  
