#!/bin/bash

export BASE=/usr/local/cuda-5.5/
export PATH=$BASE/bin:$PATH 
export C_INCLUDE_PATH=$BASE/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$BASE/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$BASE/lib:$BASE/lib64:/usr/local/cuda-5.0/lib64/:$LD_LIBRARY_PATH

export XDG_RUNTIME_DIR=""
#env
#echo "------------------------------------------------"
#now your GPU executable

source /home/HPC/rmorellihpc/anaconda3/bin/activate cell

#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/010_join_images.py
#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/011_load_images.py --start_from_zero
#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/020_cropper_yellow.py --start_from_zero
#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/030_weights_maker.py --start_from_zero
#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/040_augumentation_yellow.py --start_from_zero --no_artifact_aug

#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/060_split_file.py --images AugCropImagesBasic 
python /home/HPC/lclissahpc/workspace/cell_counting_yellow/evaluation_utils.py ResUnet


##### to tf_gpu becuas with tf 1.14 in cell enviromente cluster is not working well (very slow)
#####Remember to comment (10 lines above) the activation of cell env

#source /home/HPC/rmorellihpc/anaconda3/bin/activate tf_gpu # we need to switch
#python /storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow/050_dev_model.py --batch_size 12 --n 8 --model_name UnetOriginalNwm --weights wbce #--images AugCropImagesBasic

# n: Unet=2 (start from 16), ResUnet and ResUnetBasic = 4 (start from 16), UnetOriginal = 8 (start from 64)

#####batch_size (12 for each gpu >>> 12 with only 1 gpu, 48 with 4 gpus)
