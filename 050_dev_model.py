# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 Luca Clissa, Marco Dalla, Roberto Morelli
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Created on Tue May  7 10:42:13 2019
@author: Roberto Morelli
"""

import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Add, Flatten, Conv2D, MaxPooling2D\
, AveragePooling2D, BatchNormalization,Conv2D, UpSampling2D, Lambda, ZeroPadding2D
from keras.models import Model, load_model
from keras.layers import Input, SpatialDropout2D, SeparableConv2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import initializers, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import optimizers
import random
import argparse
import pathlib
from keras.utils import multi_gpu_model
import subprocess
from subprocess import PIPE

# from importlib import import_module
# plots = import_module('051_plotscalars')
# tb = import_module('050_tensorboard')
# lrate = import_module('043_learning_rate')

from config import *

# tot_img_after_aug = count_files_in_directory(0, dir_list = ALL_IMAGES)
#
# BATCH_SIZE = 4
# VALID_BATCH_SIZE = 4
# # seed1=1
# seed1 = 333
# random.seed(seed1)
# val_percentage = 0.25
# valid_number = int(tot_img_after_aug*val_percentage)
# train_number = tot_img_after_aug - valid_number
#
# IMG_CHANNELS = 3
# IMG_HEIGHT = 512
# IMG_WIDTH = 512

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred[:,:,:,0:1]> t)
        score, up_opt = tf.metrics.mean_iou(y_true[:,:,:,0:1], y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true[:,:,:,0:1])
    y_pred_f = K.flatten(y_pred[:,:,:,0:1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        weight_vector = y_true[:,:,:,0:1]*one_weight + (1. - y_true[:,:,:,0:1])*zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def create_weighted_binary_crossentropy_overcrowding(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        class_weight_vector = y_true[:,:,:,0:1] * one_weight + (1. - y_true[:,:,:,0:1]) * zero_weight

        weight_vector = class_weight_vector * y_true[:,:,:,1:2]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def rgb2hsv(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    image = cv2.merge([h*1./358, s*1./255, v*1./255])
    return image

def imageGenerator(color_mode = 'rgb'):

    print('path for image generatore {}'.format(ALL_IMAGES))

    image_datagen = ImageDataGenerator(rescale=1./255,validation_split = val_percentage)
                                       #preprocessing_function=rgb2hsv)
    mask_datagen = ImageDataGenerator(rescale=1./255, validation_split = val_percentage, dtype=bool)

    train_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'training')
    train_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'training',
                                                            color_mode = color_mode)

    valid_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'validation')
    valid_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'validation',
                                                            color_mode = color_mode)

    train_generator = zip(train_image_generator,train_mask_generator)
    valid_generator = zip(valid_image_generator,valid_mask_generator)
    return train_generator, valid_generator


def c_resunet(train_generator, valid_generator, weights , class_0_w , class_1_w,
             n , activation, model_name, compiler, gpus, cluster):

    inputs = Input((None, None, 3))

    c0 = Conv2D(1, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)
    c0 = BatchNormalization()(c0)
    c0 = Activation(activation)(c0)

    c1 = Conv2D(4*n, (3, 3), padding='same',  kernel_initializer='he_normal')(c0)
    c1 = BatchNormalization()(c1)
    c1 = Activation(activation)(c1)

    c1 = Conv2D(4*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    p1 = MaxPooling2D((2, 2))(c1)
    ##########################################
    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)
    # X_shortcut = BatchNormalization()(X_shortcut)

    c2 = BatchNormalization()(p1)
    c2 = Activation(activation)(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation(activation)(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
    # X_shortcut = BatchNormalization()(X_shortcut)

    c3 = BatchNormalization()(p2)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)
    ###################Bridge#######################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3)
    # X_shortcut = BatchNormalization()(X_shortcut)

    c4 = BatchNormalization()(p3)
    c4 = Activation(activation)(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation(activation)(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    c4 = Add()([c4, X_shortcut])
    X_shortcut = c4

    c5 = BatchNormalization()(c4)
    c5 = Activation(activation)(c5)
    c5 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation(activation)(c5)
    c5 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])
    ###################END BRIDGE#######################
    X_shortcut = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c5)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u6 = concatenate([X_shortcut, c3])

    # u6 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(u6)
    c6 = BatchNormalization()(u6)
    c6 = Activation(activation)(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation(activation)(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])
    ################################################
    X_shortcut = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c6)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u7 = concatenate([X_shortcut, c2])

    # u7 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(u7)

    c7 = BatchNormalization()(u7)
    c7 = Activation(activation)(c7)
    c7 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation(activation)(c7)
    c7 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    X_shortcut = Conv2DTranspose(4*n, (2, 2), strides=(2, 2), padding='same') (c7)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u8 = concatenate([X_shortcut, c1])
    # u8 = Conv2D(4*n, (1, 1), padding='same', kernel_initializer='he_normal')(u8

    c8 = BatchNormalization()(u8)
    c8 = Activation(activation)(c8)
    c8 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = BatchNormalization()(c8)
    c8 = Activation(activation)(c8)
    c8 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    if weights == 'wbce':
        print('wbce no map of weights allowed')
        WeightedLoss = create_weighted_binary_crossentropy(class_0_w, class_1_w)

    elif weights == 'map_weights':
        print('map of weight allowed')
        WeightedLoss = create_weighted_binary_crossentropy_overcrowding(class_0_w, class_1_w)

    if gpus>1:
        model = multi_gpu_model(model, gpus=gpus)

    # with tf.device("/cpu:0"):
		# initialize the model
		# model = MiniGoogLeNet.build(width=32, height=32, depth=3,
		# 	classes=10)
    if compiler == 'Adam':
        Adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.003, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])

    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1,
                             mode='auto', cooldown=0, min_lr=9e-8)

    callbacks = [checkpointer, earlystopping, ReduceLR]

    if cluster:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200,
                                      use_multiprocessing=True,
                                      workers=8*gpus
                                      )
    else:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200)


def resunet(train_generator, valid_generator, weights , class_0_w , class_1_w,
             n , activation, model_name, compiler, gpus, cluster):

    inputs = Input((None, None, 3))

    c1 = Conv2D(4*n, (3, 3), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation(activation)(c1)

    c1 = Conv2D(4*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    p1 = MaxPooling2D((2, 2))(c1)
    ##########################################
    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)
    # X_shortcut = BatchNormalization()(X_shortcut)

    c2 = BatchNormalization()(p1)
    c2 = Activation(activation)(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation(activation)(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
    # X_shortcut = BatchNormalization()(X_shortcut)

    c3 = BatchNormalization()(p2)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)
    ###################Bridge#######################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3)
    # X_shortcut = BatchNormalization()(X_shortcut)

    c4 = BatchNormalization()(p3)
    c4 = Activation(activation)(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation(activation)(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    c4 = Add()([c4, X_shortcut])

    ###################END BRIDGE#######################
    X_shortcut = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c4)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u5 = concatenate([X_shortcut, c3])

    # u6 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(u6)
    c5 = BatchNormalization()(u5)
    c5 = Activation(activation)(c5)
    c5 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation(activation)(c5)
    c5 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])
    ################################################
    X_shortcut = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c5)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u6 = concatenate([X_shortcut, c2])

    # u7 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(u7)

    c6 = BatchNormalization()(u6)
    c6 = Activation(activation)(c6)
    c6 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation(activation)(c6)
    c6 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    X_shortcut = Conv2DTranspose(4*n, (2, 2), strides=(2, 2), padding='same') (c6)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u7 = concatenate([X_shortcut, c1])
    # u8 = Conv2D(4*n, (1, 1), padding='same', kernel_initializer='he_normal')(u8

    c7 = BatchNormalization()(u7)
    c7 = Activation(activation)(c7)
    c7 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation(activation)(c7)
    c7 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    if weights == 'wbce':
        WeightedLoss = create_weighted_binary_crossentropy(class_0_w, class_1_w)

    elif weights == 'map_weights':
        WeightedLoss = create_weighted_binary_crossentropy_overcrowding(class_0_w, class_1_w)

    if gpus>1:
        model = multi_gpu_model(model, gpus=gpus)

    # with tf.device("/cpu:0"):
		# initialize the model
		# model = MiniGoogLeNet.build(width=32, height=32, depth=3,
		# 	classes=10)
    if compiler == 'Adam':
        Adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.003, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])

    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1,
                             mode='auto', cooldown=0, min_lr=9e-8)

    callbacks = [checkpointer, earlystopping, ReduceLR]

    if cluster:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200,
                                      use_multiprocessing=True,
                                      workers=8*gpus
                                      )
    else:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200)
### NOT TRAINED in the paper #####
### NOT TRAINED in the paper #####
### NOT TRAINED in the paper #####
def shorter_unet(train_generator, valid_generator, weights , class_0_w , class_1_w,
             n , model_name, compiler, gpus, cluster):


    inputs = Input((None, None, 3))

    c1 = Conv2D(8*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(8*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    # c4 = Conv2D(64*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    # c4 = BatchNormalization()(c4)
    # c4 = Conv2D(64*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    # c4 = BatchNormalization()(c4)
    # p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(64*n, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64*n, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = BatchNormalization()(u6)
    u6 = concatenate([u6, c3])
    c6 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c2])
    c7 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c1])
    c8 = Conv2D(8*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(8*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    if weights == 'wbce':
        WeightedLoss = create_weighted_binary_crossentropy(class_0_w, class_1_w)

    elif weights == 'map_weights':
        WeightedLoss = create_weighted_binary_crossentropy_overcrowding(class_0_w, class_1_w)

    if gpus>1:
        model = multi_gpu_model(model, gpus=gpus)

    # with tf.device("/cpu:0"):
		# initialize the model
		# model = MiniGoogLeNet.build(width=32, height=32, depth=3,
		# 	classes=10)
    if compiler == 'Adam':
        Adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.003, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])

    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1,
                             mode='auto', cooldown=0, min_lr=9e-8)

    callbacks = [checkpointer, earlystopping, ReduceLR]

    if cluster:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200,
                                      use_multiprocessing=True,
                                      workers=8*gpus
                                      )
    else:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200)

def unet(train_generator, valid_generator, weights , class_0_w , class_1_w,
             n , model_name, compiler, gpus, cluster):

    inputs = Input((None, None, 3))

    c1 = Conv2D(8*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    # c1 = BatchNormalization()(c1)
    c1 = Conv2D(8*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
    # c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    # c2 = BatchNormalization()(c2)
    c2 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
    # c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    # c3 = BatchNormalization()(c3)
    c3 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
    # c3 = BatchNormalization()(c3)

    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    # c4 = BatchNormalization()(c4)
    c4 = Conv2D(64*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
    # c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128*n, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    # c5 = BatchNormalization()(c5)
    c5 = Conv2D(128*n, (5, 5), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
    # c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(64*n, (2, 2), strides=(2, 2), padding='same') (c5)
    # u6 = BatchNormalization()(u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
    # c6 = BatchNormalization()(c6)
    c6 = Conv2D(64*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
    # c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c6)
    # u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    # c7 = BatchNormalization()(c7)
    c7 = Conv2D(32*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    # c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c7)
    # u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    # c8 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    # c8 = Conv2D(16*n, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    if weights == 'wbce':
        WeightedLoss = create_weighted_binary_crossentropy(class_0_w, class_1_w)

    elif weights == 'map_weights':
        WeightedLoss = create_weighted_binary_crossentropy_overcrowding(class_0_w, class_1_w)

    if gpus>1:
        model = multi_gpu_model(model, gpus=gpus)

    # with tf.device("/cpu:0"):
		# initialize the model
		# model = MiniGoogLeNet.build(width=32, height=32, depth=3,
		# 	classes=10)
    if compiler == 'Adam':
        Adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.003, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])

    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1,
                             mode='auto', cooldown=0, min_lr=9e-8)

    callbacks = [checkpointer, earlystopping, ReduceLR]

    if cluster:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200,
                                      use_multiprocessing=True,
                                      workers=8*gpus
                                      )
    else:

        results = model.fit_generator(train_generator,
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator,
                                      validation_steps=valid_number/VALID_BATCH_SIZE,
                                      callbacks=callbacks,
                                      epochs=200)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='weighting masks')

    parser.add_argument('--images', nargs="?", default = AugCropImages, help='path including images for training')
    parser.add_argument('--masks', nargs="?", default = AugCropMasks, help='path including masks for training')
    parser.add_argument('--model_name',type=str, default = 'ResUnet',  help='model name. Options are [c_resunet, resunet, unet, small_unet]')
    parser.add_argument('--activation',type=str, default = 'elu',  help='activation')

    parser.add_argument('--model_results', nargs="?", default = ModelResults, help='path where to save the models')
    parser.add_argument('--compiler', nargs="?",type = str, default = 'Adam', help='Adam or SGD')
    parser.add_argument("--gpus", type=int, default=1, help="# of GPUs to use for training")

    parser.add_argument('--batch_size', nargs="?", type = int, default = 4,  help='batch size')
    parser.add_argument('--val_split', nargs="?", type = float, default = 0.3,  help='val_split')
    parser.add_argument('--x_crop_size', nargs="?", type = int, default = 512,  help='x_crop_size')
    parser.add_argument('--y_crop_size', nargs="?", type = int, default = 512,  help='y_crop_size')
    parser.add_argument('--img_channels', nargs="?", type = int, default = 3,  help='image channels')

    parser.add_argument('--weights', nargs="?", type = str, default = 'map_weights',  help='wbce for weighted bce or map_weights for overcrowding map')
    parser.add_argument('--class_weights', nargs="?", type = list, default = [1, 1.5],  help='class 0 and class 1 weights')
    parser.add_argument('--n', nargs="?", type = int, default = 4,  help='number of layer multiplier factor, starting from n0 = 4')

    args = parser.parse_args()

    try:
        o = subprocess.run(["hostname"],  capture_output=True)
        o = o.stdout
        o = o.strip().decode('utf-8')
    except:
        o = subprocess.run(["hostname"],  stdout=PIPE, stderr=PIPE)
        o = o.stdout
        o = o.strip().decode('utf-8')

    if 'cnaf' in o:
        cluster=True
    else:
        cluster=False

    # cluster = True

    if args.images == 'AugCropImagesBasic':
        if not cluster:
            images_path = AugCropImagesBasic
            masks_path = AugCropMasksBasic
        else:
            images_path = AugCropImagesBasicSplitted
            masks_path = AugCropMasksBasicSplitted

        tot_img_after_aug = len(os.listdir(AugCropImagesBasic))

    else:
        if not cluster:
            images_path = AugCropImages
            masks_path = AugCropMasks
        else:
            images_path = AugCropImagesSplitted
            masks_path = AugCropMasksSplitted

        tot_img_after_aug = len(os.listdir(AugCropImages))

    # tot_img_after_aug = len(os.listdir(images_path))
    BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.batch_size
    # seed1=1
    seed1 = 333
    # random.seed(seed1)
    val_percentage = args.val_split
    valid_number = int(tot_img_after_aug*val_percentage)
    train_number = tot_img_after_aug - valid_number

    IMG_WIDTH = args.x_crop_size
    IMG_HEIGHT = args.y_crop_size
    IMG_CHANNELS = args.img_channels

    ALL_IMAGES = pathlib.Path(images_path)
    ALL_MASKS = pathlib.Path(masks_path)
    MODEL_CHECKPOINTS = pathlib.Path(args.model_results)

    if IMG_CHANNELS == 3:
        color_mode = 'rgb'
    elif IMG_CHANNELS == 1:
        color_mode = 'grayscale'

    #batch size, img size # model, #n, #map_weight,#name model# color_mode
    train_generator, valid_generator = imageGenerator(color_mode='rgb')

    if 'resunet' in args.model_name.lower():
        print('ResUnet')
        resunet(train_generator,valid_generator, weights = args.weights, class_0_w = args.class_weights[0]
                , class_1_w = args.class_weights[1], model_name = args.model_name + '.h5', n = args.n, activation=args.activation,
                compiler = args.compiler, gpus = args.gpus, cluster=cluster)

    elif 'c_resunet' in args.model_name.lower():
        print('c-ResUnet')
        c_resunet(train_generator,valid_generator, weights = args.weights, class_0_w = args.class_weights[0]
                , class_1_w = args.class_weights[1], model_name = args.model_name + '.h5', n = args.n, activation = args.activation,
                compiler = args.compiler, gpus = args.gpus, cluster=cluster)


    elif 'unet' in args.model_name.lower(): #n should be 8
        print('Unet')

    if args.n==4:
        print('changing default m.arg=4 to n.args=8, if you want to use n.arg=4 instead, use model_name=small_unet')
        args.n=8
        unet(train_generator,valid_generator, weights = args.weights, class_0_w = args.class_weights[0]
                , class_1_w = args.class_weights[1], model_name = args.model_name + '.h5', n = args.n,
                compiler = args.compiler, gpus = args.gpus, cluster=cluster)

    elif 'small_unet' in args.model_name.lower():
        print('small Unet')
        unet(train_generator,valid_generator, weights = args.weights, class_0_w = args.class_weights[0]
                , class_1_w = args.class_weights[1], model_name = args.model_name + '.h5', n = args.n,
                compiler = args.compiler, gpus = args.gpus, cluster=cluster)


