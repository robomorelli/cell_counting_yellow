#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Luca Clissa, Marco Dalla, Roberto Morelli
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
Created on Sat Jul 20 13:00:41 2019

@author: Luca Clissa
"""

from math import hypot
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

from evaluation_utils import mask_post_processing

# import matplotlib
# matplotlib.use('ps')
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Helvetica']
plt.rcParams["font.size"] = "16"


def plot_predicted_heatmaps(model, test_img_path, test_masks_path, head=None, suptitle=True):
    '''Plot original image with true objects and the predicted heatmap.
    
    Keyword arguments:
    model -- model object
    test_img_path -- path where the images to be plotted are stored
    test_masks_path -- path where the relative masks are stored
    head -- either None or the number of plots to display
   
    Return: None.
    '''
    counter = 0
    for idx, img_path in enumerate(test_img_path.iterdir()):
        if img_path.name != '278.tiff':
            continue

        if not img_path.name.startswith("aug_"):

            img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            mask_path = test_masks_path / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # predictions
            img_rgb = np.expand_dims(img_rgb, 0)
            pred_mask_rgb = np.squeeze(model.predict(img_rgb / 255.))

            if img_path.name == '275.tiff':
                img_rgb = np.squeeze(img_rgb)[400:850, 150:500]
                mask = mask[400:850, 150:500]
                print(pred_mask_rgb.shape)
                print(pred_mask_rgb.shape)
                pred_mask_rgb = pred_mask_rgb[400:850, 150:500]
            if img_path.name == '278.tiff':
                img_rgb = np.squeeze(img_rgb)[200: 800, 930: 1420]
                mask = mask[200: 800, 930: 1420]
                pred_mask_rgb = pred_mask_rgb[200: 800, 930: 1420]
            # img_rgb = np.squeeze(img_rgb)
            # pred_mask_rgb = np.flipud(pred_mask_rgb)

            # plot predictions
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            if suptitle:
                fig.suptitle(img_path.name)

            # original image + true objects
                axes[0].imshow(np.squeeze(img_rgb), cmap=plt.cm.RdBu, aspect = "auto")
            axes[0].contour(mask, [0.5], linewidths=1.2, colors='w')
            axes[0].set_title('Original image and mask')

            # RGB prediction
            im = axes[1].pcolormesh(np.flipud(pred_mask_rgb), cmap='jet')
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            axes[1].set_title('Predicted heatmap')
            if img_path.name == '278.tiff':
                outpath = Path('/home/luca/PycharmProjects/cell_counting_yellow/results/figures/methods')
                outname = f"fig4:orig+heatmap:{img_path.name.split('.')[0]}.png"
                print('Saving at: ', str(outpath / outname))
                plt.subplots_adjust(
                    # left=0.15,
                    # bottom=0,
                    right=0.95,
                    # top=0,
                    wspace=0.05,
                    # hspace=0
                )
                for x in axes:
                    x.axis('off')
                plt.savefig(outpath / outname, bbox_inches='tight', pad_inches=0)
            plt.show()
            counter += 1
            if counter == head:
                break


def compare_heatmaps(models_dict, test_img_path, test_masks_path, head=None):
    """Plot comparisons of all models in models_dict with bounding boxes for TP, FP, and FN.

    :param models_dict: dictionary with structure {model name: model object}
    :param test_img_path: path where original images are stored
    :param test_masks_path: path where corresponding masks are stored
    :param head: either None or the number of plots to display

    :return: None
    """
    from matplotlib import pyplot as plt

    counter = 0
    for idx, img_path in enumerate(test_img_path.iterdir()):

        if not img_path.name.startswith("aug_"):
            # fig, axes = plt.subplots(int(np.ceil(len(models_dict) / 2)), 2, figsize=(20, 6))
            fig, axes = plt.subplots(1, len(models_dict) + 1, figsize=(20, 6))
            # fig.suptitle(img_path.name, fontsize=22)
            print("\033[31m" + img_path.name)

            img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            mask_path = test_masks_path / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # original image + true objects
            axes[0].imshow(img_rgb, cmap=plt.cm.RdBu)
            axes[0].contour(mask, [0.5], linewidths=1.2, colors='w')
            axes[0].set_title('Original image and mask')

            # predictions
            img_rgb = np.expand_dims(img_rgb, 0)
            for idx, model_item in enumerate(models_dict.items()):
                model_name, model = model_item[0], model_item[1]

                pred_mask_rgb = np.squeeze(model.predict(img_rgb / 255.))
                im = axes[idx + 1].pcolormesh(np.flipud(pred_mask_rgb), cmap='jet')
                divider = make_axes_locatable(axes[idx + 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                axes[idx + 1].set_title(r"$\bf{{{}}}$".format(model_name.replace('_', '-')))
            plt.show()
            counter += 1
            if counter == head:
                break
    return (None)


def plot_postprocessing_effect(model, test_img_path, threshold, head=None, suptitle=True):
    '''Plot original image with true objects and the predicted heatmap.

    Keyword arguments:
    model -- model object
    test_img_path -- path where the images to be plotted are stored
    test_masks_path -- path where the relative masks are stored
    head -- either None or the number of plots to display

    Return: None.
    '''
    from scipy import ndimage
    import matplotlib

    qualitative_cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                         'Dark2', 'Set1', 'Set2', 'Set3',
                         'tab10', 'tab20', 'tab20b', 'tab20c']
    colors = [c for cmap_name in qualitative_cmaps for c in matplotlib.pyplot.get_cmap(cmap_name).colors]
    colors.insert(0, (0, 0, 0))
    vmin = 0
    vmax = len(colors)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('black+qualitative', colors, N=len(colors))

    struct = np.ones((3, 3))

    counter = 0
    for idx, img_path in enumerate(test_img_path.iterdir()):
        if img_path.name != '278.tiff':
            continue

        if not img_path.name.startswith("aug_"):

            img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # predictions
            img_rgb = np.expand_dims(img_rgb, 0)
            pred_mask_rgb = np.squeeze(model.predict(img_rgb / 255.))
            thresh_image = np.squeeze((pred_mask_rgb > threshold).astype('uint8'))
            post_processed_image = mask_post_processing(thresh_image)

            if img_path.name == '275.tiff':
                thresh_image = thresh_image[400:850, 150:500]
                post_processed_image = post_processed_image[400:850, 150:500]

            if img_path.name == '278.tiff':
                thresh_image = thresh_image[200: 800, 930: 1420]
                post_processed_image = post_processed_image[200: 800, 930: 1420]


            # plot predictions
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            if suptitle:
                fig.suptitle(img_path.name)

            # Thresholded mask
            blobs, number_of_blobs = ndimage.label(np.squeeze(thresh_image))  # , struct)
            axes[0].imshow(blobs, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            axes[0].set_title('Thresholded mask')

            # Post-processed mask
            blobs, number_of_blobs = ndimage.label(np.squeeze(post_processed_image))  # , struct)
            axes[1].imshow(blobs, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            axes[1].set_title('Post-processed mask')

            if img_path.name == '278.tiff':
                outpath = Path('/home/luca/PycharmProjects/cell_counting_yellow/results/figures/methods')
                outname = f"fig4:thresh+post_proc:{img_path.name.split('.')[0]}.png"
                print('Saving at: ', str(outpath / outname))
                plt.subplots_adjust(
                    # left=0.15,
                    # bottom=0,
                    right=0.95,
                    # top=0,
                    wspace=0.05,
                    # hspace=0
                )
                for x in axes:
                    x.axis('off')
                plt.savefig(outpath / outname, bbox_inches='tight', pad_inches=0)
            plt.show()

            counter += 1
            if counter == head:
                break


def plot_predicted_mask(model, test_img_path, test_masks_path, threshold, post_processing=True, head=None,
                        suptitle=True):
    '''Plot original image with true objects and the predicted heatmap.
    
    Keyword arguments:
    model -- model object
    test_img_path -- path where the images to be plotted are stored
    test_masks_path -- path where the relative masks are stored
    threshold -- cutoff for thresholding predicted heatmap
    head -- either None or the number of plots to display
    
    Return: None.
    '''
    counter = 0
    for idx, img_path in enumerate(test_img_path.iterdir()):

        if not img_path.name.startswith("aug_"):

            img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            mask_path = test_masks_path / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # predictions
            img_rgb = np.expand_dims(img_rgb, 0)
            pred_mask_rgb = np.squeeze(model.predict(img_rgb / 255.))
            thresh_image = np.squeeze((pred_mask_rgb > threshold).astype('uint8'))

            # apply post-processing
            if post_processing:
                thresh_image = mask_post_processing(thresh_image)

            plot_predictions_with_metrics(np.squeeze(img_rgb), img_path.name,
                                          thresh_image, mask, suptitle)
            counter += 1
            if counter == head:
                break

    return (None)


def draw_bounding_boxes_with_metrics(img, pred_mask, mask):
    '''Add bounding boxes for TP, FP, and FN to the original image.

    Keyword arguments:
    img -- array of the original image
    pred_mask -- array of the predicted mask
    mask -- groundtruth mask

    Return: img, tp, fp, fn, ae, pred_rgb, true_count.
    '''
    pred_mask = pred_mask.astype("bool")
    pred_label, pred_rgb = ndimage.label(pred_mask)
    pred_objs = ndimage.find_objects(pred_label)

    # extract target objects and counts
    true_label, true_count = ndimage.label(mask)
    true_objs = ndimage.find_objects(true_label)

    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                             (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

    # compute centers of target objects
    targ_center = []
    for ob in true_objs:
        targ_center.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                            (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

    # associate matching objects, true positives
    tp = 0
    tp_objs = []

    for pred_idx, pred_obj in enumerate(pred_objs):

        min_dist = 31  # 1.5-cells distance is the maximum accepted
        TP_flag = 0

        for targ_idx, targ_obj in enumerate(true_objs):

            dist = hypot(pred_centers[pred_idx][0] - targ_center[targ_idx][0],
                         pred_centers[pred_idx][1] - targ_center[targ_idx][1])

            if dist < min_dist:
                TP_flag = 1
                min_dist = dist
                index_targ = targ_idx
                index_pred = pred_idx

        if TP_flag == 1:
            tp += 1
            TP_flag = 0

            cv2.rectangle(img, (pred_objs[index_pred][1].start - 10, pred_objs[index_pred][0].start - 10),
                          (pred_objs[index_pred][1].stop + 10, pred_objs[index_pred][0].stop + 10), (0, 255, 0), 3)

            tp_objs.append(pred_objs[index_pred])
            targ_center.pop(index_targ)
            true_objs.pop(index_targ)

    # derive false negatives and false positives
    fp = 0
    for pred_obj in pred_objs:
        if pred_obj not in tp_objs:
            cv2.rectangle(img, (pred_obj[1].start - 10, pred_obj[0].start - 10),
                          (pred_obj[1].stop + 10, pred_obj[0].stop + 10), (255, 0, 0), 3)
            fp += 1

    fn = 0
    for targ_obj in true_objs:
        cv2.rectangle(img, (targ_obj[1].start - 10, targ_obj[0].start - 10),
                      (targ_obj[1].stop + 10, targ_obj[0].stop + 10), (0, 0, 255), 3)
        fn += 1

    ae = abs(true_count - pred_rgb)
    return (img, tp, fp, fn, ae, pred_rgb, true_count)


def compare_predictions_with_metrics(models_dict, test_img_path, test_masks_path, threshold="best",
                                     post_processing=True, head=None):
    """Plot comparisons of all models in models_dict with bounding boxes for TP, FP, and FN.

    :param models_dict: dictionary with structure {model name: model object}
    :param test_img_path: path where original images are stored
    :param test_masks_path: path where corresponding masks are stored
    :param threshold: Cutoff for thresholding the prediction. values:
                        - 'best' (default): it takes the best F1 threshold from eval metrics
                        - list of float between 0 and 1, one per model in models_dict.
    :param post_processing: boolean for post-processing (default: True)
    :param head: either None or the number of plots to display
    :return: None
    """
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from pathlib import Path
    from evaluation_utils import mask_post_processing

    import cv2
    import matplotlib.patches as mpatches
    from kneed import KneeLocator

    # repo_path = Path("/storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow")
    repo_path = Path("/home/luca/PycharmProjects/cell_counting_yellow")

    legend_background_color = 'white'
    line_thickness = 1.5
    counter = 0
    for idx, img_path in enumerate(test_img_path.iterdir()):

        if not img_path.name.startswith("aug_"):
            # fig, axes = plt.subplots(int(np.ceil(len(models_dict) / 2)), 2, figsize=(20, 6))
            fig, axes = plt.subplots(1, len(models_dict), figsize=(20, 6))
            # fig.suptitle(img_path.name, fontsize=22)
            print("\033[31m" + img_path.name)

            mask_path = test_masks_path / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # predictions
            for idx, model_item in enumerate(models_dict.items()):
                model_name, model = model_item[0], model_item[1]
                img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

                # predictions
                img_rgb = np.expand_dims(img_rgb, 0)
                pred_mask_rgb = np.squeeze(model.predict(img_rgb / 255.))
                if threshold == 'best':
                    opt_thresh_path = repo_path / "results/eval" / 'metrics_{}.csv'.format(model_name)
                    df = pd.read_csv(opt_thresh_path, index_col='Threshold')
                    x = df.index
                    y = df.F1
                    kn = KneeLocator(x, y, curve='concave', direction='decreasing')
                    cur_threshold = kn.knee  # df.F1.idxmax()
                else:
                    cur_threshold = threshold[idx]
                thresh_image = np.squeeze((pred_mask_rgb > cur_threshold).astype('uint8'))

                # apply post-processing
                if post_processing:
                    thresh_image = mask_post_processing(thresh_image)

                img, tp, fp, fn, ae, pred_rgb, true_count = draw_bounding_boxes_with_metrics(np.squeeze(img_rgb),
                                                                                             thresh_image, mask)

                # plot
                axes[idx].imshow(img, cmap=plt.cm.RdBu)
                tp_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                                           edgecolor="green", linewidth=line_thickness)
                fp_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                                           edgecolor="red", linewidth=line_thickness)
                fn_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                                           edgecolor="blue", linewidth=line_thickness)
                ae_patch = mpatches.Circle((0.1, 0.1), 0, facecolor=legend_background_color,
                                           edgecolor=legend_background_color, linewidth=line_thickness)
                title = "Predicted count: {} - True count: {}\n".format(pred_rgb, true_count) + r"$\bf{{{}}}$".format(
                    model_name.replace('_', '-'))
                axes[idx].set_title(title, fontsize=18)

                legend = axes[idx].legend([tp_patch, fp_patch, fn_patch, ae_patch],
                                          ["True Positive: {}".format(tp), "False Positive: {}".format(fp),
                                           "False Negative: {}".format(fn), "Absolute Error: {}".format(ae)],
                                          bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=2, fontsize=14)
                frame = legend.get_frame()
                frame.set_color(legend_background_color)
            plt.show()
            counter += 1
            if counter == head:
                break
    return (None)


def plot_predictions_with_metrics(img, img_name, pred_mask, mask, suptitle=True):
    '''Plot original image with bounding boxes for TP, FP, and FN.
    
    Keyword arguments:
    img -- array of the original image
    img_name -- name of the image to print
    pred_mask -- array of the predicted mask
    mask -- groundtruth mask
    
    Return: None.
    '''

    pred_mask = pred_mask.astype("bool")

    pred_label, pred_rgb = ndimage.label(pred_mask)
    pred_objs = ndimage.find_objects(pred_label)

    # read mask and extract target objects and counts
    true_label, true_count = ndimage.label(mask)
    true_objs = ndimage.find_objects(true_label)

    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                             (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

    # compute centers of target objects
    targ_center = []
    for ob in true_objs:
        targ_center.append(((int((ob[0].stop - ob[0].start) / 2) + ob[0].start),
                            (int((ob[1].stop - ob[1].start) / 2) + ob[1].start)))

    # associate matching objects, true positives
    tp = 0
    tp_objs = []

    for pred_idx, pred_obj in enumerate(pred_objs):

        min_dist = 31  # 1.5-cells distance is the maximum accepted
        TP_flag = 0

        for targ_idx, targ_obj in enumerate(true_objs):

            dist = hypot(pred_centers[pred_idx][0] - targ_center[targ_idx][0],
                         pred_centers[pred_idx][1] - targ_center[targ_idx][1])

            if dist < min_dist:
                TP_flag = 1
                min_dist = dist
                index_targ = targ_idx
                index_pred = pred_idx

        if TP_flag == 1:
            tp += 1
            TP_flag = 0

            cv2.rectangle(img, (pred_objs[index_pred][1].start - 10, pred_objs[index_pred][0].start - 10),
                          (pred_objs[index_pred][1].stop + 10, pred_objs[index_pred][0].stop + 10), (0, 255, 0), 3)

            tp_objs.append(pred_objs[index_pred])
            targ_center.pop(index_targ)
            true_objs.pop(index_targ)

    # derive false negatives and false positives
    fp = 0
    for pred_obj in pred_objs:
        if pred_obj not in tp_objs:
            cv2.rectangle(img, (pred_obj[1].start - 10, pred_obj[0].start - 10),
                          (pred_obj[1].stop + 10, pred_obj[0].stop + 10), (255, 0, 0), 3)
            fp += 1

    fn = 0
    for targ_obj in true_objs:
        cv2.rectangle(img, (targ_obj[1].start - 10, targ_obj[0].start - 10),
                      (targ_obj[1].stop + 10, targ_obj[0].stop + 10), (0, 0, 255), 3)
        fn += 1

    # update metrics dataframe
    #    test_metrics_rgb.loc[img_name] = [tp, fp, fn, true_count, pred_rgb]

    ae = abs(true_count - pred_rgb)

    # plot
    legend_background_color = 'white'
    line_thickness = 1.5
    plt.figure(figsize=(12, 12))
    if suptitle:
        plt.suptitle(img_name)

    plt.imshow(img, cmap=plt.cm.RdBu)
    tp_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                               edgecolor="green", linewidth=line_thickness)
    fp_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                               edgecolor="red", linewidth=line_thickness)
    fn_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                               edgecolor="blue", linewidth=line_thickness)
    ae_patch = mpatches.Circle((0.1, 0.1), 0, facecolor=legend_background_color,
                               edgecolor=legend_background_color, linewidth=line_thickness)
    plt.title("Predicted count: {} - True count: {}".format(pred_rgb, true_count), fontsize=18)

    legend = plt.legend([tp_patch, fp_patch, fn_patch, ae_patch],
                        ["True Positive: {}".format(tp), "False Positive: {}".format(fp),
                         "False Negative: {}".format(fn), "Absolute Error: {}".format(ae)],
                        bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=4, fontsize=14)
    frame = legend.get_frame()
    frame.set_color(legend_background_color)

    if img_name in ['281.tiff', '254.tiff', '278.tiff', '168.tiff']:
        outpath = Path('/home/luca/PycharmProjects/cell_counting_yellow/results/figures')
        outname = f"fig7:pred_ResUnet:{img_name.split('.')[0]}.pdf"
        print('Saving at: ', str(outpath / outname))
        plt.savefig(outpath / outname, bbox_inches='tight')
    plt.show()
    return (None)


def plot_MAE(test_metrics):
    '''Plot mean absolute error distribution based on pandas dataframe. Return None.'''

    sns.set_style('whitegrid')

    # N.B. the dataframe must contain true and predicted counts in two columns named as follows
    mae_list = list(abs(test_metrics.Target_count - test_metrics.Predicted_count))

    fig = plt.figure(figsize=(15, 6))
    suptit = plt.suptitle("Absolute Error Distribution")

    color = 'blue'

    MAX = max(mae_list)

    sb = plt.subplot(1, 2, 1)
    box = plt.boxplot(mae_list, vert=0, patch_artist=True, labels=[""])
    plt.xlabel("Absolute Error")
    plt.ylabel("MAE")

    t = plt.text(2, 1.15, 'Mean Abs. Err.: {:.2f}\nMedian Abs. Err.: {:.2f}\nStd. Dev.: {:.2f}'.format(
        np.array(mae_list).mean(), np.median(np.array(mae_list)), np.array(mae_list).std()),
                 bbox={'facecolor': color, 'alpha': 0.5, 'pad': 5})

    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
    _ = plt.xticks(range(0, MAX, 5))

    sb = plt.subplot(1, 2, 2)

    dens = sns.distplot(np.array(mae_list), bins=20, color=color, hist=True, norm_hist=False)
    _ = plt.xlim(0, MAX)
    _ = dens.axes.set_xticks(range(0, max(mae_list), 5))
    _ = plt.axvline(np.mean(mae_list), 0, 1, color="firebrick", label="Mean Abs. Err.")
    _ = plt.axvline(np.median(mae_list), 0, 1, color="goldenrod", label="Median Abs. Err.")

    # Plot formatting
    leg = plt.legend(title="Model")
    xlab = plt.xlabel('Absolute Error')
    ylab = plt.ylabel('Density')

    plt.show()
    return (None)


def plot_MPE(test_metrics):
    '''Plot mean percentage error distribution based on pandas dataframe. Return None.'''

    sns.set_style('whitegrid')

    # N.B. the dataframe must contain true and predicted counts in two columns named as follows
    mpe_list = list(
        (test_metrics.Predicted_count - test_metrics.Target_count) / (test_metrics.Target_count + 10 ** (-6)))

    fig = plt.figure(figsize=(15, 6))
    suptit = plt.suptitle("Percentage Error Distribution")

    color = 'green'

    MIN = min(mpe_list)
    MAX = max(mpe_list)

    sb = plt.subplot(1, 2, 1)
    box = plt.boxplot(mpe_list, vert=0, patch_artist=True, labels=[""])
    plt.xlabel("Percentage Error")
    plt.ylabel("MPE")

    t = plt.text(-0.9, 1.15, 'Mean Perc. Err.: {:.2f}\nMedian Perc. Err.: {:.2f}\nStd. Dev.: {:.2f}'.format(
        np.array(mpe_list).mean(), np.median(np.array(mpe_list)), np.array(mpe_list).std()),
                 bbox={'facecolor': color, 'alpha': 0.5, 'pad': 5})

    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
    # _ = plt.xticks(range(0,MAX, 5))

    sb = plt.subplot(1, 2, 2)

    dens = sns.distplot(np.array(mpe_list), bins=20, color=color, hist=True, norm_hist=False)
    _ = plt.xlim(MIN, MAX)
    # _ = dens.axes.set_xticks(range(0,max(mae_list),5))
    _ = plt.axvline(np.mean(mpe_list), 0, 1, color="firebrick", label="Mean Perc. Err.")
    _ = plt.axvline(np.median(mpe_list), 0, 1, color="goldenrod", label="Median Perc. Err.")

    # Plot formatting
    leg = plt.legend(title="Model")
    xlab = plt.xlabel('Percentage Error')
    ylab = plt.ylabel('Density')

    plt.show()
    return (None)
