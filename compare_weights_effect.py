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

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import skimage, skimage.io
    from pathlib import Path

    from keras.models import load_model
    from evaluation_utils import *

    # setup paths --> NOTE: CURRENT PATHS ARE TO BE UPDATED
    # repo_path = Path("/storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow")
    repo_path = Path("/home/luca/PycharmProjects/cell_counting_yellow")
    data_path = '/home/luca/PycharmProjects/cell_counting_yellow/DATASET/test/all_images/images'

    save_path = repo_path / 'weights_map'
    save_path.mkdir(parents=True, exist_ok=True)
    save_path.chmod(16886)  # chmod 776

    WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)

    patch_height = 240
    patch_width = 240

    model_path = "{}/model_results/{}".format(repo_path, 'ResUnet.h5')
    resunet = load_model(model_path, custom_objects={
        'mean_iou': mean_iou, 'dice_coef': dice_coef,
        'weighted_binary_crossentropy': WeightedLoss
    })  # , compile=False)

    model_path = "{}/model_results/{}".format(repo_path, 'ResUnetNwm.h5')
    resunet_nwm = load_model(model_path, custom_objects={
        'mean_iou': mean_iou, 'dice_coef': dice_coef,
        'weighted_binary_crossentropy': WeightedLoss
    })  # , compile=False)

    ### RESUNET
    # image 278
    img_278 = skimage.io.imread(Path(data_path) / '278.tiff')
    img_278 = np.expand_dims(img_278[:, :, :3], 0)
    pred_mask_278 = np.squeeze(resunet.predict(img_278 / 255.))
    patch1 = pred_mask_278[180:(180 + patch_height), 1280:(1280 + patch_width)]
    patch2 = pred_mask_278[370:(370 + patch_height), 280:(280 + patch_width)]
    # patch1 = pred_mask[200:400, 1300:1500]
    # patch2 = pred_mask[370:610, 280:520]

    # image 275
    img_275 = skimage.io.imread(Path(data_path) / '275.tiff')
    img_275 = np.expand_dims(img_275[:, :, :3], 0)
    pred_mask_275 = np.squeeze(resunet.predict(img_275 / 255.))
    patch3 = pred_mask_275[635:(635 + patch_height), 535:(535 + patch_width)]
    patch4 = pred_mask_275[205:(205 + patch_height), 955:(955 + patch_width)]
    # patch3 = pred_mask[680:830, 590:720]
    # patch4 = pred_mask[250:400, 1000:1150]

    ### RESUNET NO WEIGHTS
    # image 278
    pred_mask_278_nwm = np.squeeze(resunet_nwm.predict(img_278 / 255.))
    patch1_nwm = pred_mask_278_nwm[180:(180 + patch_height), 1280:(1280 + patch_width)]
    patch2_nwm = pred_mask_278_nwm[370:(370 + patch_height), 280:(280 + patch_width)]
    # patch1 = pred_mask[200:400, 1300:1500]
    # patch2 = pred_mask[370:610, 280:520]

    # image 275
    pred_mask_275_nwm = np.squeeze(resunet_nwm.predict(img_275 / 255.))
    patch3_nwm = pred_mask_275_nwm[635:(635 + patch_height), 535:(535 + patch_width)]
    patch4_nwm = pred_mask_275_nwm[205:(205 + patch_height), 955:(955 + patch_width)]
    # patch3 = pred_mask[680:830, 590:720]
    # patch4 = pred_mask[250:400, 1000:1150]

    ### GRIDSPECS
    ncol, nrow = 4, 2
    fig = plt.figure(figsize=(ncol, nrow))

    from matplotlib import gridspec

    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.02, hspace=0.01,
                           top=0.95, bottom=0.05,
                           height_ratios=[0.42, 0.42],
                           left=0.01, right=0.9,
                           width_ratios=[0.23, 0.23, 0.23, 0.23])

    patch_list = [patch1, patch2, patch3, patch4, patch1_nwm, patch2_nwm, patch3_nwm, patch4_nwm]

    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            im = ax.imshow(patch_list[ncol * i + j], cmap='jet')
            # im = ax.imshow(np.random.rand(240, 240), cmap='jet')
            ax.axis('off')

    fig.subplots_adjust(right=0.9)  # , left=0.05, top=0.95, bottom=0.05)
    cbar_ax = fig.add_axes([0.91, 0.05, 0.03, 0.9])
    # # cbar_ax = plt.subplot(gs[i, j+1])
    cbar = fig.colorbar(im, cax=cbar_ax)  # , ticks=[0, 0.2, 0.4, 0.6, 0.8, 0.99])
    cbar.ax.tick_params(labelsize=6)
    # cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.show()

    fig.savefig(save_path / 'weigths_effect.png', dpi=300, transparent=True)