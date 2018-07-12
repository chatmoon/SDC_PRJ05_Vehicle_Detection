from step0 import PARSE_ARGS, parameters
from step01 import list_all_images
from step02 import get_hog_features, bin_spatial, color_hist, extract_features, slide_window
from step02 import draw_boxes, single_img_features, search_windows, visualize, step02_test
from step03 import classifier

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # sklearn v 0.18import
# from sklearn.cross_validation import train_test_split


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)



def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)
    var       = parameters()

    # set variables
    y_start_stop   = var['y_start_stop'] # [400, None]  # [None, None] # [400, 656] [None, None] # Min and Max in y to search in slide_window()
    xy_window      = var['xy_window']    # (128,128) (96,96) (64,64)
    overlap        = var['overlap']      # 0.5

    X_scaler, scaled_X, svc = classifier(args, var, to_print=False)

    # list images to read/open
    example_images = glob.glob(args.test + '*.jpg')
    images, titles = [], []

    for count, img_src in enumerate(example_images):
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255
        if count % int(len(example_images) / 5) == 0:
            print('img[min: {:.2f}, max: {:.2f}]|'.format(np.min(img), np.max(img)), end=' ')

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=xy_window,
                               xy_overlap=(overlap, overlap))

        if count % int(len(example_images) / 5) == 0: print('num windows: {}|'.format(len(windows)), end=' ')

        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=var['color_space'], spatial_size=var['spatial_size'], hist_bins=var['hist_bins'],
                                     orient=var['orient'], pix_per_cell=var['pix_per_cell'], cell_per_block=var['cell_per_block'],
                                     hog_channel=var['hog_channel'], spatial_feat=var['spatial_feat'], hist_feat=var['hist_feat'], hog_feat=var['hog_feat'])

        window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
        images.append(window_img)
        titles.append('')
        if count % int(len(example_images) / 5) == 0: print(
            '{:.3f} s to process 1 img searching {} windows'.format(time.time() - t1, len(windows)))
    # fig = plt.figure(figsize=(12,18), dpi=300)
    figsize = (15, 7)
    visualize(figsize, 3, images, titles)



if __name__ == '__main__':
    main()
