from utilities import *
from main import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import csv
import time
from skimage.feature import hog # note: hog() takes in a single color channel or grayscaled image as input
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob
import random

# Helper function: experiement and choose of HOG parameters
def exp_hog_parameters(args, var, to_log=True):
    # define feature parameters
    cell_per_block  = var['cell_per_block'] # 2
    color_space     = var['color_space']    # can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hist_bins       = var['hist_bins']      # 32  # number of histogram bins
    hist_feat       = var['hist_feat']      # histogram features on or off
    hog_channel     = var['hog_channel']    # 'ALL' # can be 0, 1, 2, or 'ALL'
    hog_feat        = var['hog_feat']       # HOG features on or off
    orient          = var['orient']         # 8
    overlap         = var['overlap']        # 0.5
    pix_per_cell    = var['pix_per_cell']   # 8
    scale           = var['scale']          # 1.0
    spatial_feat    = var['spatial_feat']   # True, spatial features on or off
    spatial_size    = var['spatial_size']   # (32,32)  # spatial binning dimensions
    x_start_stop    = var['x_start_stop']   # [None, None]
    y_start_stop    = var['y_start_stop']   # [400, 656]
    xy_window       = var['xy_window']      # (128, 128)

    # list_all_images
    cars, notcars = list_all_images(args)

    t            = time.time()
    n_samples    = 1000
    random_idxs  = np.random.randint(0, len(cars), n_samples)
    test_cars    = np.array(cars)[random_idxs]
    test_noncars = np.array(notcars)[random_idxs]

    car_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(test_noncars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    t_feature_computation = round(time.time() - t, 2)
    print(t_feature_computation, 'Seconds to compute features...')
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # fit a per_column scaler
    X_scaler = StandardScaler().fit(X)
    # apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # define the labels vector
    y = np.hstack(( np.ones(len(car_features)), np.zeros(len(notcar_features)) ))

    # split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Using:', orient, 'orientations,', pix_per_cell, 'pixels per cell,', cell_per_block,
          'cells per block,', hist_bins, 'histogram bins, and', spatial_size, 'spatial sampling')
    print('Feature vector length:', len(X_train[0]))

    # use a linear SVC
    svc = LinearSVC()
    # check the training time for the SVC
    t   = time.time()

    svc.fit(X_train, y_train) # https://stackoverflow.com/questions/40524790/valueerror-this-solver-needs-samples-of-at-least-2-classes-in-the-data-but-the
    t_train = round(time.time()-t, 2)
    print(t_train, 'Seconds to train SVC...')
    # check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy)

    log = [ cell_per_block, color_space, hist_bins,
            hist_feat, hog_channel, orient,
            pix_per_cell, spatial_feat, spatial_size,
            accuracy, len(X_train[0]), t_feature_computation, t_train, t_feature_computation+t_train  ]

    # log = [ var['cell_per_block'], var['color_space'], var['hist_bins'],
    #         var['hist_feat'], var['hog_channel'], var['orient'],
    #         var['pix_per_cell'], var['spatial_feat'], var['spatial_size'],
    #         accuracy, len(X_train[0]), t_feature_computation, t_train, t_feature_computation+t_train  ]

    if to_log: log_write(args, log)

def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)
    var       = parameters()

    # define feature parameters
    var['color_space'] = 'YCrCb'  # can be HLS, HSV, LUV, RGB, YCrCb, YUV
    var['orient']      = 9      # 8

    exp_hog_parameters(args, var, to_log=True)

if __name__ == '__main__':
    main()
