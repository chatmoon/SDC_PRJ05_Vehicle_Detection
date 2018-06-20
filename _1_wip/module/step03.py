from step0 import PARSE_ARGS
from step01 import list_all_images
from step02 import get_hog_features, bin_spatial, color_hist, extract_features, slide_window
from step02 import draw_boxes, single_img_features, search_windows, visualize, step02_test

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

def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args = PARSE_ARGS(path=directory)

    # list_all_images
    cars, notcars = list_all_images(args)

    # define feature parameters
    color_space    = 'RGB' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient         = 6  # 9
    pix_per_cell   = 8
    cell_per_block = 2
    hog_channel    = 0 # 'ALL' # can be 0, 1, 2, or 'ALL'
    spatial_size   = (16,16) # (32,32)  # spatial binning dimensions
    hist_bins      = 16  # 32  # number of histogram bins
    spatial_feat   = True # spatial features on or off
    hist_feat      = True # histogram features on or off
    hog_feat       = True # HOG features on or off

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

    print(time.time()-t, 'Seconds to compute features...')


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
    svc.fit(X_train, y_train)
    print(round(time.time()-t, 2), 'Seconds to train SVC...')
    # check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


if __name__ == '__main__':
    main()
