from module._py4ipynb_.step0 import PARSE_ARGS, parameters
from module._py4ipynb_.step01 import list_all_images
from module._py4ipynb_.step02 import get_hog_features, bin_spatial, color_hist, extract_features, slide_window
from module._py4ipynb_.step02 import draw_boxes, single_img_features, search_windows, visualize, step02_test

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # sklearn v 0.18import
# from sklearn.cross_validation import train_test_split


def classifier(args, var, to_print=True):
    if os.path.exists(args.pickled + 'svc.p'):
        # de-serialize/load X_scaler, scaled_X, svc:
        X_scaler = pickle.load(open(args.pickled + "X_scaler.p", "rb"))
        scaled_X = pickle.load(open(args.pickled + "scaled_X.p", "rb"))
        svc      = pickle.load(open(args.pickled + "svc.p", "rb"))
    else:
        # list_all_images
        cars, notcars = list_all_images(args)

        # choose random car/notcar indices
        car_ind = np.random.randint(0, len(cars))
        notcar_ind = np.random.randint(0, len(notcars))

        # read in car / notcar images
        car_image = mpimg.imread(cars[car_ind])
        notcar_image = mpimg.imread(notcars[notcar_ind])

        t            = time.time()
        n_samples    = 1000
        random_idxs  = np.random.randint(0, len(cars), n_samples)
        test_cars    = np.array(cars)[random_idxs]
        test_noncars = np.array(notcars)[random_idxs]

        car_features, car_hog_image = single_img_features(car_image, color_space=var['color_space'], spatial_size=var['spatial_size'],
                                                          hist_bins=var['hist_bins'], orient=var['orient'], pix_per_cell=var['pix_per_cell'],
                                                          cell_per_block=var['cell_per_block'], hog_channel=var['hog_channel'],
                                                          spatial_feat=var['spatial_feat'], hist_feat=var['hist_feat'], hog_feat=var['hog_feat'],
                                                          vis=True)

        notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=var['color_space'],
                                                                spatial_size=var['spatial_size'],
                                                                hist_bins=var['hist_bins'], orient=var['orient'],
                                                                pix_per_cell=var['pix_per_cell'],
                                                                cell_per_block=var['cell_per_block'], hog_channel=var['hog_channel'],
                                                                spatial_feat=var['spatial_feat'], hist_feat=var['hist_feat'],
                                                                hog_feat=var['hog_feat'],
                                                                vis=True)


        car_features = extract_features(test_cars, color_space=var['color_space'], spatial_size=var['spatial_size'], hist_bins=var['hist_bins'],
                                        orient=var['orient'], pix_per_cell=var['pix_per_cell'], cell_per_block=var['cell_per_block'],
                                        hog_channel=var['hog_channel'], spatial_feat=var['spatial_feat'], hist_feat=var['hist_feat'], hog_feat=var['hog_feat'])

        notcar_features = extract_features(test_noncars, color_space=var['color_space'], spatial_size=var['spatial_size'], hist_bins=var['hist_bins'],
                                        orient=var['orient'], pix_per_cell=var['pix_per_cell'], cell_per_block=var['cell_per_block'],
                                        hog_channel=var['hog_channel'], spatial_feat=var['spatial_feat'], hist_feat=var['hist_feat'], hog_feat=var['hog_feat'])

        if to_print: print(time.time()-t, 'Seconds to compute features...')

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # fit a per_column scaler
        X_scaler = StandardScaler().fit(X)
        # apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # define the labels vector
        y = np.hstack(( np.ones(len(car_features)), np.zeros(len(notcar_features)) ))

        # split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state) # test_size=0.1

        if to_print: print('Using:', var['orient'], 'orientations,', var['pix_per_cell'], 'pixels per cell,', var['cell_per_block'],
                           'cells per block,', var['hist_bins'], 'histogram bins, and', var['spatial_size'], 'spatial sampling')
        if to_print: print('Feature vector length:', len(X_train[0]))

        # use a linear SVC
        svc = LinearSVC()
        # check the training time for the SVC
        t   = time.time()
        svc.fit(X_train, y_train) # https://stackoverflow.com/questions/40524790/valueerror-this-solver-needs-samples-of-at-least-2-classes-in-the-data-but-the
        if to_print: print(round(time.time()-t, 2), 'Seconds to train SVC...')
        # check the score of the SVC
        if to_print: print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        if to_print: print('[SHAPE] Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        # serialize/store X_scaler, scaled_X, svc:
        pickle.dump(X_scaler, open(args.pickled + "X_scaler.p", 'wb'))
        pickle.dump(scaled_X, open(args.pickled + "scaled_X.p", 'wb'))
        pickle.dump(svc, open(args.pickled + "svc.p", 'wb'))

    return X_scaler, scaled_X, svc





def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args = PARSE_ARGS(path=directory)
    var  = parameters()

    # fit a per_column scaler, apply the scaler to X, use a linear SVC
    X_scaler, scaled_X, svc = classifier(args, var, to_print=True)


    '''
    Expected result:
    86.92713189125061 Seconds to compute features...
    Using: 6 orientations, 8 pixels per cell, 2 cells per block, 16 histogram bins, and (16, 16) spatial sampling
    Feature vector length: 1992
    2.34 Seconds to train SVC...
    Test Accuracy of SVC =  0.955
    '''

if __name__ == '__main__':
    main()
