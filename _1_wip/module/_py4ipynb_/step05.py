from module._py4ipynb_.step0 import PARSE_ARGS, parameters
from module._py4ipynb_.step01 import list_all_images
from module._py4ipynb_.step02 import get_hog_features, bin_spatial, color_hist, extract_features, slide_window
from module._py4ipynb_.step02 import draw_boxes, single_img_features, search_windows, visualize, step02_test
from module._py4ipynb_.step03 import classifier
from module._py4ipynb_.step04 import convert_color

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
    args      = PARSE_ARGS(path=directory)
    var       = parameters()

    X_scaler, scaled_X, svc = classifier(args, var, to_print=False)

    # set variables
    pix_per_cell   = var['pix_per_cell']
    orient         = var['orient']
    cell_per_block = var['cell_per_block']
    spatial_size   = var['spatial_size']
    hist_bins      = var['hist_bins']
    hog_channel    = var['hog_channel']
    # = var['cell_per_block']

    # list images to read/open
    example_images = glob.glob(args.test + '*.jpg')

    out_images, out_maps, out_titles, out_boxes = [], [], [], []

    # consider a narrower swath in y
    ystart, ystop, scale = 400, 656, 1.5 # 1 2
    # iterate over test image
    for img_src in example_images:
        img_boxes = []
        t = time.time()
        count = 0
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        # make a heatmap of zeros
        heatmap = np.zeros_like(img[:,:,0])
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        if hog_channel == 'ALL':
            (ch1, ch2, ch3) = [ ctrans_tosearch[:,:,i] for i in range(3) ]
        else:
            ch1 = ctrans_tosearch[:,:,hog_channel]

        # define blocks and steps as above
        ( nyblocks, nxblocks ) = [ (ch1.shape[i] // pix_per_cell) - 1 for i in range(2) ]
        nfeat_per_block        = orient * cell_per_block**2
        window, cells_per_step = 64, 2 # instead of overlap, define how many cells to step
        nblocks_per_window     = ( window // pix_per_cell ) - 1
        ( nysteps, nxsteps )   = [ (i - nblocks_per_window )//cells_per_step for i in [nyblocks, nxblocks] ]

        # compute individual channel HOG features for the entire image
        if hog_channel == 'ALL':
            (hog1, hog2, hog3) = [get_hog_features(i, orient, pix_per_cell, cell_per_block, feature_vec=False) for i in [ch1, ch2, ch3]]
        else:
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                ( ypos, xpos ) = [ i * cells_per_step for i in [yb, xb] ]
                # extract HOG for this patch
                if hog_channel == 'ALL':
                    ( hog_feat1, hog_feat2, hog_feat3 ) = [ i[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() for i in [hog1, hog2, hog3] ]
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                ( ytop, xleft ) = [ i*pix_per_cell for i in [ypos, xpos] ]

                # extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

                # get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features    = color_hist(subimg, nbins=hist_bins)

            # scale features and make a prediction
            test_features   = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                ( xbox_left, ytop_draw, win_draw ) = [ np.int( i * scale ) for i in [xleft, ytop, window] ]
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart), (0,0,255))
                img_boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

        print(time.time()-t, 'seconds to run, total windows = ', count)

        out_images.append(draw_img)
        out_titles.append(img_src[-12:])

        out_images.append(heatmap)
        out_maps.append(heatmap)
        out_boxes.append(img_boxes)

    figsize = (15, 7) # (12,24)
    visualize(figsize, 3, images, titles)
    
if __name__ == '__main__':
    main()
