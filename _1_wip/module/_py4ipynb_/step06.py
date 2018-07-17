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
from sklearn.model_selection import train_test_split  # sklearn v 0.18import
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# parameter
directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
args      = PARSE_ARGS(path=directory)
var       = parameters()

# Helper function: iterate over test images
def find_cars(var, img):
    # set variables
    pix_per_cell   = var['pix_per_cell']
    orient         = var['orient']
    cell_per_block = var['cell_per_block']
    spatial_size   = var['spatial_size']
    hist_bins      = var['hist_bins']
    hog_channel    = var['hog_channel']

    draw_img = np.copy(img)
    # make a heatmap of zeros
    heatmap = np.zeros_like(img[:, :, 0])
    img = img.astype(np.float32) / 255

    img_tosearch = img[var['y_start_stop'][0]:var['y_start_stop'][1], :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if var['scale'] != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / var['scale']), np.int(imshape[0] / var['scale'])))

    if hog_channel == 'ALL':
        (ch1, ch2, ch3) = [ctrans_tosearch[:, :, i] for i in range(3)]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    # define blocks and steps as above
    (nyblocks, nxblocks) = [(ch1.shape[i] // var['pix_per_cell']) - 1 for i in range(2)]
    nfeat_per_block = var['orient'] * var['cell_per_block'] ** 2
    window, cells_per_step = 64, 2  # instead of overlap, define how many cells to step
    nblocks_per_window = (window // var['pix_per_cell']) - 1
    (nysteps, nxsteps) = [(i - nblocks_per_window) // cells_per_step for i in [nyblocks, nxblocks]]

    # compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        (hog1, hog2, hog3) = [get_hog_features(i, orient, pix_per_cell, cell_per_block, feature_vec=False) for i in [ch1, ch2, ch3]]
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    img_boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # count += 1
            (ypos, xpos) = [i * cells_per_step for i in [yb, xb]]
            # extract HOG for this patch
            if hog_channel == 'ALL':
                (hog_feat1, hog_feat2, hog_feat3) = [
                    i[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel() for i in
                    [hog1, hog2, hog3]]
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            (ytop, xleft) = [i * var['pix_per_cell'] for i in [ypos, xpos]]

            # extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # get color features
            spatial_features = bin_spatial(subimg, size=var['spatial_size'])
            hist_features    = color_hist(subimg, nbins=var['hist_bins'])

            print('[SHAPE] spatial_features {}, hist_features {}, hog_features {}'
                  .format(spatial_features.shape, hist_features.shape, hog_features.shape))

        # scale features and make a prediction
        X_scaler, _, svc = classifier(args, var, to_print=False)
        test_features    = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        test_prediction  = svc.predict(test_features)

        if test_prediction == 1:
            (xbox_left, ytop_draw, win_draw) = [np.int(i * var['scale']) for i in [xleft, ytop, window]]
            cv2.rectangle(draw_img, (xbox_left, ytop_draw + var['y_start_stop'][0]),
                          (xbox_left + win_draw, ytop_draw + win_draw + var['y_start_stop'][0]), (0, 0, 255))
            img_boxes.append(((xbox_left, ytop_draw + var['y_start_stop'][0]), (xbox_left + win_draw, ytop_draw + win_draw + var['y_start_stop'][0])))
            heatmap[ytop_draw + var['y_start_stop'][0]:ytop_draw + win_draw + var['y_start_stop'][0], xbox_left:xbox_left + win_draw] += 1

    return draw_img, heatmap


# Helper function: return thresholded map
def apply_threshold(heatmap, threshold):
    # zeros out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap


# Helper function: return the image labeled with bboxes
def draw_labeled_bboxes(img, labels):
    # iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # identify x and y values of those pixels
        (nonzeroy, nonzerox) = [np.array(nonzero[i]) for i in range(2)]
        # define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


def step06_test():
    out_images, out_maps = [], []
    ystart, ystop = 400, 656
    scale = 1.5
    # iterate over test images
    for img_src in example_images:
        img = mpimg.imread(img_src)
        out_img, heat_map = find_cars(img, scale)
        labels = label(heat_map)
        # draw bounding boxes on a copy of the image
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        out_images.append(draw_img)
        out_images.append(heat_map)
    fig = plt.figure(figsize=(12, 24))
    visualize(fig, 8, 2, out_images, out_titles)

# Class VEHICLE():
#     def __init__(self):
#         self.detected        = False # was the Vehicle detected in the last iteration ?
#         self.n_detections    = 0     # number of times this vehicle has been ?
#         self.n_nondetections = 0     # number of consecutive times this car has not been detected ...
#         self.xpixels         = None  # pixel x values of last detection
#         self.ypixels         = None  # pixel y values of last detection
#         self.recent_xfitted  = []    # x position of the last n fits of the bounding box
#         self.bestx           = None  # average x position of the last n fits
#         self.recent_yfitted  = []    # y position of the last n fits of the bounding box
#         self.besty           = None  # average y position of the last n fits
#         self.recent_wfitted  = []    # width of the last n fits of the bounding box
#         self.bestw           = None  # average width of the last n fits
#         self.recent_hfitted  = []    # height of the last n fits of the bounding box
#         self.besth           = None  # average height of the last n fits



def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args = PARSE_ARGS(path=directory)

    test_output = args.path + 'test.mp4'
    clip = VideoFileClip(args.path + 'test_video.mp4')
    test_clip = clip.fl_image(process_image)
    test_clip.write_videofile(test_output, audio=False)

    carslist = []
    carslist.append(VEHICLE())
    process_image(img)


if __name__ == '__main__':
    main()