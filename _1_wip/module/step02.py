from step0 import PARSE_ARGS, parameters
from step01 import list_all_images

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

# Helper function: return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis==True: # call with two outputs if vis==True
        # TODO: switch transform_sqrt = True # False
        features, hog_image = hog(img,orientations=orient,
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block),
                                  transform_sqrt  = True,
                                  visualize = vis, feature_vector = feature_vec ) # visualize instead of visualise
        return features, hog_image
    else:         # otherwise call with one output
        # TODO: switch transform_sqrt = True
        features = hog(img, orientations=orient,
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block),
                                  transform_sqrt  = True,
                                  visualize = vis, feature_vector = feature_vec )
        return features


# Helper function: compute binned color features
def bin_spatial(img, size=(32, 32)):
    # create the feature vector
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# Helper function: compute color histogram features
def color_hist(img, nbins=32): # bins_range=(0 , 256) # see t = 8min46s
    # compute the histogram of the color channels separately
    channel1_hist = np.histogram( img[:, :, 0], bins=nbins )
    channel2_hist = np.histogram( img[:, :, 1], bins=nbins )
    channel3_hist = np.histogram( img[:, :, 2], bins=nbins )
    # concatenate the histograms into a single feautre vector
    hist_features  = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # return the individual histograms, bin_centers and feature vector
    return hist_features


# Helper function: extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # create a list to append feature vectors to
    features = []
    # iterate through the list of images
    for file in imgs:
        file_features = []
        # read in each one by one
        image = mpimg.imread(file)
        # apply color conversation if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # call get_hog_feature() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], orient,
                                        pix_per_cell, cell_per_block, vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # return list of feature vectors
    return features

# Helper function: take an image, start and stop positions in both x and y, window size (x and y dimensions), and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64,64), xy_overlap=(0.5, 0.5)):
    # if x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # compute the number of windows in x/y
    nx_buffer  = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer  = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # initialize a list to append window positions to
    window_list = []
    # loop through finding x and y window positions (see note at t = 12min54s)
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx   = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy   = starty + xy_window[1]
            # append window position to list
            window_list.append(( (startx, starty), (endx,endy) ))
    # return the list of windows
    return window_list

# Helper function: draw bounding boxes
def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    # make a copy of the image
    imcopy = np.copy(img)
    # iterate through the bounding boxes
    for bbox in bboxes:
        # draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return imcopy

# Helper function: extract features from a single image window (for a single image)
def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):
    # create a list to append features
    img_features = []
    # apply color conversation if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    # compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        # apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
        # call get_hog_feature() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient,
                                    pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            # TODO: next line à commenter?
            hog_features = np.concatenate(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                           pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # append the new feature vector to the features list
        img_features.append(hog_features)

        # return concatenated array of features
        if vis == True:
            return np.concatenate(img_features), hog_image
        else:
            return np.concatenate(img_features)


# Helper function: pass an image and the list of windows to be searched ( output of slide_windows() )
def search_windows(img, windows, clf, scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True): # t = 00:16:45
    # create an empty list to receive positive detection windows
    on_windows = []
    # iterate over all windows in the list
    for window in windows:
        # extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)) # S, interpolation = cv.INTER_LINEAR)
        # extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, spatial_size=spatial_size,
                                       hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block, hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        # scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # predict using your classifier
        prediction = clf.predict(test_features)
        # if positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # return windows for positive detections
    return on_windows



def color_map(image):
    if image.shape[-1] == 3:
        cMap = None
    elif image.shape[-1] != 3 or image.shape[-1] == 1:
        cMap ='hot'
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))
    return cMap

# Helper function:  plot multiple images
def visualize(figsize, cols, imgs, titles):
    # plot images
    remainder = len(imgs) % cols
    iquotient = len(imgs) // cols
    rows      = iquotient if remainder == 0 else 1 + iquotient

    figure, axes = plt.subplots(rows, cols, figsize=figsize) # (15, 13)
    w = rows * cols - len(imgs)
    _ = [axes[-1, -i].axis('off') for i in range(1, w + 1)]
    figure.tight_layout()

    for ax, image, title in zip(axes.flatten(), imgs, titles):
        ax.imshow(image, cmap=color_map(image))
        ax.set_title(title, fontsize=15)

    plt.show()


# Helper function: plot multiple images
# def visualize(fig, rows, cols, imgs, titles):
#     for i, img in enumerate(imgs):
#         plt.subplot(rows, cols, i+1)
#         plt.title(i+1)
#         img_dims = len(img.shape)
#         if img_dims < 3:
#             plt.imshow(img, cmap='hot')
#             plt.title(titles[i])
#         else:
#             plt.imshow(img)
#             plt.title(titles[i])
#     plt.show()


def step02_test(args):
    # list_all_images
    cars, notcars = list_all_images(args)

    # choose random car/notcar indices
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # read in car / notcar images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # define feature parameters
    color_space = 'RGB'  # can be RGB HSV LUV HLS YUV YCrCb
    orient = 6
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0  # can be 0 1 2 or 'ALL'
    spatial_size = (16, 16)  # spatial binning dimensions
    hist_bins = 16  # number of histogram bins
    spatial_feat = True  # spatial features on or off
    hist_feat = True  # histogram features on or off
    hog_feat = True  # HOG features on or off

    car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
                                                      hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                      cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                                      vis=True)
    notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space,
                                                            spatial_size=spatial_size,
                                                            hist_bins=hist_bins, orient=orient,
                                                            pix_per_cell=pix_per_cell,
                                                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                            spatial_feat=spatial_feat, hist_feat=hist_feat,
                                                            hog_feat=hog_feat,
                                                            vis=True)

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car HOG image', 'notcar_image', 'notcar HOG image']
    #figure = plt.figure(figsize=(12, 3))  # , dpi=80)
    #visualize(figure, 1, 4, images, titles)
    figsize = (12, 3)
    visualize(figsize, 4, images, titles)



# t = 00:18:49

def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args = PARSE_ARGS(path=directory)

    # list_all_images
    cars, notcars = list_all_images(args)

    # choose random car/notcar indices
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # read in car / notcar images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # define feature parameters
    color_space    = 'RGB'  # can be RGB HSV LUV HLS YUV YCrCb
    orient         = 6
    pix_per_cell   = 8
    cell_per_block = 2
    hog_channel    = 0  # can be 0 1 2 or 'ALL'
    spatial_size   = (16, 16)  # spatial binning dimensions
    hist_bins      = 16  # number of histogram bins
    spatial_feat   = True  # spatial features on or off
    hist_feat      = True  # histogram features on or off
    hog_feat       = True  # HOG features on or off

    car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
                                                      hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                                      cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                      spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                                      vis=True)
    notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space,
                                                            spatial_size=spatial_size,
                                                            hist_bins=hist_bins, orient=orient,
                                                            pix_per_cell=pix_per_cell,
                                                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                            spatial_feat=spatial_feat, hist_feat=hist_feat,
                                                            hog_feat=hog_feat,
                                                            vis=True)

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car HOG image', 'notcar_image', 'notcar HOG image']
    #figure = plt.figure(figsize=(12, 3))  # , dpi=80)
    #visualize(figure, 1, 4, images, titles)
    figsize = (12, 3)
    visualize(figsize, 4, images, titles)


if __name__ == '__main__':
    main()