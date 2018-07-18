# RESULT: ...

from step0 import PARSE_ARGS, parameters
import os,cv2, time, pickle, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # sklearn v 0.18import
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import gc
from IPython.display import HTML
global heat_list, smooth_factor
heat_list = []
smooth_factor = 15


# parameter
directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
args      = PARSE_ARGS(path=directory)
var       = parameters()


def list_files(args, folder, filename, to_print=False):
    '''
    folder   = args.cars or args.notcars
    filename = cars.txt  or notcars.txt
    '''
    # list all images in Vehicle folders
    image_types = os.listdir(folder)
    list_files  = []

    for imtype in image_types:
        list_files.extend(glob.glob(folder+imtype+'/*'))

    if to_print: print('Number of images found: ', len(list_files), ' | ', filename[:-4])
    with open(args.out+filename, 'w') as f:
        for fn in list_files:
            f.write(fn+'\n')

    return list_files


def list_all_images(args, to_print=False):
    # list all images in Vehicle folders
    cars    = list_files(args, args.cars, 'cars.txt', to_print=to_print)
    # list all images in Non-vehicle folders
    notcars = list_files(args, args.notcars, 'notcars.txt', to_print=to_print)

    return cars, notcars


# CHECK OK # Helper function: return HOG features and visualization
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


# CHECK KO # Helper function: compute binned color features
def bin_spatial(img, size=(32, 32)):
    # create the feature vector
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# CHECK KO # Helper function: compute color histogram features
def color_hist(img, nbins=32): # bins_range=(0 , 256) # see t = 8min46s
    # compute the histogram of the color channels separately
    channel1_hist = np.histogram( img[:, :, 0], bins=nbins )
    channel2_hist = np.histogram( img[:, :, 1], bins=nbins )
    channel3_hist = np.histogram( img[:, :, 2], bins=nbins )
    # concatenate the histograms into a single feautre vector
    hist_features  = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # return the individual histograms, bin_centers and feature vector
    return hist_features


# CHECK OK # Helper function: extract features from a list of images
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
        feature_image = convert_color(img_tosearch, conv=color_space)

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


# CHECK OK # Helper function: take an image, start and stop positions in both x and y, window size (x and y dimensions), and overlap fraction (for both x and y)
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


# CHECK KO # Helper function: draw bounding boxes
def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    # make a copy of the image
    imcopy = np.copy(img)
    # iterate through the bounding boxes
    for bbox in bboxes:
        # draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # return the image copy with boxes drawn
    return imcopy

# CHECK KO # Helper function: extract features from a single image window (for a single image)
def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):
    # create a list to append features
    img_features = []
    # apply color conversation if other than 'RGB'
    feature_image = convert_color(img_tosearch, conv=color_space)
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


# CHECK OK # Helper function: pass an image and the list of windows to be searched ( output of slide_windows() )
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


# CHECK OK
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


def convert_color(img, conv='YCrCb'):  # 'RGB2YCrCb'
    if conv == 'RGB':
        return np.copy(img)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


# L21.35 # Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(args, var, image):
    # set variables
    X_scaler,_,svc = classifier(args, var, to_print=False)
    cell_per_block = var['cell_per_block']
    color_space    = var['color_space']
    hist_bins      = var['hist_bins']
    hog_channel    = var['hog_channel']
    orient         = var['orient']
    pix_per_cell   = var['pix_per_cell']
    scale          = var['scale']
    spatial_size   = var['spatial_size']
    ystart         = var['y_start_stop'][0]
    ystop          = var['y_start_stop'][1]


    #draw_img = np.copy(img)
    bboxes = []
    img    = image.copy().astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return bboxes # draw_img , bboxes

# L23.33 - Multi-scale Windows
def multiscale_bboxes(args, var, image):
    bboxes = []
    for var['scale'] in var['scales']:
        bboxes_scaled = find_cars(args, var, image)
        bboxes        = bboxes + bboxes_scaled
    return bboxes


# L19.37
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

# CHECK OK # Helper function: return thresholded map
def apply_threshold(heatmap, threshold):
    # zeros out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap


# CHECK OK # Helper function: return the image labeled with bboxes
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


# def process_image(image):
#     var      = parameters()
#     #bboxes   = find_cars(args, var, image)
#     bboxes   = multiscale_bboxes(args, var, image)
#     heat     = add_heat(np.zeros_like(image[:,:,0]).astype(np.float),bboxes) # Add heat to each box in box list
#
#
#     heat     = apply_threshold(heat,2)  # ,1) # Apply threshold to help remove false positives
#     heatmap  = np.clip(heat, 0, 255) # Visualize the heatmap when displaying
#     labels   = label(heatmap)
#     draw_img = draw_labeled_bboxes(np.copy(image), labels) # draw bounding boxes on a copy of the image
#     return draw_img

def process_image(image):
    var      = parameters()
    #bboxes   = find_cars(args, var, image)
    bboxes   = multiscale_bboxes(args, var, image)
    heat     = add_heat(np.zeros_like(image[:,:,0]).astype(np.float),bboxes) # Add heat to each box in box list

    heat_list.append(heat)
    heat_smooth = np.int32(np.average(heat_list, 0))

    heat     = apply_threshold(heat_smooth,2)  # ,1) # Apply threshold to help remove false positives
    heatmap  = np.clip(heat, 0, 255) # Visualize the heatmap when displaying
    labels   = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels) # draw bounding boxes on a copy of the image
    return draw_img



# video code
def video(video_input, video_output):
    clip1      = VideoFileClip(video_input)
    video_clip = clip1.fl_image(process_image)
    #% time white_clip.write_videofile(video_output, audio=False)
    video_clip.write_videofile(video_output, audio=False)

def test(args, mp4=0):
    video_input  = args.path + {0: "test_video.mp4", 1: "project_video.mp4"}[mp4]
    video_output = args.out  + {0: "video_output_test.mp4", 1: "video_output_project.mp4"}[mp4]
    video(video_input, video_output)

def test2(args, mp4=0):
    video_input  = args.video + {0: 'project_video_00-15.mp4',
                                 1: 'project_video_15-30.mp4',
                                 2: 'project_video_30-45.mp4',
                                 3: 'project_video_45-50.mp4'}[mp4]

    video_output = args.out   + {0: 'video_output_00-15.mp4',
                                 1: 'video_output_15-30.mp4',
                                 2: 'video_output_30-45.mp4',
                                 3: 'video_output_45-50.mp4'}[mp4]

    video(video_input, video_output)


def cut_video(args, piece=10, mp4=0):
    video_input  = args.path + {0: "test_video.mp4", 1: "project_video.mp4"}[mp4]
    video_output = args.video  + {0: "video_output_test.mp4", 1: "video_output_project.mp4"}[mp4]

    clip     = VideoFileClip(video_input)
    duration = int(clip.duration)
    step     = int(duration/piece)
    #print('duration: {}, step: {}'.format(duration,step))

    for t in range(0,duration,step):
        ffmpeg_extract_subclip(video_input, t, t+step, targetname=video_output[:-4]+'_'+str(t)+'.mp4')

def test3(args):
    videos = glob.glob(args.video + ('*.mp4'))
    video_input, video_output = {}, {}
    count = 0

    for clip in videos:
        if not os.path.exists(args.out + 'video_output_' + str(count) + '.mp4'):
            video_input[count]  = args.video + clip.split('\\')[-1]
            video_output[count] = args.out   + 'video_output_' + str(count) + '.mp4'
            video(video_input[count], video_output[count])
            #gc.collect()
        count += 1




def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)
    var       = parameters()

    # generate video output
    #test(args, mp4=0)
    #test(args, mp4=1)
    #test2(args, mp4=3)
    # for i in range(3):
    #     test2(args, mp4=i)

    #cut_video(args, piece=10, mp4=1)

    test3(args)

if __name__ == '__main__':
    main()