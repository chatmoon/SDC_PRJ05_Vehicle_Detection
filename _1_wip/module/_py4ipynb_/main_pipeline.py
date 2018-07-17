from step0 import PARSE_ARGS, parameters
from step01 import list_all_images
from step02 import get_hog_features, bin_spatial, color_hist, extract_features, slide_window
from step02 import draw_boxes, single_img_features, search_windows, visualize, step02_test
from step03 import classifier
from step04 import convert_color
from step06 import find_cars, apply_threshold, draw_labeled_bboxes, step06_test

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
#var['X_scaler'], var['scaled_X'], var['svc'] = classifier(args, var, to_print=False)

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


def process_image(image):
    var = parameters()
    _ , heat_map = find_cars(var, image) # var['scale'])
    labels = label(heat_map)
    # draw bounding boxes on a copy of the image
    draw_image = draw_labeled_bboxes(np.copy(image), labels)
    return draw_image

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

def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)
    var       = parameters()

    # generate video output
    test(args, mp4=0)
    # test(args, mp4=1)

if __name__ == '__main__':
    main()