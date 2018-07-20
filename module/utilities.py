from step0 import PARSE_ARGS, parameters
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
import pickle


# Helper function: plot images
def images_plot(args, camera_dictionary):
    fig, ax_lanes = plt.subplots(figsize=(15, 13), nrows=5, ncols=1)
    string_list = [' ', ' not']

    for row, ax_lane in enumerate(ax_lanes, start=1):
        if row == 1 or row == 5:
            ax_lane.set_title('{}. the camera calibration images for which the corners were{} found\n'.format(row % 3, string_list[ row % 3 - 1]), fontsize=20)
        # Turn off axis lines and ticks of the lane subplot
        ax_lane.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # remove the white frame
        ax_lane._frameon = False

    # load images with/without drawn corners
    images_corners_found = corners_draw(args, camera_dictionary)
    images_corners_not_found = [mpimg.imread(image) for image in camera_dictionary['images_corners_not_found']]
    len_found, len_not_found = len(images_corners_found), len(images_corners_not_found)
    size_x, size_y, channel = images_corners_found[0].shape
    image_white = np.zeros([size_x, size_y, 3], dtype=np.uint8)
    image_white.fill(255)

    for i in range(1, 1 + 25):
        # select where to plot the image in the grid
        ax_image = fig.add_subplot(5, args.column, i)

        if i in range(1, 1 + 17):  # images_corners_found
            offset = 0
            image = images_corners_found[(i - 1)]
            title = camera_dictionary['images_corners_found'][(i - 1)].split('\\')[-1]
            # image_qty = len_found
        elif i in range(1 + 17, 21):
            image = image_white
            title = ''
            # image_qty = len_found
        elif i in range(21, 1 + 23):  # images_corners_not_found
            offset = 20
            image = images_corners_not_found[(i - 1) - offset]
            title = camera_dictionary['images_corners_not_found'][(i - 1) - offset].split('\\')[-1]
            # image_qty = len_not_found
        else:
            image = image_white
            title = ''
            # image_qty = len_not_found
        ax_image.imshow(image)
        ax_image.axis('off')

        ax_image.set_title(title, fontsize=16)
        ax_image.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        plt.imshow(image)

    plt.tight_layout()
    plt.show()

# Helper function: plot images
def plot_images(args, image_to_plot, column=5, figsize=(15, 13)):
    # plot images
    remainder = len(image_to_plot) % column
    iquotient = len(image_to_plot) // column
    rows = iquotient if remainder == 0 else 1 + iquotient

    figure, axes = plt.subplots(rows, column, figsize=figsize)  # (15, 13)
    w = rows * column - len(image_to_plot)
    _ = [axes[-1, -i].axis('off') for i in range(1, w + 1)]
    figure.tight_layout()

    for ax, image in zip(axes.flatten(), image_to_plot):
        frame = mpimg.imread(image[1])
        ax.imshow(frame)
        ax.set_title(image[0], fontsize=15)
    plt.show()


def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)
    num_img   = 5

    # list_all_images
    cars    = glob.glob(args.cars+'/*/*') # list all images in Vehicle folders
    notcars = glob.glob(args.notcars+'/*/*') # list all images in Non-vehicle folders

    cars_image_to_plot    = [[cars[index].split('\\')[-1][:-4], cars[index]] for index in
                             [ random.randint(0, len(cars)) for i in range(num_img) ]]
    notcars_image_to_plot = [[notcars[index].split('\\')[-1][:-4], notcars[index]] for index in
                             [random.randint(0, len(notcars)) for i in range(num_img)]]

    # plot images
    plot_images(args, cars_image_to_plot)
    plot_images(args, notcars_image_to_plot)

if __name__ == '__main__':
    main()