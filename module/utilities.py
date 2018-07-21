# Helper function: command-line / parse parameters
class PARSE_ARGS(object):
    # TODO: should it be replaced it by collections.namedtuple , ref: email PyTricks 30.03.18? immutable obj
    def __init__(self, path):

        self.path    = path # root directory path
        self.out     = self.path + 'output_images/'
        self.test    = self.path + 'test_images/'
        self.cars    = self.path + 'data/vehicles/'
        self.notcars = self.path + 'data/non-vehicles/'
        self.small   = self.path + 'data/smallset/'
        self.pickled = self.path + 'data/pickled_object/'
        self.video   = self.path + 'video/'
        self.column  = 5

    def path(self):
        return self.path
    def out(self):
        return self.out
    def test(self):
        return self.test
    def cars(self):
        return self.cars
    def notcars(self):
        return self.notcars
    def small(self):
        return self.small
    def pickled(self):
        return self.pickled
    def video(self):
        return self.video
    def column(self):
        return self.column

def parameters():
    '''
    color_space		: can be RGB HSV LUV HLS YUV YCrCb
    hist_bins		: number of histogram bins
    hist_feat		: histogram features on or off
    hog_channel		: can be 0 1 2 or 'ALL'
    hog_feat		: HOG features on or off
    scales          : [0.75,1.,1.5, 1.75]
    spatial_feat	: spatial features on or off
    spatial_size	: spatial binning dimensions, (16, 16) (32, 32)
    xy_window		: (128, 128) (96,96) (64,64)
    y_start_stop	: [ystart:ystop], Min and Max in y to search in slide_window() , [None, None] [400, None] [400, 656]
    '''
    dictionary = {}
    dictionary = {'cell_per_block': 2,
                  'color_space'   : 'YCrCb',
                  'hist_bins'     : 64,
                  'hist_feat'     : True,
                  'hog_channel'   : 'ALL',
                  'hog_feat'      : True,
                  'orient'        : 8,
                  'overlap'       : 0.5,
                  'pix_per_cell'  : 8,
                  'scale'         : 1.,
                  'scales'        : [0.75,1.,1.5, 1.75],
                  'spatial_feat'  : True,
                  'spatial_size'  : (32, 32),
                  'x_start_stop'  : [None, None],
                  'y_start_stop'  : [400, 656],
                  'xy_window'     : (128,128) }
    return dictionary

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
        ax.axis('off')
        ax.set_title(image[0], fontsize=15)
    plt.show()


def cut_video(args, piece=10, mp4=0):
    video_input  = args.path  + {0: "test_video.mp4", 1: "project_video.mp4"}[mp4]
    video_output = args.video + {0: "video_output_test.mp4", 1: "video_output_project.mp4"}[mp4]

    clip     = VideoFileClip(video_input)
    duration = int(clip.duration)
    step     = int(duration/piece)
    #print('duration: {}, step: {}'.format(duration,step))

    for t in range(0,duration,step):
        ffmpeg_extract_subclip(video_input, t, t+step, targetname=video_output[:-4]+'_'+str(t)+'.mp4')


def merge_video(args):
    videos = glob.glob(args.out + '*.mp4')
    clips  = []

    for clip in videos:
        clips.append( VideoFileClip(clip) )

    clips_final = concatenate_videoclips(clips)
    clips_final.write_videofile(args.out + 'video_output.mp4') # , bitrate="5000k")


def test0(args, mp4=0, to_print=False):
    videos = glob.glob(args.video + '*.mp4')
    if to_print:
        _ = [print('{}. {}'.format(count, clip.split('\\')[-1])) for count,clip in enumerate(videos)]
    else:
        video_input  = videos[mp4]
        print('video_input: {}'.format(video_input.split('\\')[-1]))
        video_output = args.out + 'video_output_T' + videos[mp4].split('project_video_')[-1] #+ '.mp4'
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

def test3(args):
    videos = glob.glob(args.video + '*.mp4')
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

    # test each args
    print(args.path)
    print(args.out)
    print(args.test)
    print(args.cars)
    print(args.notcars)
    print(args.small)
    print(var)

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


    ## archives
    # videos = glob.glob(args.video + '*.mp4')
    # _ = [ print( '{}. {}'.format(count, clip.split('\\')[-1])) for count,clip in enumerate(videos)]

    # test0(args, mp4=0) # , to_print=True)
    # test1(args, mp4=2)
    # test2(args, mp4=3)
    # for i in range(3):
    #     test2(args, mp4=i)
    # test3(args)

    #cut_video(args, piece=10, mp4=1)
    #merge_video(args)


    # video_input  = args.path + 'project_video.mp4'
    # video_output = args.video + 'project_video_50.mp4'
    # ffmpeg_extract_subclip(video_input, 29, 50, targetname=video_output)



if __name__ == '__main__':
    main()
