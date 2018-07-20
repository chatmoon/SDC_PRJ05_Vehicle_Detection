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

if __name__ == '__main__':
    main()
