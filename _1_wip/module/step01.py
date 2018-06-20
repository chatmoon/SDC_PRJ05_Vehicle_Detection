from step0 import PARSE_ARGS
import os
import glob

def list_all_images(args):
    # list all images in Vehicle folders
    image_types = os.listdir(args.cars)
    cars        = []

    for imtype in image_types:
        cars.extend(glob.glob(args.cars+imtype+'/*'))

    print('Number of vehicle images found: ', len(cars))
    with open(args.out+'cars.txt', 'w') as f:
        for fn in cars:
            f.write(fn+'\n')

    # list all images in Non-vehicle folders
    image_types = os.listdir(args.notcars)
    notcars     = []

    for imtype in image_types:
        notcars.extend(glob.glob(args.notcars+imtype+'/*'))

    print('Number of non-vehicle images found: ', len(notcars))
    with open(args.out+'notcars.txt', 'w') as f:
        for fn in notcars:
            f.write(fn+'\n')

    return cars, notcars


def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)
    
    # list all images in Vehicle and Non-vehicle folders
    cars, notcars = list_all_images(args)
    print()
    print( 'cars   : {}'.format(cars[0].split('\\')[-1]))
    print( 'notcars: {}'.format(notcars[0].split('\\')[-1]))


if __name__ == '__main__':
    main()


















