from step0 import PARSE_ARGS, parameters
import os
import glob
import time

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


def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)

    t = time.time()
    # list all images in Vehicle and Non-vehicle folders
    cars, notcars = list_all_images(args,to_print=True)
    print()
    print( 'cars   : {}'.format(cars[0].split('\\')[-1]))
    print( 'notcars: {}'.format(notcars[0].split('\\')[-1]))
    print(time.time() - t, 'seconds to run')


if __name__ == '__main__':
    main()


















