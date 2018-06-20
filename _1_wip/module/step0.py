# Helper function: command-line / parse parameters 
class PARSE_ARGS(object):
    # TODO: replace it by collections.namedtuple , ref: email PyTricks 30.03.18
    def __init__(self, path):

        self.path    = path # root directory path
        self.out     = self.path + 'output_images/'
        self.test    = self.path + 'test_images/'
        self.cars    = self.path + 'data/vehicles/'
        self.notcars = self.path + 'data/non-vehicles/'

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

def main():
    # parameter
    directory = 'D:/USER/_PROJECT_/_PRJ05_/_1_WIP/_1_forge/_v0_/'
    args      = PARSE_ARGS(path=directory)

    # test each args
    print(args.path)
    print(args.out)
    print(args.test)
    print(args.cars)
    print(args.notcars)

if __name__ == '__main__':
    main()
