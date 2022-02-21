import sys, getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hf:')
    except:
        print('Wrong arguments')
    for opt, arg in opts:
        if (opt == '-h'):
            print('python3 main.py -f <FILE NAME>')
        elif (opt == '-f'):
            print(arg)

if __name__ == '__main__':
    main(sys.argv[1:])