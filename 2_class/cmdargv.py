import argparse, sys, os

def parse_argv(argv, ann_name):
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', help='path to dataset (.csv file)')
    # parser.add_argument('-h', '--help', help='print this help message')
    parser.add_argument('-s', '--step', help='step size for trainning',
        dest='step_size', type=int, required=True)
    parser.add_argument('-u', '--units', help='recurrent neural network size',
        dest='units', type=int, required=True)
    parser.add_argument('-m', '--model', help='save trainned moldel',
        dest='model_path')
    parser.add_argument('-l', '--log', help='save detailed trainning log (.csv file)',
        dest='log_path')
    parser.add_argument('-a', '--statistics', help='save / append trainning statistics (.csv file)',
        dest='statistics_path')
    parser.add_argument('-d', '--drop', help='drop duplicates logs (in statistics file)',
        dest='drop_duplicates', action='store_true')
    parser.add_argument('-x', '--exclude', help='exclude columns (support RegEx)',
        dest='exclude')

    # args = parser.parse_args(['--step=15', '--unit=30', '../2_classes/data1.csv'])
    args = parser.parse_args()

    if os.path.isfile(args.dataset):

        if args.model_path == None:
            args.model_path = '{}/{}_Size{}_Units{}.h5'.format(args.dataset[:-4], ann_name, args.step_size, args.units)

        if args.log_path == None:
            args.log_path = '{}/{}_Size{}_Units{}.csv'.format(args.dataset[:-4], ann_name, args.step_size, args.units)

        if args.statistics_path == None:
            args.statistics_path = '{}/statistics.csv'.format(args.dataset[:-4])
    else:

        if args.model_path == None:
            args.model_path = '{}_{}_Size{}_Units{}.h5'.format(args.dataset, ann_name, args.step_size, args.units)

        if args.log_path == None:
            args.log_path = '{}_{}_Size{}_Units{}.csv'.format(args.dataset, ann_name, args.step_size, args.units)

        if args.statistics_path == None:
            args.statistics_path = '{}_statistics.csv'.format(args.dataset)

    return args

def main(argv):
    options = parse_argv(argv, 'Test')
    print(options)

if __name__ == '__main__':
    main(sys.argv)
