import sys, getopt, os

def print_useage():
    print('useage: ' + __file__ + ' -s <step_size> -u <unit> <dataset path>')
    print('  -h,    --help      print this help message')
    print('  -s,    --step      step size for trainning')
    print('  -u,    --units     recurrent neural network size')
    print('  -m,    --model     save trainned moldel')
    print('  -l,    --log       save detailed trainning log (.csv file)')
    print('  -a,    --summary   save trainning statistics (.csv file)')
    print('  -d,    --drop      drop duplicates logs (for summary)')
    print('  -x,    --exclude   exclude columns (support RegEx)')

def parse_argv(argv, ann_name):
    options = {
        'step_size': None,
        'units': None,
        'dataset_path': None,
        'model_path': None,
        'log_path': None,
        'statistics_path': None,
        'drop_duplicates': False,
        'exclude': None
    }

    try:
        opts, args = getopt.getopt(argv[1:], 'hs:u:m:l:a:dx:',
            ['help', 'step=', 'units=', 'model=', 'log=', 'summary=', 'drop', 'exclude='])
    except getopt.GetoptError as err:
        print(err)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_useage()
            sys.exit(0)
        elif opt in ('-s', '--step'):
            options['step_size'] = int(arg)
        elif opt in ('-u', '--units'):
            options['units'] = int(arg)
        elif opt in ('-m', '--model'):
            options['model_path'] = arg
        elif opt in ('-l', '--log'):
            options['log_path'] = arg
        elif opt in ('-a', '--summary'):
            options['statistics_path'] = arg
        elif opt in ('-d', '--drop'):
            options['drop_duplicates'] = True
        elif opt in ('-x', '--exclude'):
            options['exclude'] = arg

    if len(args) < 1:
        print('Please specify source (dataset) file')
        print_useage()
        sys.exit(1)

    if os.path.isfile(args[0]):
        # single file mode
        options['dataset_path'] = args[0]
        options['model_path'] = '{}/{}_Sise{}_Units{}'.format(options['dataset_path'][:-4], ann_name, options['step_size'], options['units']) if not isinstance(options['model_path'], str) else options['model_path']
        options['log_path'] = '{}/{}_Sise{}_Units{}.csv'.format(options['dataset_path'][:-4], ann_name, options['step_size'], options['units']) if not isinstance(options['log_path'], str) else options['log_path']
        options['statistics_path'] = '{}/statistics.csv'.format(options['dataset_path'][:-4], ann_name, options['step_size'], options['units']) if not isinstance(options['statistics_path'], str) else options['statistics_path']
    else:
        # folder mode
        options['dataset_path'] = args[0]
        options['model_path'] = '{}_{}_Sise{}_Units{}'.format(options['dataset_path'], ann_name, options['step_size'], options['units']) if not isinstance(options['model_path'], str) else options['model_path']
        options['log_path'] = '{}_{}_Sise{}_Units{}.csv'.format(options['dataset_path'], ann_name, options['step_size'], options['units']) if not isinstance(options['log_path'], str) else options['log_path']
        options['statistics_path'] = '{}_statistics.csv'.format(options['dataset_path'], ann_name, options['step_size'], options['units']) if not isinstance(options['statistics_path'], str) else options['statistics_path']

    for opt_key in options:
        if options[opt_key] == None:
            print('{} is not specified'.format(opt_key))
            print_useage()
            sys.exit(1)

    return options

def main(argv):
    options = parse_argv(argv, 'Test')
    for key in options:
        print('{}, {}'.format(key, options[key]))

if __name__ == '__main__':
    main(sys.argv)
