import argparse
from warnings import warn    
import json

def parse_ecg_args() -> dict:        
    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('path_to_traces',
                        help='path to file containing ECG traces')
    parser.add_argument('path_to_csv',
                        help='path to csv file containing attributes.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--valid_split', type=float, default=0.05,
                        help='fraction of the data used for validation (default: 0.1).')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='fraction of the data kept away for testing in a latter stage (default: 0.1).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model/',
                        help='output folder (default: ./out)')
    parser.add_argument('--traces_dset', default='tracings',
                        help='traces dataset in the hdf5 file.')
    parser.add_argument('--age_col', default='age',
                        help='column with the age in csv file.')
    parser.add_argument('--gpu_id', default='0',
                        help='Which gpu to use.')
    parser.add_argument('--n_valid', type=int, default=100,
                        help='the first `n_valid` exams in the hdf will be for validation.'
                             'The rest is for training') # how is this different from train/valid split above
    parser.add_argument('--dataset_subset', default=0.001,
                        help='Size of the subset of dataset to take')
    parser.add_argument('--id_key', default="exam_id",
                        help='Name of the exam column in dataset')                        
    parser.add_argument('--tracings_key', default="tracings",
                        help='Name of the tracings column in dataset')                        

    args, unk = parser.parse_known_args()
    args = vars(args)

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    return args

def parse_ecg_json() -> dict:
    parser = argparse.ArgumentParser(add_help=True,
        description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('-f', default=None, help="Path to the json config file")
    args, unk = parser.parse_known_args()
    args = vars(args)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    if args["f"]:
        with open(args["f"], "r") as f:
            args = json.load(f)

    return args
