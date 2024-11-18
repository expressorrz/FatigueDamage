import argparse


def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description='Arguments')
# args for device
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')
parser.add_argument('--seed', type=int, default=233, help='Seed for random number generator')

# args for dataset
parser.add_argument('--data_path', type=str, default='dataset/', help='Path to the data')
# parser.add_argument('--train_source', type=int, nargs='+', default=[3, 9, 11, 12, 13, 14, 19, 20, 25], help='Source of training data')
parser.add_argument('--test_source', type=int, default=3, help='Source of testing data')
parser.add_argument('--val_ratio', type=float, default=0.3, help='Radio of validation data')


# args for training
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')

# args for model
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')



configs = parser.parse_args()
