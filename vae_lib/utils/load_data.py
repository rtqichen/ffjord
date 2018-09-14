from __future__ import print_function

import torch
import torch.utils.data as data_utils
import pickle
from scipy.io import loadmat

import numpy as np

import os


def load_static_mnist(args, **kwargs):
    """
    Dataloading function for static mnist. Outputs image data in vectorized form: each image is a vector of size 784
    """
    args.dynamic_binarization = False
    args.input_type = 'binary'

    args.input_size = [1, 28, 28]

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_freyfaces(args, **kwargs):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'multinomial'
    args.dynamic_binarization = False

    TRAIN = 1565
    VAL = 200
    TEST = 200

    # start processing
    with open('data/Freyfaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f, encoding="latin1")[0]

    data = data / 255.

    # NOTE: shuffling is done before splitting into train and test set, so test set is different for every run!
    # shuffle data:
    np.random.seed(args.freyseed)

    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28 * 20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28 * 20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28 * 20)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, args


def load_omniglot(args, **kwargs):
    n_validation = 1345

    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    omni_raw = loadmat(os.path.join('data', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-n_validation]
    x_val = train_data[-n_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_caltech101silhouettes(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    caltech_raw = loadmat(os.path.join('data', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_dataset(args, **kwargs):

    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args, **kwargs)
    elif args.dataset == 'caltech':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)

    elif args.dataset == 'freyfaces':
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args
