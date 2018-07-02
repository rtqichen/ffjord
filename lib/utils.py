import os
import logging
import torch


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



class Preprocess(object):
    """
    Preprocesses a tensor defined on compact input [0, 1.]
    Given a number of bits in [1, ..., 8], down samples bit-rate to num_bits, adds unifmorm
    noise of magnitude equal to the lost precision and scales to be in [-1, 1]

    Args:
        num_bits (int): number of bits to use, min = 1, max = 8
    """

    def __init__(self, num_bits):
        self._num_bits = num_bits
        self.num_bins = 2 ** num_bits
        self.div_val = 2 ** (8 - num_bits)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of floats in range [0., 1.]

        Returns:
            Tensor: Tensor with specified bit-rate
        """
        # scale to [0, 256)
        tensor = (255 * tensor).type(torch.int32)
        # downsample bits
        tensor = tensor / self.div_val
        # add noise
        tensor = tensor.type(torch.float32) + torch.rand(tensor.shape)
        tensor = tensor / self.num_bins
        # now in range [0, 1.)
        return tensor - .5

    def __repr__(self):
        return self.__class__.__name__ + '(num_bits={})'.format(self._num_bits)
#
# def preprocess(x, n_bits_x=None, rand=True):
#     x = tf.cast(x, 'float32')
#     if n_bits_x < 8:
#         x = tf.floor(x / 2 ** (8 - n_bits_x))
#     n_bins = 2. ** n_bits_x
#     # add [0, 1] random noise
#     if rand:
#         x = x + tf.random_uniform(tf.shape(x), 0., 1.)
#     x = x / n_bins - .5
#     return x


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
