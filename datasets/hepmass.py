import pandas as pd
import numpy as np
from collections import Counter
from os.path import join

import datasets


class HEPMASS:
    """
    The HEPMASS data set.
    http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        path = datasets.root + 'hepmass/'
        trn, val, tst = load_data_no_discrete_normalised_as_array(path)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data(path):

    data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"), index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"), index_col=False)

    return data_train, data_test


def load_data_no_discrete(path):
    """
    Loads the positive class examples from the first 10 percent of the dataset.
    """
    data_train, data_test = load_data(path)

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    return data_train, data_test


def load_data_no_discrete_normalised(path):

    data_train, data_test = load_data_no_discrete(path)
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_test


def load_data_no_discrete_normalised_as_array(path):

    data_train, data_test = load_data_no_discrete_normalised(path)
    data_train, data_test = data_train.as_matrix(), data_test.as_matrix()

    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    N = data_train.shape[0]
    N_validate = int(N * 0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test
