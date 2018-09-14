import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lib.toy_data as toy_data
import numpy as np

if __name__ == '__main__':
    batch_size = 10

    x = toy_data.inf_train_gen("rowimg",batch_size=batch_size)
    x = np.random.rand(500,500)
    sample_plot = plt.plot(x)
    plt.show(sample_plot)
