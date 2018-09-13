from __future__ import print_function
import time
import numpy as np
from scipy.misc import logsumexp
from vae_lib.optimization.loss import calculate_loss_array


def calculate_likelihood(X, model, args, logger, S=5000, MB=500):

    # set auxiliary variables for number of training and test sets
    N_test = X.size(0)

    X = X.view(-1, *args.input_size)

    likelihood_test = []

    if S <= MB:
        R = 1
    else:
        R = S // MB
        S = MB

    end = time.time()
    for j in range(N_test):

        x_single = X[j].unsqueeze(0)

        a = []
        for r in range(0, R):
            # Repeat it for all training points
            x = x_single.expand(S, *x_single.size()[1:]).contiguous()

            x_mean, z_mu, z_var, ldj, z0, zk = model(x)

            a_tmp = calculate_loss_array(x_mean, x, z_mu, z_var, z0, zk, ldj, args)

            a.append(-a_tmp.cpu().data.numpy())

        # calculate max
        a = np.asarray(a)
        a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
        likelihood_x = logsumexp(a)
        likelihood_test.append(likelihood_x - np.log(len(a)))

        if j % 1 == 0:
            logger.info('Progress: {:.2f}% | Time: {:.4f}'.format(j / (1. * N_test) * 100, time.time() - end))
        end = time.time()

    likelihood_test = np.array(likelihood_test)

    nll = -np.mean(likelihood_test)

    if args.input_type == 'multinomial':
        bpd = nll / (np.prod(args.input_size) * np.log(2.))
    elif args.input_type == 'binary':
        bpd = 0.
    else:
        raise ValueError('invalid input type!')

    return nll, bpd
