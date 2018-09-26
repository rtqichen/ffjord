import numpy as np
import sklearn
import sklearn.datasets
from scipy.stats import norm


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    elif data == "rowimg":
        imagewidth = 200
        x = np.zeros([batch_size,imagewidth],dtype=np.float64)
        sig = 10.

        for sample in x:
            mean = np.random.random()*imagewidth
            for i in range(sample.shape[0]):
                sample[i] += norm.pdf(i,loc=mean,scale=sig)
        return x

    elif data == "rowimgsmol":
        imagewidth = 50
        x = np.zeros([batch_size,imagewidth],dtype=np.float64)
        sig = 5.

        for sample in x:
            mean = np.random.random()*imagewidth
            for i in range(sample.shape[0]):
                sample[i] += 7.*5.*norm.pdf(i,loc=mean,scale=sig)
                sample[i] += 0.01*np.random.randn(*sample[i].shape)
        return x
    elif data == "willrow":
        imagewidth = 50
        x = np.zeros([batch_size,imagewidth],dtype=np.float64)
        
        for sample in x:
            n = np.random.randint(1,3)
            for blob in range(n):
                l = np.random.randint(5,10)
                o = np.random.randint(0,imagewidth - l)
                sample[o:o+l] = 4 
        x += 0.1*np.random.randn(batch_size,imagewidth)
        return x
    elif data == "1d_density":
        # x = 0.5*np.random.randn(batch_size,1)+2.0
        x = np.random.uniform(0,3,size=(batch_size,1))
        # x += 0.01*np.random.randn(*x.shape)
        return x
    elif data == "1d_density_mix":
        ## 3 gauusians
        # x1 = np.random.randn(batch_size) / 1.5 - 3.0
        # x2 = np.random.randn(batch_size) / 2.0 + 3.0
        # x3 = np.random.randn(batch_size) / 2.2 + 3.0
        # xs = np.concatenate([x1[:,None],x2[:,None],x3[:,None]],1)
        # k = np.random.randint(0,3,batch_size)
        # x = xs[np.arange(batch_size),k]       x1 = np.random.randn(batch_size) / 1.5 - 3.0
        ## 2 gaussians    
        # x1 = np.random.randn(batch_size) / 2.0  - 3.0
        # x2 = np.random.randn(batch_size) / 2.0 + 3.0
        # xs = np.concatenate([x1[:,None],x2[:,None]],1)
        # k = np.random.randint(0,2,batch_size)
        # x = xs[np.arange(batch_size),k]
        ## 2 unis
        # x1 = np.random.uniform(1,3,size=batch_size)
        # x2 = np.random.uniform(-1,-3,size=batch_size)
        # xs = np.concatenate([x1[:,None],x2[:,None]],1)
        # k = np.random.randint(0,2,batch_size)
        # x = xs[np.arange(batch_size),k]

        ## Double Gaussians
        x1 = np.random.randn(batch_size)*np.sqrt(0.4)-2.8
        x2 = np.random.randn(batch_size)*np.sqrt(0.4)-0.9
        x3 = np.random.randn(batch_size)*np.sqrt(0.4)+2.
        xs = np.concatenate([x1[:,None],x2[:,None],x3[:,None]],1)
        k = np.random.randint(0,3,batch_size)
        x = xs[np.arange(batch_size),k]
        return x[:,None]

