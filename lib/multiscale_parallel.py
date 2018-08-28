import torch
import torch.nn as nn
import lib.layers as layers
from lib.layers.odefunc import ODEnet
import numpy as np


class MultiscaleParallelCNF(nn.Module):
    """
    CNF model for image data.

    Squeezes the input into multiple scales, applies different conv-nets at each scale
    and adds the resulting gradients

    Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(
        self,
        input_size,
        n_scale=float('inf'),
        n_blocks=1,
        intermediate_dims=(32,),
        alpha=-1,
        time_length=1.,
    ):
        super(MultiscaleParallelCNF, self).__init__()
        print(input_size)
        self.n_scale = min(n_scale, self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dims = intermediate_dims
        self.alpha = alpha
        self.time_length = time_length

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        transforms.append(
            ParallelCNFLayers(
                initial_size=(c, h, w),
                idims=self.intermediate_dims,
                init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform()),
                n_blocks=self.n_blocks,
                time_length=self.time_length
            )
        )
        return nn.ModuleList(transforms)

    def get_regularization(self):
        if len(self.regularization_fns) == 0:
            return None

        acc_reg_states = tuple([0.] * len(self.regularization_fns))
        for module in self.modules():
            if isinstance(module, layers.CNF):
                acc_reg_states = tuple(
                    acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states())
                )
        return sum(state * coeff for state, coeff in zip(acc_reg_states, self.regularization_coeffs))

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._generate(x, logpx)
        else:
            return self._logdensity(x, logpx)

    def _logdensity(self, x, logpx=None):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        for idx in range(len(self.transforms)):
            x, _logpx = self.transforms[idx].forward(x, _logpx)

        return x if logpx is None else (x, _logpx)

    def _generate(self, z, logpz=None):
        _logpz = torch.zeros(z.shape[0], 1).to(z) if logpz is None else logpz
        for idx in reversed(range(len(self.transforms))):
            z, _logpz = self.transforms[idx](z, _logpz, reverse=True)
        return z if logpz is None else (z, _logpz)



class ParallelSumModules(nn.Module):
    def __init__(self, models):
        super(ParallelSumModules, self).__init__()
        self.models = nn.ModuleList(models)
        self.cpu = not torch.cuda.is_available()

    def forward(self, t, y):
        out = sum(model(t, y) for model in self.models)
        return out


class ParallelCNFLayers(layers.SequentialFlow):
    def __init__(
        self,
        initial_size,
        idims=(32,),
        scales=4,
        init_layer=None,
        n_blocks=1,
        time_length=1.,
    ):
        strides = tuple([1] + [1 for _ in idims])
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        get_size = lambda s: (initial_size[0] * (4**s), initial_size[1] // (2**s), initial_size[2] // (2**s))

        def _make_odefunc():
            nets = [ODEnet(idims, get_size(scale), strides, True, layer_type="concat", num_squeeze=scale)
                    for scale in range(scales)]
            net = ParallelSumModules(nets)
            f = layers.ODEfunc(net)
            return f

        chain += [layers.CNF(_make_odefunc(), T=time_length) for _ in range(n_blocks)]

        super(ParallelCNFLayers, self).__init__(chain)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnfs = MultiscaleParallelCNF((13, 3, 32, 32)).to(device)
    t = torch.randn(13, 3, 32, 32).to(device)
    out = cnfs(t, logpx=None)
    print("done")