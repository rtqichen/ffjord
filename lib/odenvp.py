import torch
import torch.nn as nn
import lib.layers as layers
import lib.layers.diffeq_layers as diffeq_layers
import lib.layers.wrappers.cnf_regularization as reg_lib
from lib.spectral_norm import spectral_norm


class ODENVP(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
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
        n_resblocks=2,
        multiplier=1,
        bn=True,
        bn_lag=0.,
        intermediate_dim=32,
        squash_input=True,
        alpha=0.05,
        l2_coeff=0.,
        dl2_coeff=0.,
        spectral_norm=False,
    ):
        super(ODENVP, self).__init__()
        self.n_scale = min(n_scale, self._calc_n_scale(input_size))
        self.n_resblocks = n_resblocks
        self.multiplier = multiplier
        self.intermediate_dim = intermediate_dim
        self.squash_input = squash_input
        self.alpha = alpha

        self._create_regularization_fns(l2_coeff, dl2_coeff)

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)
        if spectral_norm:
            self._add_spectral_norm()

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    idim=self.intermediate_dim,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=layers.LogitTransform(self.alpha)  # input transform
                    if self.squash_input and i == 0 else None,
                    n_resblocks=self.n_resblocks,
                    penult_multiplier=self.multiplier,
                    cnf_regularization_fns=self.regularization_fns
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _create_regularization_fns(self, l2_coeff, dl2_coeff):
        regularization_fns = []
        regularization_coeffs = []
        if l2_coeff != 0:
            regularization_coeffs.append(l2_coeff)
            regularization_fns.append(reg_lib.l2_regularzation_fn)
        if dl2_coeff != 0:
            regularization_coeffs.append(dl2_coeff)
            regularization_fns.append(reg_lib.directional_l2_regularization_fn)
        self.regularization_coeffs = tuple(regularization_coeffs)
        self.regularization_fns = tuple(regularization_fns)

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

    def _add_spectral_norm(self):
        def recursive_apply_sn(parent_module):
            for child_name in list(parent_module._modules.keys()):
                child_module = parent_module._modules[child_name]
                classname = child_module.__class__.__name__
                if classname.find('Conv') != -1 and 'weight' in child_module._parameters:
                    del parent_module._modules[child_name]
                    parent_module.add_module(child_name, spectral_norm(child_module, 'weight'))
                else:
                    recursive_apply_sn(child_module)

        recursive_apply_sn(self)

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

    def forward(self, x, logpx=None, reverse=False, include_regularition=False):
        if reverse:
            return self._generate(x, logpx)
        else:
            return self._logdensity(x, logpx)

    def _logdensity(self, x, logpx=None):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        for idx in range(len(self.transforms)):
            x, _logpx = self.transforms[idx].forward(x, _logpx)
            if idx < len(self.transforms) - 1:
                d = x.size(1) // 2
                x, factor_out = x[:, :d], x[:, d:]
            else:
                # last layer, no factor out
                factor_out = x
            out.append(factor_out)
        return tuple(out) if logpx is None else (tuple(out), _logpx)

    def _generate(self, zs, logpz=None):
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        z_prev, _logpz = self.transforms[-1](zs[-1], _logpz, reverse=True)
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            z_prev, _logpz = self.transforms[idx](z_prev, _logpz, reverse=True)
        return z_prev if logpz is None else (z_prev, _logpz)


class StackedCNFLayers(layers.SequentialFlow):
    def __init__(
        self,
        initial_size,
        idim=32,
        squeeze=True,
        init_layer=None,
        n_resblocks=2,
        penult_multiplier=1,
        bn=True,
        bn_lag=0.,
        cnf_regularization_fns=None,
    ):
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size, multiplier=1):
            return layers.ODEfunc(size, diffeq_layers.ResNet(size[0], idim, n_resblocks=n_resblocks * multiplier))

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            chain += [
                layers.CNF(_make_odefunc(initial_size), cnf_regularization_fns),
                layers.MovingBatchNorm2d(initial_size[0], bn_lag=bn_lag),
                layers.SqueezeLayer(2),
                layers.CNF(_make_odefunc(after_squeeze_size), cnf_regularization_fns),
                layers.MovingBatchNorm2d(after_squeeze_size[0], bn_lag=bn_lag),
            ]
        else:
            chain += [
                layers.CNF(_make_odefunc(initial_size, penult_multiplier), cnf_regularization_fns),
                layers.MovingBatchNorm2d(initial_size[0], bn_lag=bn_lag)
            ]

        super(StackedCNFLayers, self).__init__(chain)
