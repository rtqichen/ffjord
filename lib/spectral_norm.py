import torch
import torch.nn as nn


class SpectralNorm(object):
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_unspec')
        shape = weight.shape
        weight = weight.view(weight.shape[0], -1)
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        u_weight = u @ weight
        weight_v = weight @ v
        u = weight_v / torch.norm(weight_v)
        v = u_weight / torch.norm(u_weight)
        setattr(module, self.name + '_u', u.detach())
        setattr(module, self.name + '_v', v.detach())
        return weight.view(shape) / (u @ weight @ v)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        u_shape, v_shape = weight.view(weight.shape[0], -1).shape

        # remove w from parameter list
        del module._parameters[name]

        # add u and v as buffers, which will be updated at every iteration.
        module.register_buffer(name + '_u', torch.randn(u_shape).to(weight))
        module.register_buffer(name + '_v', torch.randn(v_shape).to(weight))
        # rename the actual un-normalized weights.
        module.register_parameter(name + '_unspec', nn.Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return module

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        del module._parameters[self.name + '_unspec']
        module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)
    return module
