import torch
import torch.nn as nn

__all__ = ['CouplingLayer', 'MaskedCouplingLayer']


class CouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim=32, swap=False):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim), nn.Tanh(), nn.Linear(intermediate_dim, (d - self.d) * 2)
        )

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


class MaskedCouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim=32, mask_type='alternate', swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer('mask', sample_mask(d, mask_type, swap).view(1, d))
        self.net_scale = nn.Sequential(
            nn.Linear(self.d, intermediate_dim), nn.Tanh(), nn.Linear(intermediate_dim, self.d)
        )
        self.net_shift = nn.Sequential(
            nn.Linear(self.d, intermediate_dim), nn.ReLU(), nn.Linear(intermediate_dim, self.d)
        )

    def forward(self, x, logpx=None, reverse=False):

        scale = torch.sigmoid(self.net_scale(x * self.mask) + 2.)
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        logdetjac = torch.sum(torch.log(masked_scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y = x * masked_scale + masked_shift
            delta_logp = -logdetjac
        else:
            y = (x - masked_shift) / masked_scale
            delta_logp = logdetjac

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


def sample_mask(dim, mask_type, swap):
    if mask_type == 'alternate':
        # Index-based masking in MAF paper.
        mask = torch.zeros(dim)
        mask[::2] = 1
        if swap:
            mask = 1 - mask
        return mask
    elif mask_type == 'channel':
        # Masking type used in Real NVP paper.
        mask = torch.zeros(dim)
        mask[:dim // 2] = 1
        if swap:
            mask = 1 - mask
        return mask
    else:
        raise ValueError('Unknown mask_type {}'.format(mask_type))
