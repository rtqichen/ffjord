import collections
import torch
import torch.nn as nn


class _RegularizationWrapper(nn.Module):
    def __init__(self, odefunc):
        super(_RegularizationWrapper, self).__init__()
        self.odefunc = odefunc

    def reset(self):
        self.regularization_loss = 0.

    def forward(self, t, y):
        return self.odefunc(t, y)

    def regularization_loss(self):
        return self.regularization_loss

    @property
    def _e(self):
        return self.odefunc._e

    @_e.setter
    def _e(self, value):
        self.odefunc._e = value

    @property
    def _num_evals(self):
        return self.odefunc._num_evals

    @_num_evals.setter
    def _num_evals(self, value):
        self.odefunc._num_evals = value


class L2Regularization(_RegularizationWrapper):
    def forward(self, t, y):
        dy = self.odefunc(t, y)
        self.regularization_loss += torch.mean(dy[1]**2)


class DirectionalL2Regularization(_RegularizationWrapper):
    def forward(self, t, y):
        dy = self.odefunc(t, y)
        directional_dy = torch.autograd.grad(dy, y, dy, create_graph=True)
        self.regularization_loss += torch.mean(directional_dy[1]**2)


class RegularizationsContainer(_RegularizationWrapper):
    def __init__(self, odefunc, dict_of_regularizations):
        """
        Args:
            odefunc: A callable with arguments (t, y)
            dict_of_regularizations: A dictionary of `_RegularizationWrapper` classes
                as keys and coefficients as values.
        """
        super(RegularizationsContainer, self).__init__()
        self.regularizations = collections.OrderedDict(dict_of_regularizations)
        self.wrapped_odefunc = odefunc
        for reg_wrapper in reversed(self.regularizations.keys()):
            self.wrapped_odefunc = reg_wrapper(self.wrapped_odefunc)

    def forward(self, t, y):
        return self.wrapped_odefunc(t, y)

    def regularization_loss(self):
        loss = 0.
        wrapped_odefunc = self.wrapped_odefunc
        for coeff in self.regularizations.values():
            loss += wrapped_odefunc.regularization_loss * coeff
            wrapped_odefunc.reset()
            wrapped_odefunc = wrapped_odefunc.odefunc
        return loss
