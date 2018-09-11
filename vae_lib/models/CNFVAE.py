import torch
import torch.nn as nn
from train_misc import build_model_tabular, set_cnf_options
import lib.layers as layers
from .VAE import VAE
import lib.layers.diffeq_layers as diffeq_layers
from lib.layers.odefunc import NONLINEARITIES


class CNFVAE(VAE):

    def __init__(self, args):
        super(CNFVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # CNF model
        self.cnf = build_model_tabular(args, args.z_size)

        # TODO: Amortized flow parameters

        if args.cuda:
            self.cuda()

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        # TODO: Amortized flow parameters

        return mean_z, var_z

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var = self.encode(x)

        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)

        zero = torch.zeros(x.shape[0], 1).to(x)
        zk, delta_logp = self.cnf(z0, zero)  # run model forward

        x_mean = self.decode(zk)

        return x_mean, z_mu, z_var, -delta_logp.view(-1), z0, zk


class AmortizedBiasODEnet(nn.Module):
    def __init__(self, hidden_dims, input_dim, layer_type="concat", nonlinearity="softplus"):
        super(AmortizedBiasODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]
        self.input_dim = input_dim

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_dim

        for dim_out in hidden_dims + (input_dim,):
            layer = base_layer(hidden_shape, dim_out)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])
            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y_am_bias):
        y, am_biases = y_am_bias[:, :self.input_dim], y_am_bias[:, self.input_dim:]
        am_biases_0 = am_biases
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            this_bias, am_biases = am_biases[:, :dx.size(1)], am_biases[:, dx.size(1):]
            dx = dx + this_bias
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        dx_am_biases = torch.cat([dx, 0. * am_biases_0], 1)
        return dx_am_biases


def build_amortized_model(args, z_dim, amortization_type="bias", regularization_fns=None):

    hidden_dims = tuple(map(int, args.dims.split("-")))
    diffeq_fn = {"bias": AmortizedBiasODEnet}[amortization_type]
    def build_cnf():
        diffeq = diffeq_fn(
            hidden_dims=hidden_dims,
            input_dim=z_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf()]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(z_dim, bn_lag=args.bn_lag)]
        bn_chain = [layers.MovingBatchNorm1d(z_dim, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model



class AmortizedBiasCNFVAE(VAE):

    def __init__(self, args):
        super(AmortizedBiasCNFVAE, self).__init__(args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # CNF model
        self.cnfs = nn.ModuleList([build_amortized_model(args, args.z_size, "bias") for _ in range(args.num_blocks)])

        hidden_dims = tuple(map(int, args.dims.split("-"))) + (args.z_size,)
        bias_size = sum(hidden_dims)
        self.q_am_biases = nn.ModuleList([nn.Linear(256, bias_size) for _ in range(args.num_blocks)])

        if args.cuda:
            self.cuda()

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        am_biases = [q_am_bias(h) for q_am_bias in self.q_am_biases]

        return mean_z, var_z, am_biases

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        z_mu, z_var, am_biases = self.encode(x)

        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)

        delta_logp = torch.zeros(x.shape[0], 1).to(x)
        z = z0
        for cnf, am_bias in zip(self.cnfs, am_biases):
            z_with_am_bias = torch.cat([z, am_bias], 1)  # add bias to ode state
            z_with_am_bias, delta_logp = cnf(z_with_am_bias, delta_logp)  # run model forward
            z = z_with_am_bias[:, :z.size(1)]  # remove bias from ode state

        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, -delta_logp.view(-1), z0, z
