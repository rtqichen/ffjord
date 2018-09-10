import torch
from train_misc import build_model_tabular
from .VAE import VAE


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

        return x_mean, z_mu, z_var, delta_logp.view(-1), z0, zk
