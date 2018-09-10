# Unfortunately, due to odeint_adjoint requiring a nn.Module to obtain its "parameters", we need a bit of a hack
# to support amortised parameters. A naive way would be to specify the odefunc as a function of x and the parameters
# of the encoder Q, but this would require calling Q(x) at every evaluation.
# Instead, we can compute the amortised parameters once, detach it from the graph and set them to be parameters
# of a newly created nn.Module. In the backward pass, we then stitch the encoder and decoder together manually.

# Given a stochastic encoder Q and a minibatch of samples x,
#  1. Get z_0, params = Q(x)
#  2. params_ = params.detach()
#  2. z_0_ = z_0.detach()
#  3. Construct ODEfunc(params), a new nn.Module.
#  4. Do odeint, decoder, compute loss.
#  5. Call grad_z0, grad_params = torch.autograd.grad(loss, [z_0_, params_], only_inputs=False)
#  6. Call torch.autograd.backward([z_0, params], [grad_z0, grad_params])
