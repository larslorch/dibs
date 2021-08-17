import jax.numpy as jnp
from jax import vmap
from jax import random
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal
from jax.tree_util import tree_map, tree_reduce


import jax.experimental.stax as stax
from jax.experimental.stax import Dense, Sigmoid, LeakyRelu, Relu, Tanh

from jax.nn.initializers import normal

from dibs.utils.graph import graph_to_mat
from dibs.utils.tree import tree_shapes


def DenseNoBias(out_dim, W_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer _without_ bias"""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, (W, )

    def apply_fun(params, inputs, **kwargs):
        W, = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def makeDenseNet(*, hidden_layers, sig_weight, sig_bias, bias=True, activation='relu'):
    """
    Generates functions defining a fully-connected NN
    with Gaussian initialized parameters

    Args:
        hidden_layers (list): list of ints specifying the dimensions of the hidden sizes
        sig_weight: std dev of weight initialization
        sig_bias: std dev of weight initialization
    
    Returns:
        stax.serial neural net object
    """

    # features: [hidden_layers[0], hidden_layers[0], ..., hidden_layers[-1], 1]
    if activation == 'sigmoid':
        f_activation = Sigmoid
    elif activation == 'tanh':
        f_activation = Tanh
    elif activation == 'relu':
        f_activation = Relu
    elif activation == 'leakyrelu':
        f_activation = LeakyRelu
    else:
        raise KeyError(f'Invalid activation function `{activation}`')

    modules = []
    if bias:
        for dim in hidden_layers:
            modules += [
                Dense(dim, W_init=normal(stddev=sig_weight),
                        b_init=normal(stddev=sig_bias)),
                f_activation
            ]
        modules += [Dense(1, W_init=normal(stddev=sig_weight),
                            b_init=normal(stddev=sig_bias))]
    else:
        for dim in hidden_layers:
            modules += [
                DenseNoBias(dim, W_init=normal(stddev=sig_weight)),
                f_activation
            ]
        modules += [DenseNoBias(1, W_init=normal(stddev=sig_weight))]

    return stax.serial(*modules)
    

class DenseNonlinearGaussianJAX:
    """	
    Non-linear Gaussian BN with interactions modeled by a fully-connected neural net
    See: https://arxiv.org/abs/1909.13189    
    """

    def __init__(self, *, obs_noise, sig_param, hidden_layers, g_dist=None, verbose=False, activation='relu', bias=True):
        super(DenseNonlinearGaussianJAX, self).__init__()

        self.obs_noise = obs_noise
        self.sig_param = sig_param
        self.hidden_layers = hidden_layers
        self.g_dist = g_dist
        self.verbose = verbose
        self.activation = activation
        self.bias = bias

        # init single neural net function for one variable with jax stax
        self.nn_init_random_params, nn_forward = makeDenseNet(
            hidden_layers=self.hidden_layers, 
            sig_weight=self.sig_param,
            sig_bias=self.sig_param,
            activation=self.activation,
            bias=self.bias)
        
        # [?], [N, d] -> [N,]
        self.nn_forward = lambda theta, x: nn_forward(theta, x).squeeze(-1)
        
        # vectorize init and forward functions
        self.eltwise_nn_init_random_params = vmap(self.nn_init_random_params, (0, None), 0)
        self.double_eltwise_nn_init_random_params = vmap(self.eltwise_nn_init_random_params, (0, None), 0)
        self.triple_eltwise_nn_init_random_params = vmap(self.double_eltwise_nn_init_random_params, (0, None), 0)
        
        # [d2, ?], [N, d] -> [N, d2]
        self.eltwise_nn_forward = vmap(self.nn_forward, (0, None), 1)

        # [d2, ?], [d2, N, d] -> [N, d2]
        self.double_eltwise_nn_forward = vmap(self.nn_forward, (0, 0), 1)


    def get_theta_shape(self, *, n_vars):
        """ Returns tree shape of the parameters of the neural networks
        Args:
            n_vars

        Returns:
            PyTree of parameter shape
        """
        
        dummy_subkeys = jnp.zeros((n_vars, 2), dtype=jnp.uint32)
        _, theta = self.eltwise_nn_init_random_params(dummy_subkeys, (n_vars, )) # second arg is `input_shape` of NN forward pass

        theta_shape = tree_shapes(theta)
        return theta_shape

    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G) 
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : PyTree with leading dimension of `n_particles`
        """

        if batch_size == 0:
            subkeys = random.split(key, n_particles * n_vars).reshape(n_particles, n_vars, -1)
            _, theta = self.double_eltwise_nn_init_random_params(subkeys, (n_vars, ))
        else:
            subkeys = random.split(key, batch_size * n_particles * n_vars).reshape(batch_size, n_particles, n_vars, -1)
            _, theta = self.triple_eltwise_nn_init_random_params(subkeys, (n_vars, ))
            
        # to float64
        theta = tree_map(lambda arr: arr.astype(jnp.float64), theta)
        return theta

    def sample_parameters(self, *, key, g):
        """Samples parameters for neural network. Here, g is ignored.
        Args:
            g (igraph.Graph): graph
            key: rng

        Returns:
            theta : list of (W, b) tuples, dependent on `hidden_layers`
        """
        n_vars = len(g.vs)

        subkeys = random.split(key, n_vars)
        _, theta = self.eltwise_nn_init_random_params(subkeys, (n_vars, ))

        return theta


    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv={}):
        """
        Samples `n_samples` observations by doing single forward passes in topological order
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : PyTree of parameters
            interv: {intervened node : clamp value}

        Returns:
            x : [n_samples, d] 
        """

        # find topological order for ancestral sampling
        if toporder is None:
            toporder = g.topological_sorting()

        n_vars = len(g.vs)
        x = jnp.zeros((n_samples, n_vars))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, n_vars))

        g_mat = graph_to_mat(g)

        # ancestral sampling
        # for simplicity, does d full forward passes for simplicity, which avoids indexing into python list of parameters
        for j in toporder:

            # intervention
            if j in interv.keys():
                x = index_update(x, index[:, j], interv[j])
                continue

            # regular ancestral sampling
            parents = g_mat[:, j].reshape(1, -1)

            has_parents = parents.sum() > 0

            if has_parents:
                # [N, d] = [N, d] * [1, d] mask non-parent entries of j
                x_msk = x * parents

                # [N, d] full forward pass
                means = self.eltwise_nn_forward(theta, x_msk)

                # [N,] update j only
                x = index_update(x, index[:, j], means[:, j] + z[:, j])
            else:
                x = index_update(x, index[:, j], z[:, j])

        return x


    def log_prob_parameters(self, *, theta, w):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 

        Args:
            theta: parmeter PyTree
            w: adjacency matrix of graph [n_vars, n_vars]

        Returns:
            logprob [1,]
        """
        # compute log prob for each weight
        logprobs = tree_map(lambda leaf_theta: jax_normal.logpdf(x=leaf_theta, loc=0.0, scale=self.sig_param), theta)

        # mask logprobs of first layer weight matrix [0][0] according to graph
        # [d, d, dim_first_layer] = [d, d, dim_first_layer] * [d, d, 1]
        if self.bias:
            first_weight_logprobs, first_bias_logprobs = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None], first_bias_logprobs)
        else:
            first_weight_logprobs,  = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None],)

        # sum logprobs of every parameter tensor and add all up 
        return tree_reduce(jnp.add, tree_map(jnp.sum, logprobs))


    def log_likelihood(self, *, data, theta, w, interv_targets):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation
        
        Args:
            data: observations [N, d]
            theta: parameter PyTree
            w: adjacency matrix [n_vars, n_vars]
            interv_targets: boolean indicator of intervention locations [n_vars, ]
        
        Returns:
            logprob [1, ]
        """

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = data[None] * w.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j 
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data, loc=all_means, scale=jnp.sqrt(self.obs_noise))
            )
        )
       