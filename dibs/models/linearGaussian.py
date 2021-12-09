import jax.numpy as jnp
from jax import random
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal


class LinearGaussian:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise

    Each variable distributed as Gaussian with mean being the linear combination of its parents 
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    The noise variance at each node is equal by default, which implies the causal structure is identifiable.
    """

    def __init__(self, *, graph_dist, obs_noise, mean_edge, sig_edge):
        super(LinearGaussian, self).__init__()

        self.graph_dist = graph_dist
        self.n_vars = graph_dist.n_vars
        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge

        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)


    def get_theta_shape(self, *, n_vars):
        """PyTree of parameter shape"""
        return jnp.array((n_vars, n_vars))


    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G)
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : [batch_size, n_particles, n_vars, n_vars] with dimensions equal to 0 dropped
        """
        shape = (batch_size, n_particles, *self.get_theta_shape(n_vars=n_vars))
        theta = self.mean_edge + self.sig_edge * random.normal(key, shape=tuple(d for d in shape if d != 0))
        return theta

    
    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        """Samples `n_samples` observations given graph and parameters
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : parameters
            interv: {intervened node : clamp value}

        Returns:
            x : [n_samples, d] 
        """
        if interv is None:
            interv = {}
        if toporder is None:
            toporder = g.topological_sorting()

        x = jnp.zeros((n_samples, len(g.vs)))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, len(g.vs)))

        # ancestral sampling
        for j in toporder:

            # intervention
            if j in interv.keys():
                x = index_update(x, index[:, j], interv[j])
                continue
            
            # regular ancestral sampling
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j]
                x = index_update(x, index[:, j], mean + z[:, j])
            else:
                x = index_update(x, index[:, j], z[:, j])

        return x
    
    """
    The following functions need to be functionally pure and @jit-able
    """

    def log_prob_parameters(self, *, theta, g):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 
        In the linear Gaussian model, g does not matter.
        
        Arguments:
            theta: [n_vars, n_vars]
            g: [n_vars, n_vars]
            
        Returns:
            [1, ]
        """
        return jnp.sum(g * jax_normal.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge))


    def log_likelihood(self, *, x, theta, g, interv_targets):
        """Computes p(x | theta, G). Assumes N(mean_obs, obs_noise) distribution for any given observation

        Arguments:
            x :             [n_observations, n_vars]
            theta:          [n_vars, n_vars]
            g:              [n_vars, n_vars]
            interv_targets: [n_vars, ]

        Returns:
            [1, ]
        """
        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=x, loc=x @ (g * theta), scale=jnp.sqrt(self.obs_noise))
            )
        )


    """
    Distributions used by DiBS for inference:  prior and joint likelihood 
    """

    def log_graph_prior(self, g_prob):
        """ log p(G)

        Arguments:
            g_prob: [n_vars, n_vars] of edge probabilities in G

        Returns:
            [1, ]
        """
        return self.graph_dist.unnormalized_log_prob_soft(soft_g=g_prob)


    def observational_log_joint_prob(self, g, theta, x, rng):
        """ log p(D, theta | G)  =  log p(D | G, theta) + log p(theta | G)

        Arguments:
           g: [n_vars, n_vars] graph adjacency matrix
           theta: PyTree
           x: [n_observations, n_vars] observational data
           rng

        Returns:
           [1, ]
        """
        log_prob_theta = self.log_prob_parameters(g=g, theta=theta)
        log_likelihood = self.log_likelihood(g=g, theta=theta, x=x, interv_targets=self.no_interv_targets)
        return log_prob_theta + log_likelihood

