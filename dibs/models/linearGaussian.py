import scipy
from scipy.stats import multivariate_normal

import jax.numpy as jnp
from jax import random
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal

from .basic import BasicModel


class LinearGaussian(BasicModel):
    """
    Linear Gaussian BN model (continuous variables)
    corresponding to linear structural equation model (SEM) with Gaussian noise

    Each variable distributed as Gaussian with mean being the linear combination of its parents 
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    """

    def __init__(self, *, g_dist, obs_noise, mean_edge, sig_edge, verbose=False):
        super(LinearGaussian, self).__init__(g_dist=g_dist, verbose=verbose)

        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge

    def sample_parameters(self, *, key, g):
        """Samples parameters given graph g, here corresponding to edge weights
        Args:
            g (igraph.Graph): grpah
            key: rng key

        Returns:
            theta : [n_vars, n_vars] to have consistent shape
        """
        key, subk = random.split(key)
        theta = self.mean_edge + self.sig_edge * random.normal(subk, shape=(len(g.vs), len(g.vs)))
        return theta
    
    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv={}):
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
    
    def log_prob_parameters(self, *, theta, g):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 
        In the linear Gaussian model, g does not matter.
        
        Args:
            g (igraph.Graph): graph
            theta : parameters
        
        Returns:
            [1, ]
        """
        logprob = 0.0
        # done analogously to `sample_obs` to ensure indexing into theta matrix 
        # is consistent and not silently messed up by e.g. ig.Graph vertex index naming
        for j in range(len(g.vs)):
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)
            if parents:
                logprob += scipy.stats.norm.logpdf(x=theta[jnp.array(parents), j], loc=self.mean_edge, scale=self.sig_edge).sum()

        # assert(logprob == jnp.sum(graph_to_mat(g) * scipy.stats.norm.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge)).item())
        return logprob


    def log_likelihood(self, *, x, theta, g):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise) distribution for any given observation
        
        Args:
            g (igraph.Graph): graph
            theta : parameters
            x : [n_samples, d] 

        Returns:
            [1, ]
        """
        n_vars = x.shape[-1]    
        logp = 0.0

        for j in range(n_vars):
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)
            
            if parents:
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j]
            else:
                mean = jnp.zeros(x.shape[0])

            # since observations iid, faster not to use multivariate_normal
            logp += scipy.stats.norm.logpdf(x=x[..., j], loc=mean, scale=jnp.sqrt(self.obs_noise)).sum()

        return logp

    def log_marginal_likelihood_given_g_single(self, g, x, j):
        """Computes log p(x | G) in closed form for a single node 
        using conjugacy properties of Gaussian

        Args:
            g (igraph.Graph): graph
            x : [n_samples, d] 
            j (int): node index for score

        Returns:
            [1, ]
        """
        n_samples, n_vars = x.shape

        parent_edges = g.incident(j, mode='in')
        parents = list(g.es[e].source for e in parent_edges)
        n_parents = len(parents)
        
        # mean
        mean_theta_j = self.mean_edge * jnp.ones(n_parents)
        mean_j = x[..., parents] @ mean_theta_j

        # cov
        # Note: `cov_j` is a NxN cov matrix, which can be huge for large N
        cov_theta_j = self.sig_edge ** 2.0 * jnp.eye(n_parents)
        cov_j = self.obs_noise * jnp.eye(n_samples) + \
            x[..., parents] @ cov_theta_j @ x[..., parents].T

        # log prob
        return multivariate_normal.logpdf(x=x[..., j], mean=mean_j, cov=cov_j)


    def log_marginal_likelihood_given_g(self, g, x):
        """Computes log p(x | G) in closed form using properties of Gaussian

        Args:
            g (igraph.Graph): graph
            x : [n_samples, d] 

        Returns:
            [1, ]
        """
        _, n_vars = x.shape
        logp = 0.0
        for j in range(n_vars):
            logp += self.log_marginal_likelihood_given_g_single(g=g, x=x, j=j)
        return logp


"""	
+++++   JAX implementation +++++	
"""


class LinearGaussianJAX:
    """	
    LinearGaussianas above but using JAX and adjacency matrix representation	
    jit() and vmap() not yet implemented for this score as it tricky with indexing	
    """

    def __init__(self, *, obs_noise, mean_edge, sig_edge, verbose=False):
        super(LinearGaussianJAX, self).__init__()

        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge
        self.verbose = verbose


    def get_theta_shape(self, *, n_vars):
        """PyTree of parameter shape"""
        return jnp.array((n_vars, n_vars))


    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G) 
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : [n_particles, n_vars, n_vars]
        """
        key, subk = random.split(key)
        
        if batch_size == 0:
            theta = self.mean_edge + self.sig_edge * random.normal(subk, shape=(n_particles, *self.get_theta_shape(n_vars=n_vars)))
        else:
            theta = self.mean_edge + self.sig_edge * random.normal(subk, shape=(batch_size, n_particles, *self.get_theta_shape(n_vars=n_vars)))

        return theta


    def log_marginal_likelihood_given_g(self, *, w, data, interv_targets):
        """Computes log p(x | G) in closed form using properties of Gaussian

        Args:
            data : [n_samples, n_vars]	
            w:     [n_vars, n_vars]	

        Returns:
            [1, ]
        """
        n_samples, n_vars = data.shape

        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                interv_targets,
                0.0,
                self.log_marginal_likelihood_given_g_j(jnp.arange(n_vars), w, data)
            )
        )
        # prev code without interventions
        # return jnp.sum(self.log_marginal_likelihood_given_g_j(jnp.arange(n_vars), w, data))


    def log_prob_parameters(self, *, theta, w):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 
        In the linear Gaussian model, g does not matter.
        
        Args:
            theta: [n_vars, n_vars]
            w: [n_vars, n_vars]
            
        Returns:
            [1, ]
        """

        return jnp.sum(w * jax_normal.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge))


    def log_likelihood(self, *, data, theta, w, interv_targets):
        """Computes p(x | theta, G). Assumes N(mean_obs, obs_noise) distribution for any given observation
            data :          [n_observations, n_vars]
            theta:          [n_vars, n_vars]
            w:              [n_vars, n_vars]
            interv_targets: [n_vars, ]
        """
        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data, loc=data @ (w * theta), scale=jnp.sqrt(self.obs_noise))
            )
        )
        # prev code without interventions
        # return jnp.sum(jax_normal.logpdf(x=data, loc=data @ (w * theta), scale=jnp.sqrt(self.obs_noise)))


