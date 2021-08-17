import time
import scipy
from scipy.stats import multivariate_normal
import igraph as ig

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
import jax.lax as lax
from jax.scipy.special import logsumexp, gammaln
from jax.ops import index, index_add, index_update
from jax.scipy.stats import multivariate_normal as jax_multivariate_normal

from .basic import BasicModel

from dibs.utils.func import sel


class GBNSampler:
    """
    Linear Gaussian BN parameter and observational sampler
    Follows GBN setup of 

    Cho 2016: Reconstructing Causal Biological Networks through Active Learning

    Given graph `g` the procedure samples parameters of GBN by 
        uniformly sampling each edge weight from union of `edge_interval`
        sampling the base level of each node from Normal(node_mean, node_sig)
    
    The observations are Gaussian with std dev `obs_sig`
    """

    def __init__(self, *, node_mean=0.0, node_sig=1.0, edge_interval=[(-3.0, -1.0), (1.0, 3.0)], obs_sig=0.1, verbose=False):
        super(GBNSampler, self).__init__()

        self.node_mean = jnp.array(node_mean)
        self.node_sig = jnp.array(node_sig)
        self.edge_interval = jnp.array(edge_interval)
        self.obs_sig = jnp.array(obs_sig)

        # preprocessing
        self.interval_len = self.edge_interval[:, 1] - self.edge_interval[:, 0]
        self.interval_prob = self.interval_len / self.interval_len.sum()

    def sample_parameters(self, *, key, g):
        """Samples parameters given graph `g` with N nodes and M edges

        Returns 3-tuple of
            node_mean:      [N, ]   mean of node
            obs_sig :       [N, ]   noise level at a given node
            edge_weight:    [M, ]   regression coefficient of edge
        """
        n_vars, n_edges = len(g.vs), len(g.es)

        # node means
        key, subk = random.split(key)
        node_mean = self.node_mean + self.node_sig * random.normal(subk, shape=(n_vars, ))

        # noise levels
        obs_sig = self.obs_sig * jnp.ones(n_vars)

        # regression coefficient of edge
        key, subk = random.split(key)
        interv_idx = random.choice(subk, len(self.interval_prob), 
            p=self.interval_prob, shape=(n_edges, ))  # which range
        interv = self.edge_interval[interv_idx]

        key, subk = random.split(key)
        edge_weight = random.uniform(subk, minval=interv[:, 0], maxval=interv[:, 1], shape=(n_edges, ))

        return (node_mean, obs_sig, edge_weight)
    
    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv={}):
        """Samples `n_samples` observations given index i of graph and theta
            key
            n_samples 
            g 
            theta : 3-tuple of (node_mean, obs_sig, edge_weight)
            interv: {intervened node : clamp value}
        Returns:
            x : [n_samples, n_vars] 
        """
        node_mean, obs_sig, edge_weight = theta

        if toporder is None:
            toporder = g.topological_sorting()

        x = jnp.zeros((n_samples, len(g.vs)))

        # noise levels
        key, subk = random.split(key)
        z = jnp.sqrt(obs_sig) * random.normal(subk, shape=(n_samples, len(g.vs)))

        # ancestral sampling
        for j in toporder:
            
            # intervention
            if j in interv.keys():
                x = index_update(x, index[:, j], interv[j])
                continue

            # regular ancestral sampling
            parent_edges = jnp.array(g.incident(j, mode='in'))
            parents = jnp.array(list(g.es[e].source for e in parent_edges))

            if parents.shape[0]:
                mean = node_mean[j] + x[:, parents] @ edge_weight[parent_edges]
            else:
                mean = node_mean[j]
            
            x = index_update(x, index[:, j], mean + z[:, j])

        return x
    
  
