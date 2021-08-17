import time
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import igraph as ig
import itertools

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_update

from .basic import BasicModel


class BDeu(BasicModel):
    """
    Dirichlet-Categorical model (discrete)
    
    Each variable distributed as a Categorical with different Dirichlet parameters depending on the state of its parents.
    Assumes the BDeu setting (Bayesian Dirichlet likelihood equivalent uniform), i.e. that Dirichlet hyperparameter alpha_ijk is constant.
    """

    def __init__(self, *, g_dist, n_categories, alpha=1.0, verbose=False):
        super(BDeu, self).__init__(g_dist=g_dist, verbose=verbose)

        self.alpha = alpha  # "equivalent sample size"; default is 1

        # [n_vars]; n_categories[i] indicates the number of categories of variable i
        self.n_categories = n_categories

    def sample_parameters(self, *, key, g):
        """Samples parameters given igraph.Graph g
        For each variable i, sample parameters for every possible state of parents
        Returns:
            theta 
        """
        n_vars = len(g.vs)
        theta = []

        for j in range(n_vars):

            parent_edges = jnp.array(g.incident(j, mode='in'), dtype=jnp.int32)
            parents = jnp.array(list(g.es[e].source for e in parent_edges), dtype=jnp.int32)

            # number of parent states
            r = self.n_categories[j]
            n_ps = self.n_categories[parents].prod()
            alph = self.alpha / (r * n_ps)  # BDeu assumption

            key, subk = random.split(key)
            th = random.dirichlet(subk, alpha=alph * jnp.ones(r), shape=(n_ps,))

            # params[j][k][l] = p(X_j = l | X_parents_j = k)
            theta.append(th)

        return theta

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None):
        """Samples `n_samples` observations given g and theta
            n_samples : int
            g : graph
            theta : [n_edges]
        Returns:
            x : [n_samples, n_vars] 
        """
        if toporder is None:
            toporder = g.topological_sorting()

        n_vars = len(g.vs)
        x = jnp.zeros((n_samples, n_vars), dtype=jnp.int32)

        # ancestral sampling
        for j in toporder:

            parent_edges = jnp.array(g.incident(j, mode='in'), dtype=jnp.int32)
            parents = jnp.array(list(g.es[e].source for e in parent_edges), dtype=jnp.int32)

            # `all_states`:  array listing all possible parent states given number of possible categories for each parent
            # [n_states, n_parents]
            all_states = jnp.array(sorted(itertools.product(
                *[range(n) for n in self.n_categories[parents]])))
            n_states, n_parents = all_states.shape

            # `x_pa_state_idx` are indices of observed parent states in list `all_states`
            # [n_states]
            x_pa = x[..., parents]
            x_pa_tiled = jnp.tile(jnp.expand_dims(
                x_pa, axis=1), (1, n_states, 1))
            x_pa_state_idx = jnp.all(
                x_pa_tiled - all_states == 0, axis=2).argmax(axis=1)

            # retrieve correct parameter for observed parent state s
            thetas = theta[j][x_pa_state_idx]

            # sample state of variable j based on parent state parameters (use jax to support batched sampling)
            key, subk = random.split(key)
            x = index_update(x, index[..., j], random.categorical(subk, jnp.log(thetas)))

        return x

    def log_prob_parameters(self, *, theta, g):
        """p(theta | G). Assumes Dirichlet distribution for each possible parent state"""

        n_vars = len(g.vs)
        logprob = 0.0

        for j in range(n_vars):

            parent_edges = jnp.array(g.incident(j, mode='in'), dtype=jnp.int32)
            parents = jnp.array(list(g.es[e].source for e in parent_edges), dtype=jnp.int32)


            # number of parent states
            r = self.n_categories[j]
            n_ps = self.n_categories[parents].prod()
            alph = self.alpha / (r * n_ps)  # BDeu assumption

            # logprob of every possible parent state
            for k in range(n_ps):
                logprob += scipy.stats.dirichlet.logpdf(
                    x=theta[j][k], alpha=alph * jnp.ones(r))

        return logprob

    def log_likelihood(self, *, x, theta, g):
        """Computes p(x | theta, G). Assumes Categorical distribution for any given observation
            x : [..., n_vars]
            g : graph
            theta : [n_edges]
        """
        n_samples, n_vars = x.shape
        log_lik = 0.0

        for j in range(n_vars):

            parent_edges = jnp.array(g.incident(j, mode='in'), dtype=jnp.int32)
            parents = jnp.array(list(g.es[e].source for e in parent_edges), dtype=jnp.int32)

            # `all_states`:  array listing all possible parent states given number of possible categories for each parent
            # [n_states, n_parents]
            all_states = jnp.array(sorted(itertools.product(
                *[range(n) for n in self.n_categories[parents]])))
            n_states, n_parents = all_states.shape

            # `x_pa_state_idx` are indices of observed parent states in list `all_states`
            # [n_states]
            x_pa = x[..., parents]
            x_pa_tiled = jnp.tile(jnp.expand_dims(
                x_pa, axis=1), (1, n_states, 1))
            x_pa_state_idx = jnp.all(
                x_pa_tiled - all_states == 0, axis=2).argmax(axis=1)

            # retrieve correct parameter for observed parent state s
            thetas = theta[j][x_pa_state_idx]

            # Categorical log likelihood of variable j based on parent state parameters
            log_lik += jnp.log(jnp.sum(thetas * jnp.eye(thetas.shape[1])[x[..., j]], axis=-1)).sum()
            
        return log_lik

    def log_marginal_likelihood_given_g(self, *, g, x):
        """Computes log p(x | G) in closed form using conjugacy properties of Dirichlet-Categorical
            x : [n_samples, n_vars]
            g: graph
        """
        n_samples, n_vars = x.shape
        logprob = 0.0

        for j in range(n_vars):

            parent_edges = jnp.array(g.incident(j, mode='in'), dtype=jnp.int32)
            parents = jnp.array(list(g.es[e].source for e in parent_edges), dtype=jnp.int32)

            # number of parent states
            r = self.n_categories[j]
            n_ps = self.n_categories[parents].prod()

            # `all_states`:  array listing all possible parent states given number of possible categories for each parent
            # [n_states, n_parents]
            all_states = jnp.array(sorted(itertools.product(
                *[range(n) for n in self.n_categories[parents]])))
            n_states, n_parents = all_states.shape

            # `x_pa_state_idx` are indices of observed parent states in list `all_states`
            # [n_states]
            x_pa = x[..., parents]
            x_pa_tiled = jnp.tile(jnp.expand_dims(
                x_pa, axis=1), (1, n_states, 1))
            x_pa_state_idx = jnp.all(
                x_pa_tiled - all_states == 0, axis=2).argmax(axis=1)

            co_occurrence = jnp.stack((x[..., j], x_pa_state_idx), axis=1)

            # co-occurence `counts` with shape [# possible states of j, # possible states of all parents of j]
            # counts[l, k] : how often (X[j] = l, X[parents[j]] = k) occurred in data, where k is index for parent state
            '''TODO still numpy'''
            counts = np.zeros((r, n_ps), dtype=np.int32)
            unique, c = np.unique(co_occurrence, axis=0, return_counts=True)
            counts[unique[:, 0], unique[:, 1]] = c
            counts_sum = jnp.array(counts.sum(0), dtype=jnp.int32)
            '''TODO still numpy'''

            # Closed-form Dirichlet-Categorical marginal likelihood
            # BDeu assumption: alpha = alpha_ / (r * n_ps)
            logprob += scipy.special.loggamma(self.alpha / n_ps) * n_ps
            logprob -= scipy.special.loggamma(self.alpha /
                                              n_ps + counts_sum).sum()

            logprob += scipy.special.loggamma(self.alpha /
                                              (r * n_ps) + counts).sum()
            logprob -= scipy.special.loggamma(self.alpha /
                                              (r * n_ps)) * (r * n_ps)

        return logprob




'''
+++++   JAX implementation +++++
'''


class BDeuJAX:
    """
    Dirichlet-Categorical model (discrete) as above but using JAX and adjacency matrix representation

    jit() and vmap() not yet implemented for this score as it tricky with indexing
    """

    def __init__(self, *, n_categories, alpha=1.0, verbose=False):
        super(BDeuJAX, self).__init__()

        self.alpha = alpha  # "equivalent sample size"; default is 1
        self.verbose = verbose

        # [n_vars]; n_categories[i] indicates the number of categories of variable i
        self.n_categories = n_categories

        # init
        self.init_jax_functions()

    def init_jax_functions(self):
        pass

    def log_marginal_likelihood_given_g(self, *, w, data):
        """Computes log p(x | G) in closed form using conjugacy properties of Dirichlet-Categorical
            data : [n_samples, n_vars]
            w:     [n_vars, n_vars]
        """
        n_samples, n_vars = data.shape

        logprob = 0.0

        for j in range(n_vars):

            parents = jnp.where(w[:, j] == 1)

            # number of parent states
            r = self.n_categories[j]
            n_ps = self.n_categories[parents].prod()

            # `all_states`:  array listing all possible parent states given number of possible categories for each parent
            # [n_states, n_parents]
            all_states = jnp.array(sorted(itertools.product(
                *[range(n) for n in self.n_categories[parents]])))
            n_states, n_parents = all_states.shape

            # `data_pa_state_idx` are indices of observed parent states in list `all_states`
            # [n_states]
            data_pa = data[..., parents]
            data_pa_tiled = jnp.tile(jnp.expand_dims(
                data_pa, axis=1), (1, n_states, 1))
            data_pa_state_idx = jnp.all(
                data_pa_tiled - all_states == 0, axis=2).argmax(axis=1)

            co_occurrence = jnp.stack((data[..., j], data_pa_state_idx), axis=1)

            # co-occurence `counts` with shape [# possible states of j, # possible states of all parents of j]
            # counts[l, k] : how often (X[j] = l, X[parents[j]] = k) occurred in data, where k is index for parent state

            '''TODO still numpy'''
            counts = np.zeros((r, n_ps), dtype=np.int32)
            unique, c = np.unique(co_occurrence, axis=0, return_counts=True)
            counts[unique[:, 0], unique[:, 1]] = c
            counts_sum = jnp.array(counts.sum(0), dtype=jnp.int32)
            '''TODO still numpy'''

            # Closed-form Dirichlet-Categorical marginal likelihood
            # BDeu assumption: alpha = alpha_ / (r * n_ps)
            logprob += gammaln(self.alpha / n_ps) * n_ps
            logprob -= gammaln(self.alpha / n_ps + counts_sum).sum()

            logprob += gammaln(self.alpha / (r * n_ps) + counts).sum()
            logprob -= gammaln(self.alpha / (r * n_ps)) * (r * n_ps)

        return logprob
