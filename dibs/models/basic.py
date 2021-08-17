

import time
import numpy as np
import scipy
import tqdm
from scipy.stats import multivariate_normal
import igraph as ig
import itertools
import torch

import jax.numpy as jnp
from jax import random


class BasicModel:
    """
    Basic observational model
    Given 
        p(G)

    Implements

        p(theta | G)
        p(x | theta, G)
    
    """

    def __init__(self, *, g_dist, verbose=False):
        super(BasicModel, self).__init__()
        
        self.verbose = verbose
        self.g_dist = g_dist
       

    def sample_parameters(self, *, g):
        """Samples parameters given igraph.Graph g
        For each variable i, sample parameters for every possible state of parents
        Returns:
            theta 
        """
        raise NotImplementedError
    

    def sample_obs(self, *, n_samples, g, theta, toporder=None):
        """Samples `n_samples` observations given g and theta
            n_samples : int
            g : graph
            theta : [n_edges]
        Returns:
            x : [n_samples, n_vars] 
        """

        raise NotImplementedError
        
        
    def log_prob_parameters(self, *, theta, g):
        """Computes p(theta | G)"""

        raise NotImplementedError

    def log_likelihood(self, *, x, theta, g):
        """Computes p(x | theta, G)"""

        raise NotImplementedError
        

    def log_marginal_likelihood_given_g(self, *, g, x):
        """Computes log p(x | G) 
            x : [n_samples, n_vars]
            g: graph
        """

        raise NotImplementedError
       

    def log_marginal_likelihood(self, *, x, all_g, z_g=None):
        """Computes log p(x) in closed form using conjugacy properties of Dirichlet-Categorical
            x : [n_samples, n_vars]
            all_g : list of all possible igraph.Graph objects in domain; is exhaustively summed over
        """

        # log p(x, G)
        log_prob_obs_g = np.zeros(len(all_g))

        # normalizing constant for log p(G) using exhaustive normalization
        if z_g is None:
            z_g = self.g_dist.log_normalization_constant(all_g=all_g)

        # log p(x, G)
        for i, g in enumerate(tqdm.tqdm(all_g, desc='p(X) log_marginal_likelihood', disable=not self.verbose)):

            # log p(x, G) = log (p(G)/Z) + log p(x | G)
            log_prob_obs_g[i] = self.g_dist.unnormalized_log_prob(g=g) - z_g \
                + self.log_marginal_likelihood_given_g(g=g, x=x)

        # log p(x) = log(sum_G exp(log p(x, G)))
        return scipy.special.logsumexp(log_prob_obs_g)

    def log_posterior_graph_given_obs(self, *, g, x, log_marginal_likelihood, z_g):
        """Computes p(G | D) given the previously computed normalization constant
            x : [..., n_vars]
            i : int (graph)
        """

        log_prob_g = self.g_dist.unnormalized_log_prob(g=g) - z_g
        log_marginal_likelihood_given_g = self.log_marginal_likelihood_given_g(
            g=g, x=x)
        return log_prob_g + log_marginal_likelihood_given_g - log_marginal_likelihood

    ####
    # Monte Carlo Integration to validate (closed-form computation) of marginal likelihood
    ####

    def log_prob_parameters_mc(self, *, key, theta, n_samples=3e4):
        """Approximates p(theta) using Monte Carlo integration
            theta : parameters
        """

        logliks = []
        for tt in range(int(n_samples)):

            # sample from p(G)
            key, subk = random.split(key)
            g = self.g_dist.sample_G(key=subk)

            # evaluate log prob p(theta | G)
            logliks.append(self.log_prob_parameters(theta=theta, g=g))

            # print
            if not tt % int(n_samples / 1000) and tt > 0:
                curr = scipy.special.logsumexp(
                    np.array(logliks[:tt + 1]) - np.log(tt + 1))
                print(f'iter = {tt}: log p(theta | G) [MC] = {curr}', end='\r')

        log_prob_obs = scipy.special.logsumexp(
            np.array(logliks) - np.log(n_samples))
        return log_prob_obs

    def log_marginal_likelihood_given_g_mc(self, *, key, x, g, n_samples=3e4):
        """Approximates p(x | G) using Monte Carlo integration
            x : [n_samples, n_vars]
            g : graph
        """

        logliks = []
        for tt in range(int(n_samples)):

            # sample from p(theta | G)
            key, subk = random.split(key)
            theta = self.sample_parameters(key=subk, g=g)

            # evaluate likelihood log p(X | theta, G)
            logliks.append(self.log_likelihood(x=x, theta=theta, g=g))

            # print
            if not tt % int(n_samples / 1000) and tt > 0:
                curr = scipy.special.logsumexp(
                    np.array(logliks[:tt + 1]) - np.log(tt + 1))
                print(f'iter = {tt}: log p(X | G) [MC] = {curr}', end='\r')

        log_prob_obs = scipy.special.logsumexp(
            np.array(logliks) - np.log(n_samples))
        return log_prob_obs

    def log_marginal_likelihood_mc(self, *, key, x, n_samples=3e4):
        """Approximates normalization constant p(x) using Monte Carlo integration
            x : [n_samples, n_vars]
        """

        logliks = []
        for tt in range(int(n_samples)):

            # sample from p(G, theta) = p(G) p(theta | G)
            key, subk = random.split(key)
            g = self.g_dist.sample_G(key=subk)

            key, subk = random.split(key)
            theta = self.sample_parameters(key=subk, g=g)

            # evaluate likelihood log p(X | theta, G)
            logliks.append(self.log_likelihood(x=x, theta=theta, g=g))

            # print
            if not tt % int(n_samples / 1000) and tt > 0:
                curr = scipy.special.logsumexp(
                    (logliks[:tt + 1] - np.log(tt + 1)))
                print(f'iter = {tt}: log p(X) [MC] = {curr}', end='\r')
        print()
        log_prob_obs = scipy.special.logsumexp((logliks - np.log(n_samples)))
        return log_prob_obs
