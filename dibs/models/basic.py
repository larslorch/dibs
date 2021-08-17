

import numpy as np
import scipy
import tqdm


class BasicModel:
    """
    Basic observational model
    Given 
        p(G)
    implements
        p(theta | G)
        p(x | theta, G)
    
    """

    def __init__(self, *, g_dist, verbose=False):
        super(BasicModel, self).__init__()
        
        self.verbose = verbose
        self.g_dist = g_dist
       

    def sample_parameters(self, *, g):
        """Samples parameters given graph `g`
        Args:
            g (igraph.Graph): graph

        Returns:
            PyTree of parameters 
        """
        raise NotImplementedError
    

    def sample_obs(self, *, n_samples, g, theta, toporder=None):
        """Samples `n_samples` observations given `g` and `theta`
        
        Args:
            n_samples (int): number of iid samples
            g (igraph.Graph): graph
            theta (PyTree): parameters

        Returns:
            x: observations [N, d] 
        """

        raise NotImplementedError
        
        
    def log_prob_parameters(self, *, theta, g):
        """Computes p(theta | G)

        Args:
            g (igraph.Graph): graph
            theta (PyTree): parameters

        Returns:
            [1,]
        """

        raise NotImplementedError

    def log_likelihood(self, *, x, theta, g):
        """Computes p(x | theta, G)

        Args:
            g (igraph.Graph): graph
            theta (PyTree): parameters
            x: observations [N, d]
            
        Returns:
            [1,]
        """

        raise NotImplementedError
        

    def log_marginal_likelihood_given_g(self, *, g, x):
        """Computes log p(x | G) 

        Args:
            g (igraph.Graph): graph
            x: observations [N, d]
            
        Returns:
            [1,]
        """

        raise NotImplementedError
       

    def log_marginal_likelihood(self, *, x, all_g, z_g=None):
        """Computes the evidence log p(x) by exhaustive enumeration

        Args:
            x: observations [N, d]
            all_g: list of all possible DAGs with d nodes (igraph.Graph objects)
            
        Returns:
            [1,]
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
        Args:
            g (igraph.Graph): graph
            x : [N, d]
            log_marginal_likelihood: previously computed evidence 
                log p(x) from `log_marginal_likelihood`
            z_g: previously computed normalization constant of graph prior
                from GraphDistribution.log_normalization_constant

        Returns:
            [1,]
          
        """

        log_prob_g = self.g_dist.unnormalized_log_prob(g=g) - z_g
        log_marginal_likelihood_given_g = self.log_marginal_likelihood_given_g(
            g=g, x=x)
        return log_prob_g + log_marginal_likelihood_given_g - log_marginal_likelihood

   