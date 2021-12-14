import numpy as onp

import jax.numpy as jnp
from jax import random, vmap, lax
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal
from jax.scipy.special import gammaln

from dibs.utils.func import leftsel


class BGe:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise
    Uses Normal-Wishart conjugate parameter prior to allow for closed-form marginal likelihood p(D | G)
    and thus inference of p(G | D)

    Refer to:
        Geiger et al (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981
        Kuipers et al (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013

    Assumes the commonly-used default diagonal parameter matrix T

    Inspiration for implementation was drawn from
    https://bitbucket.org/jamescussens/pygobnilp/src/master/pygobnilp/scoring.py

    Our implementation uses properties of the determinant to make the computation of the marginal likelihood
    @jit-compilable

    """

    def __init__(self, *,
                 graph_dist,
                 mean_obs,
                 alpha_mu,
                 alpha_lambd,
                 ):
        super(BGe, self).__init__()

        self.graph_dist = graph_dist
        self.n_vars = graph_dist.n_vars
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_lambd = alpha_lambd

        assert self.alpha_lambd > self.n_vars + 1
        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)

    def get_theta_shape(self, *, n_vars):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    """
    The following functions need to be functionally pure and @jit-able
    """

    def slogdet_jax(self, m, parents, n_parents):
        """
        Log determinant of a submatrix
        jax.jit-compatible

        Done by masking everything but the submatrix and
        adding a diagonal of ones everywhere else for the
        valid determinant

        Args:
            m: [d, d] matrix
            parents: [d, ] boolean indicator of parents
            n_parents: number of parents total

        Returns:
            natural log of determinant of `m`
        """

        n_vars = parents.shape[0]
        submat = leftsel(m, parents, maskval=onp.nan)
        submat = leftsel(submat.T, parents, maskval=onp.nan).T
        submat = jnp.where(jnp.isnan(submat), jnp.eye(n_vars), submat)
        return jnp.linalg.slogdet(submat)[1]

    def log_marginal_likelihood_single(self, j, n_parents, R, g, x, log_gamma_terms):
        """
        Computes node specific term of BGe metric
        jax.jit-compatible

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node j
            R: [d, d] internal matrix for BGe score
            g: [d, d] adjacency matrix
            x: [N, d] observations
            log_gamma_terms: internal values for BGe score [d, ]

        Returns:
            BGe score for node j
        """

        N, d = x.shape

        isj = jnp.arange(d) == j
        parents = g[:, j] == 1
        parents_and_j = parents | isj

        # if `JAX_DEBUG_NANS` flag raises NaN error here:  ignore
        # happens due to lax.cond evaluating the second clause when n_parents == 0
        log_term_r = lax.cond(
            n_parents == 0,
            # leaf node case
            lambda _: (
                # log det(R)^(...)
                    - 0.5 * (N + self.alpha_lambd - d + 1) * jnp.log(jnp.abs(R[j, j]))
            ),
            # child case
            lambda _: (
                # log det(R_II)^(..) / det(R_JJ)^(..)
                    0.5 * (N + self.alpha_lambd - d + n_parents) *
                    self.slogdet_jax(R, parents, n_parents)
                    - 0.5 * (N + self.alpha_lambd - d + n_parents + 1) *
                    self.slogdet_jax(R, parents_and_j, n_parents + 1)
            ),
            operand=None,
        )

        return log_gamma_terms[n_parents] + log_term_r

    def eltwise_log_marginal_likelihood_single(self, *args):
        """
        Same inputs as `log_marginal_likelihood_single`,
        but batched over `j` and `n_parents` dimensions
        """
        return vmap(self.log_marginal_likelihood_single, (0, 0, None, None, None, None), 0)(*args)

    def log_marginal_likelihood(self, *, g, x, interv_targets=None):
        """Computes BGe marginal likelihood  log p(x | G) in closed form
        jax.jit-compatible

        Args:
            g: adjacency matrix [d, d]
            x: observations [N, d]
            interv_targets: boolean mask of shape [d,] of whether or not a node was intervened on
                    intervened nodes are ignored in likelihood computation

        Returns:
            [1, ] BGe Score
        """
        assert g.dtype == jnp.int32, f"g has dtype `{g.dtype}` but should have `int32`. Make sure you don't " \
                                     "use the Gumbel-softmax but the score function gradient estimator with BGe "
        N, d = x.shape

        # intervention
        if interv_targets is None:
            interv_targets = jnp.zeros(d).astype(bool)

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / (self.alpha_mu + 1)
        T = small_t * jnp.eye(d)

        x_bar = x.mean(axis=0, keepdims=True)
        x_center = x - x_bar
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
        # their supplementary contains the correct term
        R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
            ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        # store log gamma terms for all possible values of l
        all_l = jnp.arange(d)
        log_gamma_terms = (
                0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
                + gammaln(0.5 * (N + self.alpha_lambd - d + all_l + 1))
                - gammaln(0.5 * (self.alpha_lambd - d + all_l + 1))
                - 0.5 * N * jnp.log(jnp.pi)
                # log det(T_JJ)^(..) / det(T_II)^(..) for default T
                + 0.5 * (self.alpha_lambd - d + 2 * all_l + 1) *
                jnp.log(small_t)
        )

        # compute number of parents for each node
        n_parents_all = g.sum(axis=0)

        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                interv_targets,
                0.0,
                self.eltwise_log_marginal_likelihood_single(
                    jnp.arange(d), n_parents_all, R, g, x, log_gamma_terms)
            )
        )

    """
    Distributions used by DiBS for inference:  prior and marginal likelihood 
    """

    def log_graph_prior(self, g_prob):
        """ log p(G)

        Arguments:
            g_prob: [n_vars, n_vars] of edge probabilities in G

        Returns:
            [1, ]
        """
        return self.graph_dist.unnormalized_log_prob_soft(soft_g=g_prob)

    def observational_log_marginal_prob(self, g, _, x, rng):
        """ log p(D | G)

        To unify the function signatures for the marginal and joint inference classes `MarginalDiBS` and `JointDiBS`,
        this marginal likelihood is defined with dummy `theta` inputs, i.e., like a joint likelihood

        Arguments:
           g: [n_vars, n_vars] graph adjacency matrix (entries must be in {0,1} and of type `jnp.int32`)
           _:
           x: [n_observations, n_vars] observational data
           rng

        Returns:
           [1, ]
        """
        return self.log_marginal_likelihood(g=g, x=x, interv_targets=self.no_interv_targets)


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

