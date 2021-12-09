import functools 

import numpy as onp

import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax
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

    Assumes the commonly-sued default diagonal parameter matrix T

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
        assert g.dtype == jnp.int32, f"g has dtype `{g.dtype}` but should have `int32`. Make sure you don't "\
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

