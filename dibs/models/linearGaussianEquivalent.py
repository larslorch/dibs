import functools 

import numpy as np
import scipy

from .basic import BasicModel

import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax
from jax.scipy.special import gammaln

from dibs.utils.func import leftsel

class BGe(BasicModel):
    """
    Linear Gaussian-Gaussian model (continuous)

    Each variable distributed as Gaussian with mean being the linear combination of its parents 
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).

    The parameter prior over (mu, lambda) of the joint Gaussian distribution (mean `mu`, precision `lambda`) over x is Gaussian-Wishart, 
    as introduced in 
        Geiger et al (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981

    Computation is based on
        Kuipers et al (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013 

    Note: 
        - (mu, Sigma) of joint is not factorizable into independent theta, but there exists a one-to-one correspondence.
        - lambda = Sigma^{-1}
        - assumes default diagonal parametric matrix T

    Some inspiration was drawn from: https://bitbucket.org/jamescussens/pygobnilp/src/master/pygobnilp/scoring.py 

    """

    def __init__(self, *,
            g_dist,
            mean_obs,
            alpha_mu,
            alpha_lambd,
            verbose=False
            ):
        super(BGe, self).__init__(g_dist=g_dist, verbose=verbose)

        self.n_vars = g_dist.n_vars
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_lambd = alpha_lambd

        assert(self.alpha_lambd > self.n_vars + 1)

    def log_marginal_likelihood_given_g_single(self, g, x, j, R=None):
        """Computes log p(x | G) in closed form using properties of Gaussian-Wishart
        
        Args:
            g (igraph.Graph): graph
            x: observations [N, d]
            j (int): node index for node score

        Returns:
            [1, ]        
        """

        N, d = x.shape
        assert(d == self.n_vars)

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / (self.alpha_mu + 1)

        if R is None:
            T = small_t * np.eye(d)
            x_bar = x.mean(axis=0, keepdims=True)
            x_center = x - x_bar
            s_N = x_center.T @ x_center  # [d, d]

            # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
            # the supplementary contains the correct term
            R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        parent_edges = g.incident(j, mode='in')
        parents = list(g.es[e].source for e in parent_edges)
        l = len(parents)

        # compute gamma term + log ratio of det(T) (since we use default prior matrix T)
        log_gamma_terms_const = 0.5 * (np.log(self.alpha_mu) - np.log(N + self.alpha_mu))
        log_gamma_term = (
            log_gamma_terms_const
            + scipy.special.loggamma(0.5 * (N + self.alpha_lambd - d + l + 1))
            - scipy.special.loggamma(0.5 * (self.alpha_lambd - d + l + 1))
            - 0.5 * N * np.log(np.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_lambd - d + 2 * l + 1) * \
            np.log(small_t)
        )

        # leaf node case
        if l == 0:
            # log det(R)^(...)
            R_II = R[j, j]
            log_term_r = - 0.5 * \
                (N + self.alpha_lambd - d + 1) * np.log(np.abs(R_II))
        else:
            # log det(R_II)^(..) / det(R_JJ)^(..)
            log_term_r = (
                0.5 * (N + self.alpha_lambd - d + l) *
                np.linalg.slogdet(R[np.ix_(parents, parents)])[1]
                - 0.5 * (N + self.alpha_lambd - d + l + 1) *
                np.linalg.slogdet(
                    R[np.ix_([j] + parents, [j] + parents)])[1]
            )

        return log_gamma_term + log_term_r


    def log_marginal_likelihood_given_g(self, g, x, interv={}):
        """Computes log p(x | G) in closed form using properties of Gaussian-Wishart

        Args:
            g (igraph.Graph): graph
            x: observations [N, d]
            interv (dict): {intervened node : clamp value}

        Returns:
            [1, ]        
        """

        N, d = x.shape
        assert(d == self.n_vars)

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / (self.alpha_mu + 1)
        T = small_t * np.eye(d)

        x_bar = x.mean(axis=0, keepdims=True)
        x_center = x - x_bar
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
        # the supplementary contains the correct term
        R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
            ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        # node-wise score
        logp = 0.0
        for j in range(self.n_vars):

            # intervention: skip if intervened
            if j in interv.keys():
                continue

            logp += self.log_marginal_likelihood_given_g_single(g=g, x=x, j=j, R=R)

        return logp


class BGeJAX:
    """
    JAX implementation of BGe that allows for @jax.jit
    """

    def __init__(self, *,
            mean_obs,
            alpha_mu,
            alpha_lambd,
            verbose=False
            ):
        super(BGeJAX, self).__init__()

        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_lambd = alpha_lambd


    def slogdet_jax(self, m, parents, n_parents):
        """
        jax.jit-compatible log determinant of a submatrix

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
        submat = leftsel(m, parents, maskval=np.nan)
        submat = leftsel(submat.T, parents, maskval=np.nan).T
        submat = jnp.where(jnp.isnan(submat), jnp.eye(n_vars), submat)
        return jnp.linalg.slogdet(submat)[1]


    def log_marginal_likelihood_given_g_single(self, j, n_parents, R, w, data, log_gamma_terms):
        """
        Computes node specific term of BGe metric
        jit-compatible

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node j
            R: internal matrix for BGe score [d, d]
            w: adjacency matrix [d, d] 
            data: observations [N, d] 
            log_gamma_terms: internal values for BGe score [d, ]

        Returns:
            BGe score for node j
        """

        N, d = data.shape

        isj = jnp.arange(d) == j
        parents = w[:, j] == 1
        parents_and_j = parents | isj

        # if JAX_DEBUG_NANS raises NaN error here,
        # ignore (happens due to lax.cond evaluating the second clause when n_parents == 0)
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
        

    @functools.partial(jit, static_argnums=(0, ))
    def eltwise_log_marginal_likelihood_given_g_single(self, *args):
        """
        Same inputs as `log_marginal_likelihood_given_g_single`,
        but batched over `j` and `n_parents` dimensions
        """
        return vmap(self.log_marginal_likelihood_given_g_single, (0, 0, None, None, None, None), 0)(*args)


    def log_marginal_likelihood_given_g(self, *, w, data, interv_targets=None):
        """Computes BGe marignal likelihood  log p(x | G) in closed form 

        Args:	
            data: observations [N, d]	
            w: adjacency matrix [d, d]	
            interv_targets: boolean mask of shape [d,] of whether or not a node was intervened on
                    intervened nodes are ignored in likelihood computation

        Returns:
            [1, ] BGe Score
        """
        
        N, d = data.shape        

        # intervention
        if interv_targets is None:
            interv_targets = jnp.zeros(d).astype(bool)

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / \
            (self.alpha_mu + 1)
        T = small_t * jnp.eye(d)

        x_bar = data.mean(axis=0, keepdims=True)
        x_center = data - x_bar
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
        # the supplementary contains the correct term
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
            + 0.5 * (self.alpha_lambd - d + 2 * all_l + 1) * \
            jnp.log(small_t)
        )

        # compute number of parents for each node
        n_parents_all = w.sum(axis=0)

        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                interv_targets,
                0.0,
                self.eltwise_log_marginal_likelihood_given_g_single(jnp.arange(
                    d), n_parents_all, R, w, data, log_gamma_terms)
            )
        )

        # prev code without interventions
        # return jnp.sum(self.log_marginal_likelihood_given_g_j(jnp.arange(d), n_parents_all, R, w, data, log_gamma_terms))


            
            
