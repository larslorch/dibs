import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
import collections
import numpy as np

import jax.numpy as jnp
from jax import random

from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.utils.graph import *

from dibs.models.linearGaussianGaussianEquivalent import BGe
from dibs.models.dirichletCategorical import BDeu

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian


def test():

    '''This script tests that the closed-form computation of log p(D)
    by comparing with Monte Carlo integration 
    '''
    key = random.PRNGKey(0)

    jnp.set_printoptions(precision=6, suppress=True)
    print_graphs_per_score = True

    n_vars = 3

    g_dist = UniformDAGDistributionRejection(n_vars=n_vars)

    models = [
        # BDeu(
        #     g_dist=g_dist,
        #     n_categories=3 * jnp.ones(n_vars, dtype=jnp.int32),
        #     # higher alpha attributes higher confidence 
        #     # to the obtained samples compared to the empty graph
        #     alpha=100.0, 
        # ),
        LinearGaussianGaussian(
            g_dist=g_dist,
            obs_noise=1.0,
            mean_edge=2.0,
            sig_edge=2.0,
        ),
        
        # works for structure learning - but sampling and marginal likelihood in principle not consistent 
        # (MC validation not applicable)
        # BGe(
        #     g_dist=g_dist,
        #     mean_obs=10.0 * np.ones(n_vars),
        #     alpha_mu=1.0,
        #     alpha_lambd=n_vars + 2,
        # ),
    ]

    n_observations = 10
    # mc_validation_samples = 1e4
    mc_validation_samples = 0
    all_dags = make_all_dags(n_vars=n_vars)

    # validate ground truth model by computing exact density values using exhaustive search
    for model in models:

        print(type(model).__name__)

        # G ~ p(G)
        key, subk = random.split(key)
        g = g_dist.sample_G(key=subk)

        print(g)

        # theta ~ p(theta | G)
        key, subk = random.split(key)
        theta = model.sample_parameters(key=subk, g=g)

        # x_1, ..., x_n ~ p(x | theta, G) [n_samples, n_vars]
        key, subk = random.split(key)
        x = model.sample_obs(key=subk, n_samples=n_observations, g=g, theta=theta)

        print('logp(theta | G): ', model.log_prob_parameters(theta=theta, g=g))
        print('logp(X | G, theta): ', model.log_likelihood(x=x, theta=theta, g=g))

        # validate computation of log evidence
        log_marginal_likelihood_given_g = model.log_marginal_likelihood_given_g(x=x, g=g)
        print('logp(X|G): ', log_marginal_likelihood_given_g)

        if mc_validation_samples:
            key, subk = random.split(key)
            log_marginal_likelihood_given_g_mc = model.log_marginal_likelihood_given_g_mc(key=subk, x=x, g=g, n_samples=mc_validation_samples)
            print('logp(X|G) [MC]: ', log_marginal_likelihood_given_g_mc)

        #  log p(X) sanity check (compare closed-form with Monte Carlo integration)
        # z_g = log(sum_G p(G))
        z_g = g_dist.log_normalization_constant(all_g=all_dags)


        log_marginal_likelihood = model.log_marginal_likelihood(
            x=x, all_g=all_dags, z_g=z_g)
        print('logp(X): ', log_marginal_likelihood)

        if mc_validation_samples:
            key, subk = random.split(key)
            log_marginal_likelihood_mc = model.log_marginal_likelihood_mc(
                key=subk, x=x, n_samples=mc_validation_samples)
            print('logp(X) [MC]: ', log_marginal_likelihood_mc)

        # exhaustively print all DAGs with same score
        kk = 5
        if print_graphs_per_score:
            print(f'Top {kk} MECs:')
            mecs = collections.defaultdict(list)
            for j, g_ in enumerate(all_dags):
                log_posterior_G = model.log_posterior_graph_given_obs(
                    g=g_, x=x, log_marginal_likelihood=log_marginal_likelihood, z_g=z_g)
                mecs[round(log_posterior_G.item(), 6)].append(g_)

            for k in sorted(mecs.keys(), reverse=True)[:kk]:
                print(f'p(G | D) = {jnp.exp(k):6.4f}')
                for g_ in mecs[k]:
                    print(adjmat_to_str(graph_to_mat(g_))
                        + ('  ===> GROUND TRUTH' if np.all(graph_to_mat(g) == graph_to_mat(g_)) else ''))
                print()


if __name__ == '__main__':
    test()
