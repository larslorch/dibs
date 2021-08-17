import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
import collections
import numpy as np
import time
import tqdm

import jax.numpy as jnp
from jax import random
import jax.lax as lax

from dibs.graph.distributions import UniformDAGDistributionRejection, LowerTriangularDAGDistribution
from dibs.utils.graph import *

from dibs.models.dirichletCategorical import BDeu

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX

from dibs.models.linearGaussianGenerativeModel import GBNSampler 
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX

from dibs.utils.func import leftsel


def test():
    
    jnp.set_printoptions(precision=6, suppress=True)
    key = random.PRNGKey(1)

    '''This script tests that the closed-form computation of log p(D)
    by comparing with Monte Carlo integration 
    '''

    n_tests = 50

    n_vars = 4
    n_observations = 10

    # models
    compute_posterior = True
    g_dist = LowerTriangularDAGDistribution(n_vars=n_vars, sparsity_factor=1.0)
    models = [
        [
            # generative model
            GBNSampler(
                edge_interval=[(-3.0, -1.0), (1.0, 3.0)],
            ),
            # scoring models
            BGe(
                g_dist=g_dist,
                mean_obs=jnp.zeros(n_vars),
                # if 0 < alpha_mu << 1 posterior converges to correct MEC
                # a lot quicker than for alpha_mu >> 1;
                # 0 < alpha_mu << 1 makes posterior more peaky
                alpha_mu=1, #1e-3, 
                alpha_lambd=n_vars + 2, 
            ),
            BGeJAX(
                mean_obs=jnp.zeros(n_vars),
                # if 0 < alpha_mu << 1 posterior converges to correct MEC
                # a lot quicker than for alpha_mu >> 1;
                # 0 < alpha_mu << 1 makes posterior more peaky
                alpha_mu=1e-3,
                alpha_lambd=n_vars + 2,
            ),
        ],
        [
            # generative model
            LinearGaussianGaussian(
                g_dist=g_dist,
                obs_noise=0.1,
                mean_edge=0.0,
                sig_edge=3.0,
            ),
            # scoring models
            BGe(
                g_dist=g_dist,
                mean_obs=jnp.zeros(n_vars),
                # if 0 < alpha_mu << 1 posterior converges to correct MEC
                # a lot quicker than for alpha_mu >> 1;
                # 0 < alpha_mu << 1 makes posterior more peaky
                alpha_mu=1, #1e-3,
                alpha_lambd=n_vars + 2,
            ),
            BGeJAX(
                mean_obs=jnp.zeros(n_vars),
                # if 0 < alpha_mu << 1 posterior converges to correct MEC
                # a lot quicker than for alpha_mu >> 1;
                # 0 < alpha_mu << 1 makes posterior more peaky
                alpha_mu=1e-3,
                alpha_lambd=n_vars + 2,
            ),
        ],
        # [   
        #     # generative model
        #     LinearGaussianGaussian(
        #         g_dist=g_dist,
        #         obs_noise=0.1,
        #         mean_edge=0.0,
        #         sig_edge=3.0,
        #     ),
        #     # scoring models
        #     LinearGaussianGaussian(
        #         g_dist=g_dist,
        #         obs_noise=0.1,
        #         mean_edge=0.0,
        #         sig_edge=3.0,
        #     ),
        #     LinearGaussianGaussianJAX(
        #         obs_noise=0.1,
        #         mean_edge=0.0,
        #         sig_edge=3.0,
        #     ),
        # ],        
    ]


    if n_vars <= 4:
        all_dags = make_all_dags(n_vars=n_vars)

    for _ in range(5):
        print('----------------------------------------------')
        
        # validate ground truth model by computing exact density values using exhaustive search
        for generative_model, ig_model, jax_model in models:

            print('Generative: ', type(generative_model).__name__)
            print('Score: ', type(ig_model).__name__, type(jax_model).__name__)

            n_passed, n_failed = 0, 0
            ig_time, jax_time = 0.0, 0.0

            # jit jax version
            jit_jax_log_marginal_likelihood_given_g = jit(
                lambda w_, x_: jax_model.log_marginal_likelihood_given_g(w=w_, data=x_)
            )

            for t in range(n_tests):

                # G ~ p(G)
                key, subk = random.split(key)
                g = g_dist.sample_G(key=subk)
                g_mat = graph_to_mat(g)

                # theta ~ p(theta | G)
                key, subk = random.split(key)
                theta = generative_model.sample_parameters(key=subk, g=g)

                # x_1, ..., x_n ~ p(x | theta, G) [n_samples, n_vars]
                key, subk = random.split(key)
                x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g, theta=theta)

                # validate computation of log marginal likelihood
                t0 = time.time()
                ig_log_marginal_likelihood_given_g = ig_model.log_marginal_likelihood_given_g(x=x, g=g)
                t1 = time.time()
                jax_log_marginal_likelihood_given_g = jit_jax_log_marginal_likelihood_given_g(g_mat, x)
                t2 = time.time()

                ig_time += t1 - t0
                jax_time += t2 - t1

                same = jnp.allclose(ig_log_marginal_likelihood_given_g,
                                    jax_log_marginal_likelihood_given_g, atol=1e-3)
                if same:
                    n_passed += 1
                else:
                    n_failed += 1

            print(f'passed: {n_passed}\t failed: {n_failed}')
            if n_failed:
                print('Double check that IG/JAX arguments match.')
                exit(1)
            print(f'ig time: {ig_time}')
            print(f'jax time: {jax_time}')

            if n_vars <= 4 and compute_posterior:
                
                print('\n' + adjmat_to_str(g_mat) + '\n')
                print('theta:', theta, '\n')
                
                # posterior
                z_g = g_dist.log_normalization_constant(all_g=all_dags)
                log_marginal_likelihood = ig_model.log_marginal_likelihood(
                    x=x, all_g=all_dags, z_g=z_g)

                kk = 5
                print(f'Top {kk} MECs:')
                mecs = collections.defaultdict(list)
                for j, g_ in enumerate(all_dags):
                    log_posterior_G = ig_model.log_posterior_graph_given_obs(
                        g=g_, x=x, log_marginal_likelihood=log_marginal_likelihood, z_g=z_g)
                    mecs[round(log_posterior_G.item(), 6)].append(g_)

                for k in sorted(mecs.keys(), reverse=True)[:kk]:
                    p_each = jnp.exp(k)
                    print(f'p(G | D) = {p_each:6.4f}  [Total: {len(mecs[k]) * p_each:6.4f}]')
                    for g_ in mecs[k]:
                        print(adjmat_to_str(graph_to_mat(g_))
                            + ('  ===> GROUND TRUTH' if jnp.all(graph_to_mat(g) == graph_to_mat(g_)) else ''))
                    print()
        


if __name__ == '__main__':
    test()
