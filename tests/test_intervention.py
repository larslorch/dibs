import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
import collections
import numpy as np

import jax.numpy as jnp
from jax import random

from dibs.graph.distributions import LowerTriangularDAGDistribution
from dibs.utils.graph import *

from dibs.models.linearGaussianGenerativeModel import GBNSampler 
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX


def test():

    '''This script tests interventions 
    '''
    key = random.PRNGKey(0)

    jnp.set_printoptions(precision=6, suppress=True)
    print_graphs_per_score = True

    n_vars = 5
    n_observations = 12

    g_dist = LowerTriangularDAGDistribution(n_vars=n_vars, sparsity_factor=1.0)

    # generative model
    # gen_model = GBNSampler(
    #     edge_interval=[(-3.0, -1.0), (1.0, 3.0)],
    # )
    gen_model = LinearGaussianGaussian(
        g_dist=g_dist,
        obs_noise=0.1,
        mean_edge=0.0,
        sig_edge=3.0,
    )

    # scoring models
    ig_model = BGe(
        g_dist=g_dist,
        mean_obs=jnp.zeros(n_vars),
        alpha_mu=1,
        alpha_lambd=n_vars + 2,
    )
    jax_model = BGeJAX(
        mean_obs=jnp.zeros(n_vars),
        alpha_mu=1,
        alpha_lambd=n_vars + 2,
    )
    jit_jax_log_marginal_likelihood_given_g = jit(
        lambda w_, x_, interv_targets_: jax_model.log_marginal_likelihood_given_g(
            w=w_, data=x_, interv_targets=interv_targets_)
    )   

    # random testing
    for _ in range(20):

        # G ~ p(G)
        key, subk = random.split(key)
        g = g_dist.sample_G(key=subk)
        g_mat = graph_to_mat(g)

        # theta ~ p(theta | G)
        key, subk = random.split(key)
        theta = gen_model.sample_parameters(key=subk, g=g)

        # random interventions
        # interv = {node index : clamp value)
        key, subk = random.split(key)
        n_interv = random.randint(subk, minval=0, maxval=n_vars + 1, shape=(1, ))[0]

        key, subk = random.split(key)
        interv_targets = random.choice(subk, n_vars, shape=(n_interv,), replace=False)

        key, subk = random.split(key)
        interv_vals = random.choice(subk, jnp.array([-1.0, 0.0, 1.0]), shape=(n_interv,), replace=True)

        interv = {k:v for k,v in zip(interv_targets, interv_vals)}
        # interv = {}

        # x_1, ..., x_n ~ p(x | theta, G, interv) [n_samples, n_vars]
        key, subk = random.split(key)
        x = gen_model.sample_obs(key=subk, n_samples=n_observations, g=g, theta=theta, interv=interv)

        # compute interventional likelihood (assuming hard intervention)
        ig_log_marginal_interv_likelihood_given_g = ig_model.log_marginal_likelihood_given_g(
            x=x, g=g, interv=interv)

        # jit jax version
        interv_targets = jnp.isin(jnp.arange(n_vars), jnp.array(list(interv.keys())))
        jax_log_marginal_interv_likelihood_given_g = jit_jax_log_marginal_likelihood_given_g(
            g_mat, x, interv_targets)

        same = jnp.allclose(ig_log_marginal_interv_likelihood_given_g,
                            jax_log_marginal_interv_likelihood_given_g, atol=1e-3)
        
        # print(same, ig_log_marginal_interv_likelihood_given_g, jax_log_marginal_interv_likelihood_given_g)
        assert(same)
    
    print('passed')




if __name__ == '__main__':
    test()
