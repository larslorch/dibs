import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
import collections
import numpy as np
import time
import tqdm

import jax.numpy as jnp
from jax import random
import jax.lax as lax
from jax import grad, jit, vmap, vjp, jvp
from jax.ops import index, index_add, index_update

from dibs.graph.distributions import UniformDAGDistributionRejection, LowerTriangularDAGDistribution
from dibs.utils.graph import *


from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX


if __name__ == '__main__':

    jnp.set_printoptions(precision=6, suppress=True)
    key = random.PRNGKey(0)

    '''This script tests that the closed-form computation of log p(D)
    by comparing with Monte Carlo integration 
    '''
    n_tests = 10

    n_vars = 5
    # n_vars = 30
    n_observations = 50

    obs_noise = 0.1
    mean_edge = 0.0
    sig_edge = 3.0

    g_dist = LowerTriangularDAGDistribution(n_vars=n_vars, sparsity_factor=1.0)

    # generative model
    generative_model = LinearGaussianGaussian(
        g_dist=g_dist,
        obs_noise=obs_noise,
        mean_edge=mean_edge,
        sig_edge=sig_edge,
    )
    
    # inference models
    hard_model = LinearGaussianGaussian(
        g_dist=g_dist,
        obs_noise=obs_noise,
        mean_edge=mean_edge,
        sig_edge=sig_edge,
    )

    def hard_target(x, theta, g):
        return hard_model.log_prob_parameters(theta=theta, g=g) \
             + hard_model.log_likelihood(x=x, theta=theta, g=g)


    soft_model = LinearGaussianGaussianJAX(
        obs_noise=obs_noise,
        mean_edge=mean_edge,
        sig_edge=sig_edge,
    )

    # [N, d], [d, d], [d, d] -> [1,]
    def soft_target(x, theta, g):
        return soft_model.log_prob_parameters(theta=theta, w=g) \
            + soft_model.log_likelihood(data=x, theta=theta, w=g)

    # [N, d], [d, d], [d, d] -> [d, d]
    grad_soft_target_theta = jit(grad(soft_target, 1))

    # [N, d], [d, d], [d, d] -> [d, d]
    grad_soft_target_g = jit(grad(soft_target, 2))


    def theta_to_mat(theta, g):
        '''Converts edge thetas into mat form'''

        theta_mat = jnp.zeros((n_vars, n_vars))
        for j in range(n_vars):
            parent_edges = g.incident(j, mode='in')
            parents = jnp.array(list(g.es[e].source for e in parent_edges))
            
            if parent_edges:
                theta_mat = index_update(theta_mat, index[parents, j], theta[jnp.array(parent_edges)])

        return theta_mat

    # validate that soft and hard models compute the same probabilities when fed with hard graphs
    for t in range(n_tests):

        # G ~ p(G)
        key, subk = random.split(key)
        g = g_dist.sample_G(key=subk)
        g_mat = graph_to_mat(g).astype(jnp.float64)

        # theta ~ p(theta | G)
        key, subk = random.split(key)
        theta = generative_model.sample_parameters(key=subk, g=g)
        theta_mat = theta_to_mat(theta, g)

        # x_1, ..., x_n ~ p(x | theta, G) [n_samples, n_vars]
        key, subk = random.split(key)
        x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g, theta=theta)

    
        # validate computation of log marginal likelihood
        hard = hard_target(x, theta, g)
        soft = soft_target(x, theta_mat, g_mat)

        # print(hard)
        # print(soft)

        same = jnp.allclose(soft, hard, atol=1e-3)
        print(same)
        assert(same)

    print('passed.\n')

    print(g_mat)

    # g_mat = g_mat + 0.01
    # theta_mat = theta_mat + 0.01

    print('Example gradient:')    
    print('d/dtheta  p(theta, D | G)')
    dtheta = grad_soft_target_theta(x, theta_mat, g_mat)
    print(dtheta)
    print(dtheta.shape)

    print('d/dG  p(theta, D | G)')
    dg = grad_soft_target_g(x, theta_mat, g_mat)
    print(dg)
    print(dg.shape)



    # 

    #  import numpy as np

    # d = 10
    # for d in [5, 10, 25, 50]:
    #     print(f'==== nodes: {d * 2}')
    #     for c in [0.5, 0.2, 0.1, 0.05]:
    #         print(c)

    #         for gamma in [1, 2, 3, 5, 10, 20, 40]:
    #             h = (gamma * np.random.uniform(0, c, d) *
    #                         np.random.uniform(1.0 - c, 1.0, d)).prod()
    #             print(gamma, h)

    #         print()
    #     print()


    # exit()

