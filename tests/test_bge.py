import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

from collections import defaultdict
import numpy as onp
import time
import tqdm
import pandas as pd

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_update

from cdt.metrics import get_CPDAG

from dibs.utils.func import id2bit, bit2id, mask_topk
from dibs.utils.graph import adjmat_to_str, make_all_dags, mat_to_graph, graph_to_mat

from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX

from dibs.models.pygobnilb import BGe as BGeTEST
from dibs.models.pygobnilb import ContinuousData


import numpy as np

if __name__ == '__main__':

    n_observations = 100
    verbose = True

    # gt = jnp.array([
    #     [0, 1, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0],
    # ])

    gt = jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])

    # gt = jnp.array([
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [0, 0, 0],
    # ])


    """
    BGe testing
    See for reference implmentation
    https://bitbucket.org/jamescussens/pygobnilp/src/master/pygobnilp/scoring.py
    """

    jnp.set_printoptions(precision=6, suppress=True)
    key = random.PRNGKey(0)
    
    # igraph.Graph
    print('\nGround truth DAG')
    print(gt)
    print(adjmat_to_str(gt))
    g_gt = mat_to_graph(gt)
    n_vars = gt.shape[-1]

    # # CPDAG
    # print('\nGround truth CPDAG')
    # gt_cpdag = jnp.array(get_CPDAG(onp.array(gt)), dtype=gt.dtype)
    # print(gt_cpdag)
    # print(adjmat_to_str(gt_cpdag))
    # print()

    """Generate standard observations"""
    generative_model = LinearGaussianGaussian(
        obs_noise=0.1,
        mean_edge=0.0,
        sig_edge=1.0,
        g_dist=None,
        verbose=verbose,
    )

    key, subk = random.split(key)
    theta = generative_model.sample_parameters(key=subk, g=g_gt)

    print('theta')
    print(len(theta))
    print(theta[0])
    print(theta[1])

    key, subk = random.split(key)
    x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta)

    print('x')
    print(x.shape)

    path = '/Users/Lars/Not iCloud/thesis/data/gaussian.dat'
    x = np.array(pd.read_csv(path, sep=' ').values)[:5]

    gt = jnp.array([
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ])
    g_gt = mat_to_graph(gt)
    n_vars = 7

    """BGe settings"""
    g_dist = UniformDAGDistributionRejection(n_vars=n_vars)
    model = BGe(
        g_dist=g_dist,
        mean_obs=0.0 * jnp.ones(n_vars),
        alpha_mu=1.0,
        alpha_lambd=n_vars + 2,
        verbose=verbose,
    )


    score_own = model.log_marginal_likelihood_given_g(g=g_gt, x=x)
    print('score own', score_own)


    # testing
    
    # data = ContinuousData(data=np.array(x), varnames=['x0', 'x1', 'x2', 'x3'])

    # model_test = BGeTEST(data=data, alpha_mu=1.0, nu=np.zeros(n_vars))

    # score_test = (
    #     model_test.bge_score(child='x0', parents=tuple([]))[0] + 
    #     model_test.bge_score(child='x1', parents=tuple(['x0']))[0] +
    #     model_test.bge_score(child='x2', parents=tuple([]))[0] +
    #     model_test.bge_score(child='x3', parents=tuple(['x1', 'x2']))[0]
    # )
    # print('score test', score_test)

       
    data = ContinuousData(data=np.array(x), varnames=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

    model_test = BGeTEST(data=data, alpha_mu=1.0, nu=np.zeros(n_vars))

    score_test = (
        model_test.bge_score(child='A', parents=tuple([]))[0] +
        model_test.bge_score(child='B', parents=tuple([]))[0] +
        model_test.bge_score(child='C', parents=tuple(['A', 'B']))[0] +
        model_test.bge_score(child='D', parents=tuple(['B']))[0] +
        model_test.bge_score(child='E', parents=tuple([]))[0] +
        model_test.bge_score(child='F', parents=tuple(['A', 'D', 'E', 'G']))[0] +
        model_test.bge_score(child='G', parents=tuple([]))[0]
    )
    print('score test', score_test)


