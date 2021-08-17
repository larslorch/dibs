import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import collections
import pprint
import igraph as ig
import tqdm
import scipy

import numpy as np
import jax.numpy as jnp
from jax import random

from dibs.models.dirichletCategorical import BDeu
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian

from dibs.graph.distributions import UniformDAGDistributionRejection, LowerTriangularDAGDistribution
from dibs.utils.graph import *
from dibs.utils.func import id2bit, particle_empirical_mixture

from dibs.bootstrap.bootstrap import NonparametricDAGBootstrap
from dibs.bootstrap.learners import GES, PC

from cdt.metrics import get_CPDAG

if __name__ == '__main__':

    key = random.PRNGKey(0)
    jnp.set_printoptions(precision=6, suppress=True)

    '''
    This script tests DAG Bootstrap

    n_boot_samples = 100, n_observations = 1000, n_vars = 4
    GES -- LinearGaussianGaussian   =   works (80/100 correct CPDAG)
    PC  -- LinearGaussianGaussian   =   works (100/100 correct CPDAG)

    For BDeu, since GES has BIC as score, might be misspecified
    GES -- BDeu                     =   doesn't work (as expected)
    PC  -- BDeu                     =   works somewhat (need large-ish n_boot_samples)

    '''
    
    toy_graph = True
    n_boot_samples = 30
    n_observations = 2000
    verbose = True
    
    recover_cpdags = True

    # generate ground truth observations
    if toy_graph:
        n_vars = 4
        gt_g_prior = UniformDAGDistributionRejection(n_vars=n_vars)
        g_gt = ig.Graph(directed=True)
        g_gt.add_vertices(4)
        g_gt.add_edges([(1, 0), (1, 2), (0, 2), (3, 2)])

    else:
        n_vars = 10
        gt_g_prior = LowerTriangularDAGDistribution(n_vars=n_vars, sparsity_factor=1.0)
        g_gt = gt_g_prior.sample_G()
    

    # This closely corresponds to the BIC score
    # (same Gaussain linear SEM assumption)
    # gt_model = LinearGaussianGaussian(
    #     g_dist=gt_g_prior,
    #     obs_noise=1.0,
    #     mean_edge=0.0,
    #     sig_edge=2.0,
    # )

    gt_model = BDeu(
        g_dist=gt_g_prior,
        n_categories=3 * jnp.ones(n_vars, dtype=jnp.int32),
        alpha=1.0,
    )

    # sample data    
    key, subk = random.split(key)
    theta_gt = gt_model.sample_parameters(key=subk, g=g_gt)

    key, subk = random.split(key)
    x = gt_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta_gt)

    print('Ground truth DAG:     ', adjmat_to_str(graph_to_mat(g_gt)))
    print('Ground truth CPDAG:   ', adjmat_to_str(get_CPDAG(np.array(graph_to_mat(g_gt)))))

    # init DAG bootstrap
    # learner = GES()
    learner = PC(ci_test='discrete', ci_alpha=1e-2)
    dag_bootstrap = NonparametricDAGBootstrap(learner=learner, verbose=verbose)
    print(type(learner).__name__)

    # sample
    key, subk = random.split(key)
    boot_samples = dag_bootstrap.sample_particles(key=subk, n_samples=n_boot_samples, x=x)

    # CPDAG results
    if recover_cpdags:
        dag_counter = collections.Counter()
        cpdag_counter = collections.Counter()
        for l in tqdm.tqdm(range(n_boot_samples), desc='Recovering CPDAGs'):

            # check whether is DAG
            assert(mat_is_dag(boot_samples[l]))
            dag_counter[tuple(boot_samples[l].reshape(n_vars * n_vars).tolist())] += 1
            cpdag = get_CPDAG(np.array(boot_samples[l]))
            cpdag_counter[tuple(cpdag.reshape(n_vars * n_vars).tolist())] += 1

        # print
        print('\nFound {} CPDAGs with frequencies: '.format(len(cpdag_counter)))
        for dag, freq in cpdag_counter.most_common():
            print(freq, adjmat_to_str(jnp.array(dag).reshape(n_vars, n_vars)))

        
        print('\nFound {} DAGs with frequencies: '.format(len(dag_counter)))
        for dag, freq in dag_counter.most_common():
            print(freq, adjmat_to_str(jnp.array(dag).reshape(n_vars, n_vars)))

    # print mixture distribution components 
    def unnormalized_log_target(g):
        return gt_model.g_dist.unnormalized_log_prob(g=g) + \
            gt_model.log_marginal_likelihood_given_g(g=g, x=x)

    def eltwise_log_target(g_array):
        # [N, d, d] -> [N, ]
        return jnp.array([unnormalized_log_target(mat_to_graph(g)) for g in g_array])

    unique_dag_ids, log_probs = particle_empirical_mixture(boot_samples, eltwise_log_target)
    unique_dags = id2bit(unique_dag_ids, n_vars)

    print('\nComponent of mixture distribution implied by DAG bootstrap')
    print_first = 20
    sort_idx = log_probs.argsort()[::-1][:print_first]
    for idx in sort_idx:
        print('p = {:8.4f}   |  log p = {:14.4f}   |  {}'.format(
            jnp.exp(log_probs[idx]), log_probs[idx], adjmat_to_str(unique_dags[idx]))
            + ('   ===> GROUND TRUTH' if jnp.all(unique_dags[idx] == graph_to_mat(g_gt)) else ''))
