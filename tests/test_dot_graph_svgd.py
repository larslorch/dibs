import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging


import collections
import pprint
import tqdm
import pickle
import igraph as ig
import numpy as onp
from tabulate import tabulate
from varname import nameof
import time

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax.scipy.stats as jstats
import jax.scipy as jsp
import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
import jax.nn as nn
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_mul, index_update
import jax.lax as lax 


from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.utils.graph import *

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX

from dibs.svgd.dot_graph_svgd import DotProductGraphSVGD

from dibs.kernel.basic import (
    FrobeniusSquaredExponentialKernel, 
    AngularSquaredExponentialKernel, 
    StructuralHammingSquaredExponentialKernel,
)

from dibs.eval.mmd import MaximumMeanDiscrepancy
from dibs.utils.func import mask_topk, id2bit, bit2id, log_prob_ids, particle_empirical, kullback_leibler_dist

from dibs.eval.target import make_ground_truth_posterior

from config.svgd import marginal_config

import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype")


if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)

    key = random.PRNGKey(0)
    c = 0
    
    '''
    This script tests Graph SVGD idea for at most 5 nodes
    '''
    # target
    n_vars = 20
    n_dim = 10
    verbose = True
    metric_every = 50

    n_observations = 20
    n_ho_observations = 100
    n_posterior_g_samples = 100
    n_intervention_sets = 10
    perc_intervened = 0.1

    generative_model_str = 'lingauss'
    generative_model_kwargs = dict(
            mean_edge=0.0,
            sig_edge=1.0,
            obs_noise=0.1,
        )

    inference_model_str = 'bge'
    inference_model_kwargs = dict(
        mean_obs=jnp.zeros(n_vars),
        alpha_mu=1e-3,
        alpha_lambd=n_vars + 2.0,
    )

    graph_embedding_representation = True
    # graph_embedding_representation = False


    '''SVGD'''
    tuned = marginal_config['bge'][n_vars if n_vars in marginal_config['bge'].keys() else 20]
    
    n_steps = 2000
    n_particles = 10
    n_grad_mc_samples = 32
    n_acyclicity_mc_samples = 16
    optimizer = dict(name='rmsprop', stepsize=tuned.get("opt_stepsize", 0.005))
    # score_function_baseline = 0.001
    score_function_baseline = 0.00

    constraint_prior_graph_sampling = 'soft'

    fix_rotation = "not"

    kernel = FrobeniusSquaredExponentialKernel(
        h=tuned['h'],
        scale=tuned.get('kernel_scale', 1.0),
        graph_embedding_representation=graph_embedding_representation)

    # temperature hyperparameters
    def linear_alpha(t):
        return (tuned.get('alpha_slope', 0.5) * t) + tuned.get('alpha', 1.0)

    def linear_beta(t):
        return (tuned.get('beta_slope', 0.5) * t) + tuned.get('beta', 1.0)

    def linear_gamma(t):
        return (tuned.get('gamma_slope', 1e-4) * t) + tuned.get('gamma', 1.0)

    def linear_tau(t):
        return (tuned.get('tau_slope', 0.01) * t) + tuned.get('tau', 1.0)

    def const_alpha(t):
        return jnp.array([1.0])

    def const_beta(t):
        return jnp.array([1.0])

    def const_gamma(t):
        return jnp.array([1.0])

    def const_tau(t):
        return jnp.array([1.0])

    alpha_sched = linear_alpha
    beta_sched = linear_beta
    gamma_sched = linear_gamma
    tau_sched = const_tau

    # temperature schedule
    print_sched = 5
    ts = jnp.arange(0, int(n_steps*(print_sched + 1) / print_sched),
                    step=int(n_steps/print_sched))

    print(tabulate(
        [['t', *['{:10d}'.format(t) for t in ts]],
         ['alpha', *['{:10.4f}'.format(b) for b in alpha_sched(ts)]],
         ['beta', *['{:10.4f}'.format(b) for b in beta_sched(ts)]],
         ['gamma', *['{:10.4f}'.format(b) for b in gamma_sched(ts)]],
         ['tau', *['{:10.4f}'.format(b) for b in tau_sched(ts)]],
         ]))



    '''Target'''
    key, subk = random.split(key)
    target = make_ground_truth_posterior(
        key=subk, c=c, n_vars=n_vars,
        graph_prior_str='er',
        generative_model_str=generative_model_str,
        generative_model_kwargs=generative_model_kwargs,
        inference_model_str=inference_model_str,
        inference_model_kwargs=inference_model_kwargs,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        n_posterior_g_samples=n_posterior_g_samples,
        n_intervention_sets=n_intervention_sets,
        perc_intervened=perc_intervened,
        load=False, verbose=verbose)

    def log_prior(single_w_prob):
        ''' p(G) 
            returns shape [1, ]
            will later be vmapped

            single_w_prob : [n_vars, n_vars] in [0, 1]
                encoding probabilities of edges
        '''
        return target.g_dist.unnormalized_log_prob_soft(soft_g=single_w_prob)

    # inference model
    if inference_model_str == 'bge':
        model = BGeJAX(**inference_model_kwargs)

    elif inference_model_str == 'lingauss':
        model = LinearGaussianGaussianJAX(**inference_model_kwargs)
        
    else:
        raise NotImplementedError()


    x = jnp.array(target.x)
    x_ho = jnp.array(target.x_ho)

    def log_target(single_w):
        ''' p(D | G) 
            returns shape [1, ]
            will later be vmapped

            single_w : [n_vars, n_vars] in {0, 1}
        '''
        score = model.log_marginal_likelihood_given_g(w=single_w, data=x)
        return score

    # evaluates [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
    eltwise_log_marg_likelihood = jit(vmap(lambda w, x_: model.log_marginal_likelihood_given_g(w=w, data=x_), (0, None), 0))

    '''SVGD'''

    # assuming dot product representation of graph
    key, subk = random.split(key)
    if graph_embedding_representation:
        latent_prior_std = 1.0 / jnp.sqrt(n_dim)
        init_particles = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * latent_prior_std
    else:
        latent_prior_std = 1.0
        init_particles = random.normal(subk, shape=(n_particles, n_vars, n_vars)) * latent_prior_std

    # svgd
    svgd = DotProductGraphSVGD(
        n_vars=n_vars,
        n_dim=n_dim,
        optimizer=optimizer,
        kernel=kernel, 
        target_log_prior=log_prior,
        target_log_prob=log_target,
        alpha=alpha_sched,
        beta=beta_sched,
        gamma=gamma_sched,
        tau=tau_sched,
        n_grad_mc_samples=n_grad_mc_samples,
        n_acyclicity_mc_samples=n_acyclicity_mc_samples,
        clip=None,
        fix_rotation=fix_rotation,
        grad_estimator='score',
        score_function_baseline=score_function_baseline,
        repulsion_in_prob_space=False,
        latent_prior_std=latent_prior_std,
        constraint_prior_graph_sampling=constraint_prior_graph_sampling,
        graph_embedding_representation=graph_embedding_representation,
        verbose=True)

    # metrics
    mmd_kernel = StructuralHammingSquaredExponentialKernel(h=1.0)
    metrics = svgd.make_metrics(
        x_ho=x_ho, eltwise_log_marg_likelihood=eltwise_log_marg_likelihood, target=target,
        log_posterior=target.log_posterior, mmd_kernel=mmd_kernel, n_mmd_samples=100)

    # evaluate
    key, subk = random.split(key)
    particles = svgd.sample_particles(
        key=subk,
        n_steps=n_steps, 
        init_particles=init_particles.copy(), 
        metric_every=metric_every,
        eval_metrics=([
            metrics['kl_metric_hard'],
            metrics['kl_metric_hard_mixture'],
            metrics['mmd_metric_hard'],
        ] if n_vars <= 5 else []) + \
        [
            # metrics['log_marginal_likelihood_hard'],
            metrics['log_marginal_likelihood_mixture'],
            # metrics['edge_belief_hard'],
            metrics['edge_belief_mixture'],
            # metrics['particle_norm_print'],
            # metrics['phi_norm_print'],
        ])
    

    '''Analysis'''
    topk = 10
    svgd_g = svgd.particle_to_hard_g(particles)
    svgd_empirical = particle_empirical(svgd_g)
    svgd_ids, svgd_logprobs = svgd_empirical

    # number of cyclic graphs returned by SVGD
    svgd_unique_g = id2bit(svgd_ids, n_vars)
    dag_count = (eltwise_acyclic_constr(svgd_g, n_vars)[0] == 0).sum()
    unique_dag_count = (eltwise_acyclic_constr(svgd_unique_g, n_vars)[0] == 0).sum()

    print(
        '\nSVGD Acyclicity\n' + \
        'graphs : {}\t'.format(n_particles) +
        'unique graphs : {}\n'.format(svgd_ids.shape[0]) +
        'cyclic graphs :        {}\t'.format(n_particles - dag_count) +
        'percentage cyclic :        {:6.04f} %\n'.format(100 * (n_particles - dag_count) / n_particles) +
        'cyclic unique graphs : {}\t'.format(svgd_ids.shape[0] - unique_dag_count) + \
        'percentage unique cyclic : {:6.04f} %\n'.format(100 * (svgd_ids.shape[0] - unique_dag_count) / svgd_ids.shape[0])
    )

    if n_vars > 5:
        exit(0)

    # overlap in highest probability graphs
    gt_ids, gt_logprobs = target.log_posterior

    svgd_topk = mask_topk(svgd_logprobs, topk)
    gt_topk = mask_topk(gt_logprobs, topk)

    gt_of_gt_topk =     log_prob_ids(gt_ids[gt_topk, :],     target.log_posterior)
    svgd_of_svgd_topk = log_prob_ids(svgd_ids[svgd_topk, :], svgd_empirical)
    svgd_of_gt_topk =   log_prob_ids(gt_ids[gt_topk, :],     svgd_empirical)
    gt_of_svgd_topk =   log_prob_ids(svgd_ids[svgd_topk, :], target.log_posterior)

    id_intersect = [id for id in set([tuple(x) for x in gt_ids[gt_topk, :]]) 
                            & set([tuple(x) for x in svgd_ids[svgd_topk, :]])]

    print('Overlap in highest probability mass')
    print(tabulate(
        [['GT prob',   jnp.exp(gt_of_gt_topk),   jnp.exp(gt_of_svgd_topk)],
         ['SVGD prob', jnp.exp(svgd_of_gt_topk), jnp.exp(svgd_of_svgd_topk)]],
        headers=['Intersect: {}'.format(len(id_intersect)), 'Top K (by GT)', 'Top K (by SVGD)'],
        floatfmt="10.4f"))
    print()

    # visualize highest probability graphs
    topk_vis = n_particles # 5
    print('Highest probability SVGD graphs')
    plot_topk = mask_topk(svgd_logprobs, topk_vis)

    plot_ids = svgd_ids[plot_topk, :]
    plot_g = id2bit(plot_ids, n_vars)
    plot_logprob = log_prob_ids(plot_ids, svgd_empirical)
    plot_logprob_gt = log_prob_ids(plot_ids, target.log_posterior)

    for g, logp, logp_gt in zip(plot_g, plot_logprob, plot_logprob_gt):
        print('SVGD p(G | D) = {:5.04f}    GT p(G | D) = {:5.04f} '.format(jnp.exp(logp), jnp.exp(logp_gt)),
            end='\t [cyclic]\n' if not mat_is_dag(g) else '\n')
        print(adjmat_to_str(g), 
            end='\t===> GROUND TRUTH\n\n' if jnp.all(target.g == g) else '\n\n')

