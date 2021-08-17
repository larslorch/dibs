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
import jax

from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.utils.graph import *

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX

from dibs.svgd.batch_dot_graph_svgd import BatchedDotProductGraphSVGD

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
jnp.set_printoptions(precision=4, suppress=True)


if __name__ == '__main__':

    '''
    This script tests Graph SVGD idea for at most 5 nodes
    '''
    '''SVGD'''
    batch_size = 3
    n_variants = 3

    # target
    n_vars = 20
    n_dim = 10
    verbose = True
    metric_every = 50
    sparsity_factor = 1.0

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

    '''SVGD'''
    tuned = marginal_config['bge'][n_vars if n_vars in marginal_config['bge'].keys() else 20]

    n_steps = 2000
    n_particles = 10
    n_grad_mc_samples = 32
    n_acyclicity_mc_samples = 16
    optimizer = dict(name='rmsprop', stepsize=tuned.get("opt_stepsize", 0.005))
    # score_function_baseline = 0.001
    score_function_baseline = 0.00

    # fix_rotation = 'parallel'
    # fix_rotation = 'orthogonal'
    fix_rotation = "not"

    constraint_prior_graph_sampling = 'soft'

    graph_embedding_representation = True
    # graph_embedding_representation = False

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

    '''Multiple targets approximated in parallel in batch'''
    random_seed = 0
    key = random.PRNGKey(random_seed)

    variants_x = []
    variants_log_posterior = []
    variants_target = []
    key, *subk = random.split(key, n_variants + 1)
    for c in range(n_variants):
        target = make_ground_truth_posterior(
            key=subk[c], c=c, n_vars=n_vars,
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
            load=False, verbose=True)

        variants_x.append(jnp.array(target.x))
        variants_log_posterior.append(target.log_posterior)
        variants_target.append(target)

    variants_x = jnp.array(variants_x)

    @jit
    def log_prior(single_w_prob):
        ''' p(G) 
            returns shape [1, ]
            will later be vmapped

            single_w_prob : [n_vars, n_vars] in [0, 1]
                encoding probabilities of edges
        '''
        # use last target here since prior doesn't change
        # no batch variable b
        return target.g_dist.unnormalized_log_prob_soft(soft_g=single_w_prob)

    # inference model
    if inference_model_str == 'bge':
        model = BGeJAX(**inference_model_kwargs)

    elif kwargs.inference_model == 'lingauss':
        model = LinearGaussianGaussianJAX(**inference_model_kwargs)
        
    else:
        raise NotImplementedError()

    @jit
    def log_target(single_w, b):
        ''' p(D | G) 
            returns shape [1, ]
            will later be vmapped
            
            single_w:   [n_vars, n_vars] in {0, 1}
            b:          int indicating batch
        '''
        v = b % n_variants
        score = model.log_marginal_likelihood_given_g(w=single_w, data=variants_x[v])
        return score


    '''SVGD'''
    

    # assuming dot product representation of graph
    key, subk = random.split(key)
    if graph_embedding_representation:
        latent_prior_std = 1.0 / jnp.sqrt(n_dim) # results expected const. norm/inner product
        init_particles = random.normal(subk, shape=(batch_size, n_particles, n_vars, n_dim, 2)) * latent_prior_std
    else:
        latent_prior_std = 1.0
        init_particles = random.normal(subk, shape=(batch_size, n_particles, n_vars, n_vars))


    # svgd
    svgd = BatchedDotProductGraphSVGD(
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

    # evaluates [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
    eltwise_log_marg_likelihood = jit(vmap(lambda w, x_: model.log_marginal_likelihood_given_g(w=w, data=x_), (0, None), 0))

    # metrics
    mmd_kernel = StructuralHammingSquaredExponentialKernel(h=1.0)
    # metrics = svgd.make_metrics_ground_truth(variants_log_posterior, mmd_kernel=mmd_kernel, n_mmd_samples=100)
    metrics = svgd.make_metrics(variants_target, eltwise_log_marg_likelihood)

    # evaluate
    key, subk = random.split(key)
    particles = svgd.sample_particles(
        key=subk,
        n_steps=n_steps, 
        init_particles=init_particles.copy(), 
        metric_every=metric_every,
        eval_metrics=[
            metrics['neg_ave_log_marginal_likelihood'],
            metrics['edge_belief'],
        ])
    



    
