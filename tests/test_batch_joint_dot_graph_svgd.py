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
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.utils.graph import *

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX
from dibs.models.FCGaussian import FCGaussianJAX

from dibs.svgd.batch_joint_dot_graph_svgd import BatchedJointDotProductGraphSVGD

from dibs.kernel.joint import (
    JointMultiplicativeFrobeniusSEKernel,
    JointAdditiveFrobeniusSEKernel,
)

from dibs.utils.func import mask_topk, id2bit, bit2id, log_prob_ids, particle_joint_empirical

from dibs.eval.target import make_ground_truth_posterior

from config.svgd import joint_config

import numpy as onp

import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype")


if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)

    key = random.PRNGKey(0)
    
    '''
    This script tests Graph SVGD idea for at most 5 nodes
    '''
    batch_size = 2
    n_variants = 2

    # target
    n_vars = 20
    n_dim = 10
    verbose = True
    sparsity_factor = 1.0

    n_observations = 20
    n_ho_observations = 100
    n_posterior_g_samples = 100
    n_intervention_sets = 10
    perc_intervened = 0.1

    # mode = 'lingauss'
    mode = 'fcgauss'

    n_grad_batch_size = 4

    graph_embedding_representation = True
    # graph_embedding_representation = False

    graph_prior_str = 'er'

    if mode == 'lingauss':

        # Linear Gaussian
        obs_noise = 0.1
        mean_edge = 0.0
        sig_edge = 1.0
        singular_dim_theta = 2

        generative_model_str = 'lingauss'
        generative_model_kwargs = dict(
            obs_noise=obs_noise,
            mean_edge=mean_edge,
            sig_edge=sig_edge,
        )

        inference_model_str = 'lingauss'
        inference_model_kwargs = dict(
            obs_noise=obs_noise,
            mean_edge=mean_edge,
            sig_edge=sig_edge,
        )

    else:
        # FC Gaussian
        obs_noise = 0.1
        sig_param = 1.0
        dims = [5, 5]
        singular_dim_theta = 3

        generative_model_str = 'fcgauss'
        generative_model_kwargs = dict(
            obs_noise=obs_noise,
            sig_param=sig_param,
            dims=dims,
        )

        inference_model_str = 'fcgauss'
        inference_model_kwargs = dict(
            obs_noise=obs_noise,
            sig_param=sig_param,
            dims=dims,
        )


    # grad_estimator_x = 'score'
    grad_estimator_x = 'reparam_soft'
    # grad_estimator_x = 'reparam_hard'

    grad_estimator_theta = 'hard'

    soft_graph_mask = False


    '''SVGD'''
    tuned = joint_config[inference_model_str][n_vars if n_vars in joint_config['lingauss'].keys() else 20]
    
    n_steps = 1000
    n_particles = 20
    n_grad_mc_samples = 128
    n_acyclicity_mc_samples = 32
    optimizer = dict(name='rmsprop', stepsize=tuned.get("opt_stepsize", 0.005))
    score_function_baseline = 0.0

    metric_every = 50

    constraint_prior_graph_sampling = 'soft'

    fix_rotation = "not"

    kernel = JointAdditiveFrobeniusSEKernel(
        h_latent=tuned['h_latent'],
        h_theta=tuned['h_theta'],
        scale_latent=tuned.get('kernel_scale_latent', 1.0),
        scale_theta=tuned.get('kernel_scale_theta', 1.0),
        soft_graph_mask=soft_graph_mask,
        singular_dim_theta=singular_dim_theta,
        graph_embedding_representation=graph_embedding_representation)

    # temperature hyperparameters
    def linear_alpha(t):
        return (tuned.get('alpha_slope', 0.5) * t) + tuned.get('alpha', 0.0)

    def linear_beta(t):
        return (tuned.get('beta_slope', 0.5) * t) + tuned.get('beta', 0.0)

    def const_gamma(t):
        return jnp.array([1.0])

    def const_tau(t):
        return jnp.array([1.0])

    alpha_sched = linear_alpha
    beta_sched = linear_beta
    gamma_sched = const_gamma
    tau_sched = const_tau

    # temperature schedule
    print_sched = 5
    ts = jnp.arange(0, int(n_steps*(print_sched + 1) / print_sched),
                    step=int(n_steps/print_sched))

    print(tabulate(
        [['t', *['{:10d}'.format(t) for t in ts]],
         ['alpha', *['{:10.4f}'.format(b) for b in alpha_sched(ts)]],
         ['beta', *['{:10.4f}'.format(b) for b in beta_sched(ts)]],
         ['tau', *['{:10.4f}'.format(b) for b in tau_sched(ts)]],
         ]))

    '''Target'''
    variants_x = []
    variants_target = []
    key, *subk = random.split(key, n_variants + 1)
    for c in range(n_variants):
        target = make_ground_truth_posterior(
            key=subk[c], c=c, n_vars=n_vars,
            graph_prior_str=graph_prior_str,
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
        return target.g_dist.unnormalized_log_prob_soft(soft_g=single_w_prob)

    # inference model
    if inference_model_str == 'bge':
        model = BGeJAX(**inference_model_kwargs)

    elif inference_model_str == 'lingauss':
        model = LinearGaussianGaussianJAX(**inference_model_kwargs)

    elif inference_model_str == 'fcgauss':
        model = FCGaussianJAX(**inference_model_kwargs)
        
    else:
        raise NotImplementedError()


    x = jnp.array(target.x)
    x_ho = jnp.array(target.x_ho)

    # evaluates [n_particles, n_vars, n_vars], [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
    no_interv_targets = jnp.zeros(n_vars).astype(bool)
    eltwise_log_likelihood = jit(vmap(
        lambda w_, theta_, x_: (
            model.log_likelihood(theta=theta_, w=w_, data=x_, interv_targets=no_interv_targets)
        ), (0, 0, None), 0))

    def log_joint_target(single_w, single_theta, b, rng):
        ''' p(theta, D | G) 
            returns shape [1, ]
            will later be vmapped

            single_w :      [n_vars, n_vars]
            single_theta:   [n_vars, n_vars]
            b:              int indicating batch
            rng:            [1,]
        '''
        v = b % n_variants
        x = variants_x[v]

        # minibatch
        idx = random.choice(rng, a=x.shape[0], shape=(min(n_grad_batch_size, x.shape[0]),), replace=False)
        x_batch = x[idx, :]

        # compute target
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=x_batch, interv_targets=no_interv_targets)

        return log_prob_theta + log_lik

    def log_joint_target_no_batch(single_w, single_theta, b):
        ''' Same as above but using full data; for metrics computed on the flu
        '''
        v = b % n_variants
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=variants_x[v], interv_targets=no_interv_targets)
        return log_prob_theta + log_lik

    '''SVGD'''

    # assuming dot product representation of graph
    key, subk = random.split(key)
    if graph_embedding_representation:
        latent_prior_std = 1.0 / jnp.sqrt(n_dim)
        init_particles_x = random.normal(subk, shape=(batch_size, n_particles, n_vars, n_dim, 2)) * latent_prior_std
    else:
        latent_prior_std = 1.0
        init_particles_x = random.normal(subk, shape=(batch_size, n_particles, n_vars, n_vars)) * latent_prior_std
    print('init_particles_x    ', init_particles_x.shape)

    key, subk = random.split(key)
    init_particles_theta = model.init_parameters(key=subk, n_particles=n_particles, n_vars=n_vars, batch_size=batch_size)
    theta_flat, theta_pytree = tree_flatten(init_particles_theta)

    print('init_particles_theta', theta_pytree)
    for th in theta_flat:
        print('\t', th.shape)
   
    # svgd
    svgd = BatchedJointDotProductGraphSVGD(
        n_vars=n_vars,
        n_dim=n_dim,
        optimizer=optimizer,
        kernel=kernel, 
        target_log_prior=log_prior,
        target_log_joint_prob=log_joint_target,
        target_log_joint_prob_no_batch=log_joint_target_no_batch,
        alpha=alpha_sched,
        beta=beta_sched,
        gamma=gamma_sched,
        tau=tau_sched,
        n_grad_mc_samples=n_grad_mc_samples,
        n_grad_batch_size=n_grad_batch_size,
        n_acyclicity_mc_samples=n_acyclicity_mc_samples,
        clip=None,
        fix_rotation=fix_rotation,
        grad_estimator_x=grad_estimator_x,
        grad_estimator_theta=grad_estimator_theta,
        score_function_baseline=score_function_baseline,
        repulsion_in_prob_space=False,
        latent_prior_std=latent_prior_std,
        constraint_prior_graph_sampling=constraint_prior_graph_sampling,
        graph_embedding_representation=graph_embedding_representation,
        verbose=True)

    # metrics
    metrics = svgd.make_metrics(
        variants_target=variants_target, 
        eltwise_log_likelihood=eltwise_log_likelihood)

    # evaluate
    key, subk = random.split(key)
    particles_x, particles_theta = svgd.sample_particles(
        key=subk,
        n_steps=n_steps, 
        init_particles_x=init_particles_x, 
        init_particles_theta=init_particles_theta, 
        metric_every=metric_every,
        # eval_metrics=[])
        eval_metrics=[
            metrics['neg_log_likelihood_metric'],
            metrics['edge_belief_metric'],
        ])


