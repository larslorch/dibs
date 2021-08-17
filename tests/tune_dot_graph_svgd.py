import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging


import collections
import pprint
import json
import tqdm
import pickle
import igraph as ig
import numpy as onp
from tabulate import tabulate
from varname import nameof
import time
import argparse
import shutil
import multiprocessing

import jax.scipy.stats as jstats
import jax.scipy as jsp
import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
import jax.nn as nn
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_mul, index_update
import jax.lax as lax 
from jax import device_put


from dibs.graph.distributions import UniformDAGDistributionRejection
from dibs.utils.graph import *

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX
from dibs.models.FCGaussian import FCGaussianJAX

from dibs.svgd.dot_graph_svgd import DotProductGraphSVGD
from dibs.svgd.batch_dot_graph_svgd import BatchedDotProductGraphSVGD
from dibs.svgd.batch_joint_dot_graph_svgd import BatchedJointDotProductGraphSVGD

from dibs.kernel.basic import (
    FrobeniusSquaredExponentialKernel, 
    AngularSquaredExponentialKernel, 
)
from dibs.kernel.joint import (
    JointAdditiveFrobeniusSEKernel, 
    JointMultiplicativeFrobeniusSEKernel, 
)

from dibs.eval.mmd import MaximumMeanDiscrepancy

from dibs.eval.target import make_ground_truth_posterior, parse_target_str
from dibs.eval.tune import compute_tune_metrics, compute_tune_metrics_joint

from dibs.utils.version_control import get_version_datetime, get_version
from dibs.utils.system import str2bool

import numpy as np

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch # Tree-Parzen
from ray.tune.schedulers import AsyncHyperBandScheduler # early stopping of bad trials
from ray.tune import CLIReporter

import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype")
jnp.set_printoptions(precision=4, suppress=True)

STORE_ROOT = ['tuning']


def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--smoke_test", action="store_true", help="If passed, minimal iterations to see if something breaks") 
    parser.add_argument("--descr", required=True, help="set experiment filename; keep the same to resume in case interrupted")

    # tune
    parser.add_argument("--joint", action="store_true", help="If true, tunes inference of /joint/ posterior p(G, theta | D)") 

    parser.add_argument("--tune_stepsize", action="store_true", help="If true, stepsize is tuned") 
    # parser.add_argument("--tune_alpha_mode", default="linear", choices=["const", "linear", "expo"]) 
    # parser.add_argument("--tune_beta_mode", default="linear", choices=["const", "linear", "expo"]) 
    # parser.add_argument("--tune_tau_mode", default="const", choices=["const", "linear", "expo"]) 

    parser.add_argument("--num_tune_samples", type=int, default=300, help="number of configurations tested")
    parser.add_argument("--resume", action="store_true", help="If true, resumes experiment") 
    parser.add_argument("--objective_metric", default="neg_test_ave_log_marginal_likelihood_mixture",
        choices=[
            "log_kl_hard", "log_kl_mixture", 
            "neg_test_ave_log_marginal_likelihood_hard", "neg_test_ave_log_marginal_likelihood_mixture",
            "neg_test_ave_log_likelihood_hard", "neg_test_ave_log_likelihood_mixture",
            "neg_test_log_posterior_pred_hard", "neg_test_log_posterior_pred_mixture",
            "symlog_neg_test_ave_log_marginal_likelihood_hard", "symlog_neg_test_ave_log_marginal_likelihood_mixture",
            "symlog_neg_test_ave_log_likelihood_hard", "symlog_neg_test_ave_log_likelihood_mixture",
            "symlog_neg_test_log_posterior_pred_hard", "symlog_neg_test_log_posterior_pred_mixture",
            "edge_belief_hard", "edge_belief_mixture",
            "log_edge_belief_hard", "log_edge_belief_mixture",
            "shd_hard", "shd_mixture",
            "log_shd_hard", "log_shd_mixture",
            "neg_roc_auc_hard", "neg_roc_auc_mixture",
        ], 
        help="Objective metric") 
    parser.add_argument("--batch_size", type=int, default=8, help="number of random restarts per configuration run in batch") 
    parser.add_argument("--cpu_per_trial", type=int, default=2, help="number of cpus allocated per parallel configuration run") 

    # svgd fixed
    parser.add_argument("--n_steps", type=int, default=3000, help="svgd maximum steps")
    parser.add_argument("--n_grace_period", type=int, default=1000, help="svgd minimum steps")
    parser.add_argument("--n_particles", type=int, default=10, help="svgd particles")
    parser.add_argument("--latent_dim", type=int, help="svgd latent dim")
    parser.add_argument("--fix_rotation", default="not", choices=["not", "parallel", "orthogonal"], help="whether and how to fix u0 = v0")
    parser.add_argument("--n_grad_mc_samples", type=int, default=128, help="svgd score function grad estimator samples")
    parser.add_argument("--n_acyclicity_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--constraint_prior_graph_sampling", default="soft", choices=[None, "soft", "hard"], help="acyclicity constraint sampling")
    parser.add_argument("--score_function_baseline", type=float, default=0.0, help="gradient estimator baseline; 0.0 corresponds to not using a baseline")
    parser.add_argument("--metric_every", type=int, default=100, help="frequency of metric evaluation")
    parser.add_argument("--kernel", default="frob", choices=["frob"], help="kernel")
    parser.add_argument("--kernel_joint", default="additive-frob", choices=["additive-frob", "multiplicative-frob"], help="joint kernel")
    parser.add_argument("--soft_graph_mask", type=str2bool, default=False, help="whether joint kernel (soft-)masks unused parameters")
    parser.add_argument("--opt_name", default="rmsprop", choices=["rmsprop", "adam", "gd", "adagrad", "momentum"], help="optimizer")
    parser.add_argument("--opt_stepsize", type=float, default=0.005, help="learning rate")
    parser.add_argument("--rel_init_scale", type=float, default=1.0, help="initial scaling of svgd latent prior samples")

    parser.add_argument("--grad_estimator_x", default="reparam_soft", choices=["score", "reparam_soft", "reparam_hard"], help="gradient estimator for x in joint inference")
    parser.add_argument("--grad_estimator_theta", default="hard", choices=["hard"], help="gradient estimator for theta in joint inference")

    parser.add_argument("--alpha_linear", type=float, default=1.0, help="alpha linear default")
    parser.add_argument("--beta_linear", type=float, default=1.0, help="beta linear default")
    parser.add_argument("--tau_linear", type=float, default=1.0, help="tau linear default")

    parser.add_argument("--alpha_expo", type=float, default=0.0, help="alpha expo default")
    parser.add_argument("--beta_expo", type=float, default=0.0, help="beta expo default")
    parser.add_argument("--tau_expo", type=float, default=0.0, help="tau expo default")

    parser.add_argument("--ceil_alpha", type=float, default=1e9, help="maximum value for alpha")
    parser.add_argument("--ceil_beta", type=float, default=1e9, help="maximum value for beta")
    parser.add_argument("--ceil_tau", type=float, default=1e9, help="maximum value for tau")

    # generative model    
    parser.add_argument("--graph_prior", default="er", choices=["er", "sf"], help="inference model")
    parser.add_argument("--inference_model", default="bge", choices=["bge", "lingauss"], help="inference model")
    parser.add_argument("--joint_inference_model", default="lingauss", choices=["lingauss", "fcgauss"], help="joint inference model")

    parser.add_argument("--n_vars", type=int, default=5, help="number of variables in graph")
    parser.add_argument("--n_observations", type=int, default=100, help="number of observations defining the ground truth posterior")
    parser.add_argument("--n_ho_observations", type=int, default=100, help="number of held out observations for validation")
    parser.add_argument("--n_intervention_sets", type=int, default=10, help="number of sets of observations sampled with random interventions")
    parser.add_argument("--perc_intervened", type=float, default=0.2, help="percentage of nodes intervened upon")

    parser.add_argument("--n_posterior_g_samples", type=int, default=100, help="number of ground truth graph samples")
    parser.add_argument("--n_variants", type=int, default=1, help="number of different targets optimized and averaged; equally assigned to batch rollouts")

    # inference model
    parser.add_argument("--gbn_lower", type=float, default=1.0, help="GBN Sampler")
    parser.add_argument("--gbn_upper", type=float, default=3.0, help="GBN Sampler")
    parser.add_argument("--gbn_node_mean", type=float, default=0.0, help="GBN Sampler")
    parser.add_argument("--gbn_node_sig", type=float, default=1.0, help="GBN Sampler")
    parser.add_argument("--gbn_obs_sig", type=float, default=0.1, help="GBN Sampler")

    parser.add_argument("--lingauss_obs_noise", type=float, default=0.1, help="linear Gaussian")
    parser.add_argument("--lingauss_mean_edge", type=float, default=0.0, help="linear Gaussian")
    parser.add_argument("--lingauss_sig_edge", type=float, default=1.0, help="linear Gaussian")

    parser.add_argument("--fcgauss_obs_noise", type=float, default=0.1, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_sig_param", type=float, default=1.0, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_hidden_layers", type=int, default=1, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_n_neurons", type=int, default=5, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_activation", type=str, default="relu", help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_bias", type=str2bool, default=True, help="fully-connected NN Gaussian")

    parser.add_argument("--bge_alpha_mu", type=float, default=1.0, help="BGe")
    parser.add_argument("--bge_alpha_lambd_add", type=float, default=2.0, help="BGe")

    parser.add_argument("--create_targets_from_scratch", action="store_true")

    # ablation study
    parser.add_argument("--graph_embedding_representation", type=str2bool, default=True, help="whether to use graph embedding representation")

    return parser


def qloguniform(lower, upper, q, rng, n=1e6):
    """Reproduces tune.qloguniform, which is buggy"""

    raw = jnp.exp(random.uniform(rng, minval=jnp.log(jnp.array(lower)), 
                                      maxval=jnp.log(jnp.array(upper)),
                                      shape=(n, )))

    rounded = jnp.around(raw / q, decimals=0) * q
    rounded = jnp.where(rounded < lower, lower, rounded)
    rounded = jnp.where(rounded > upper, upper, rounded)

    return tune.choice(rounded.tolist())


if __name__ == '__main__':

    # # debugging
    # ray.init(num_cpus=multiprocessing.cpu_count(), local_mode=True)

    '''
    This script tunes Graph SVGD idea for at most 5 nodes
    '''

    parser = make_parser()
    kwargs = parser.parse_args()

    key = random.PRNGKey(kwargs.seed)

    # add current timestamp and version
    descr = kwargs.descr + get_version()

    # svgd
    if kwargs.smoke_test:
        kwargs.num_tune_samples = 1
        kwargs.batch_size = 2
        kwargs.n_variants = 2

    if kwargs.joint:
        tune_options = {
            'alpha_linear':         tune.loguniform(0.001, 1.0), # smaller than marginal
            'beta_linear':          tune.loguniform(0.01, 50.0),  # bigger than marginal
            'tau_const':            tune.loguniform(1.0, 100.0),
            'opt_stepsize':         tune.loguniform(1e-3, 5e-2),
            'kernel_joint':         tune.choice(['additive-frob', 'multiplicative-frob']),
            'grad_estimator_x':     tune.choice(['score', 'reparam_soft', 'reparam_hard']),
            'soft_graph_mask':      tune.choice([False, True]),
            'h_latent':             tune.loguniform(1.0, 20.0),
            'h_theta':              tune.loguniform(1.0, 10000.0),
        }
        tune_selection = [
            'alpha_linear',
            # 'beta_linear',
            'h_latent', 
            'h_theta', 
        ]

    else:
        tune_options = {
            'alpha_linear':         tune.loguniform(0.1, 50.0),
            'beta_linear':          tune.loguniform(0.001, 5.0),
            'tau_const':            tune.loguniform(1.0, 100.0),
            'opt_stepsize':         tune.loguniform(1e-3, 5e-2),
            'h':                    tune.loguniform(1.0, 100.0),
            # 'h':                    qloguniform(1, 100, q=1, rng=key),
        }
        tune_selection = [
            'alpha_linear',
            # 'beta_linear',
            'h',
        ]

    # tune options
    if kwargs.tune_stepsize:
        tune_selection += ['opt_stepsize']

    tuned = {k: tune_options[k] for k in tune_selection}


    '''Generative and inference models'''
    (generative_model_str, generative_model_kwargs), (inference_model_str, inference_model_kwargs) = parse_target_str(kwargs)

    # inference model
    if inference_model_str == 'bge':
        model = BGeJAX(**inference_model_kwargs)
    elif inference_model_str == 'lingauss':
        model = LinearGaussianGaussianJAX(**inference_model_kwargs)
    elif inference_model_str == 'fcgauss':
        model = FCGaussianJAX(**inference_model_kwargs)
    else:
        raise NotImplementedError()

    # create target
    variants_x = []
    variants_target = []
    key, *subk = random.split(key, kwargs.n_variants + 1)
    for c in range(kwargs.n_variants):
        target = make_ground_truth_posterior(
            c=c, key=subk[c], n_vars=kwargs.n_vars,
            graph_prior_str=kwargs.graph_prior,
            generative_model_str=generative_model_str,
            generative_model_kwargs=generative_model_kwargs,
            inference_model_str=inference_model_str,
            inference_model_kwargs=inference_model_kwargs,
            n_observations=kwargs.n_observations,
            n_ho_observations=kwargs.n_ho_observations,
            n_posterior_g_samples=kwargs.n_posterior_g_samples,
            n_intervention_sets=kwargs.n_intervention_sets,
            perc_intervened=kwargs.perc_intervened,
            load=False, verbose=True)

        variants_x.append(jnp.array(target.x))
        variants_target.append(target)

    variants_x = jnp.array(variants_x)

    # eval metric objects
    batch_g_gt = [jnp.array(variants_target[b % kwargs.n_variants].g) for b in range(kwargs.batch_size)]
    batch_x_train = [jnp.array(variants_target[b % kwargs.n_variants].x) for b in range(kwargs.batch_size)]
    batch_x_ho = [jnp.array(variants_target[b % kwargs.n_variants].x_ho) for b in range(kwargs.batch_size)]
    batch_x_interv = [variants_target[b % kwargs.n_variants].x_interv for b in range(kwargs.batch_size)]

    key = jnp.array(key) # ray/tune pickle breaks without this

    # by construction
    if kwargs.graph_embedding_representation:
        print(f'n_vars : {kwargs.n_vars}  latent_dim: {kwargs.latent_dim}')
        latent_prior_std = 1.0 / jnp.sqrt(kwargs.latent_dim) # results in expected const. norm/inner product
    else:
        print(f'n_vars : {kwargs.n_vars}')
        latent_prior_std = 1.0         

    if kwargs.joint:
        # evaluates [n_particles, n_vars, n_vars], [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
        no_interv_targets = jnp.zeros(kwargs.n_vars).astype(bool)
        eltwise_log_likelihood = jit(vmap(
            lambda w_, theta_, x_, interv_targets_: (
                model.log_likelihood(theta=theta_, w=w_, data=x_, interv_targets=interv_targets_)
            ), (0, 0, None, None), 0))

        def log_joint_target_no_batch(single_w, single_theta, b):
            ''' p(theta, D | G) 
                returns shape [1, ]
                will later be vmapped

                single_w :      [n_vars, n_vars]
                single_theta:   [n_vars, n_vars]
            '''
            v = jnp.mod(b, kwargs.n_variants)
            log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
            log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=device_put(variants_x)[v], interv_targets=no_interv_targets)
            return log_prob_theta + log_lik
        
        # no minibatching during tune
        def log_joint_target(single_w, single_theta, b, subk): 
            return log_joint_target_no_batch(single_w, single_theta, b)

    else:
        # evaluates [n_particles, n_vars, n_vars], [n_observations, n_vars], [n_vars] in batch on held-out data 
        eltwise_log_marg_likelihood = jit(vmap(lambda w_, x_, interv_targets_: model.log_marginal_likelihood_given_g(
            w=w_, data=x_, interv_targets=interv_targets_), (0, None, None), 0))

        def log_target(single_w, b):
            ''' p(D | G) 
                returns shape [1, ]
                will later be vmapped
                
                single_w:   [n_vars, n_vars] in {0, 1}
                b:          int indicating batch
            '''
            v = jnp.mod(b, kwargs.n_variants)
            score = model.log_marginal_likelihood_given_g(w=single_w, data=device_put(variants_x)[v])
            return score
            

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

    def train_func(config):
        '''
        Executes SVGD given tune `config`
        '''

        # initialize particles
        if kwargs.joint:
            local_key, subk = random.split(key)
            if kwargs.graph_embedding_representation:
                init_particles_x = random.normal(subk, shape=(kwargs.batch_size, kwargs.n_particles, kwargs.n_vars, kwargs.latent_dim, 2)) * latent_prior_std * kwargs.rel_init_scale
            else:
                init_particles_x = random.normal(subk, shape=(kwargs.batch_size, kwargs.n_particles, kwargs.n_vars, kwargs.n_vars)) * latent_prior_std * kwargs.rel_init_scale

            local_key, subk = random.split(local_key)
            init_particles_theta = model.init_parameters(key=subk, n_particles=kwargs.n_particles, n_vars=kwargs.n_vars, batch_size=kwargs.batch_size)
            
        else:
            local_key, subk = random.split(key)
            if kwargs.graph_embedding_representation:
                init_particles = random.normal(subk, shape=(kwargs.batch_size, kwargs.n_particles, kwargs.n_vars, kwargs.latent_dim, 2)) * latent_prior_std * kwargs.rel_init_scale
            else:
                init_particles = random.normal(subk, shape=(kwargs.batch_size, kwargs.n_particles, kwargs.n_vars, kwargs.n_vars)) * latent_prior_std * kwargs.rel_init_scale

        # kernel
        if kwargs.joint:
            kernel_joint_choice = config.get('kernel_joint', kwargs.kernel_joint)
            singular_dim_theta = 3 if kwargs.joint_inference_model == 'fcgauss' else 2
            if kernel_joint_choice == 'additive-frob':
                kernel = JointAdditiveFrobeniusSEKernel(
                    h_latent=config.get('h_latent', 1.0),
                    h_theta=config.get('h_theta', 1.0),
                    scale_latent=config.get('kernel_scale_latent', 1.0),
                    scale_theta=config.get('kernel_scale_theta', 1.0),
                    soft_graph_mask=config.get('soft_graph_mask', kwargs.soft_graph_mask),
                    singular_dim_theta=singular_dim_theta,
                    graph_embedding_representation=kwargs.graph_embedding_representation)

            elif kernel_joint_choice == 'multiplicative-frob':
                kernel = JointMultiplicativeFrobeniusSEKernel(
                    h_latent=config.get('h_latent', 1.0),
                    h_theta=config.get('h_theta', 1.0),
                    scale=config.get('kernel_scale_joint', 1.0),
                    soft_graph_mask=config.get('soft_graph_mask', kwargs.soft_graph_mask),
                    singular_dim_theta=singular_dim_theta,
                    graph_embedding_representation=kwargs.graph_embedding_representation)

            else:
                raise ValueError('Invalid joint kernel identifier')

        else:
            if kwargs.kernel == 'frob':
                kernel = FrobeniusSquaredExponentialKernel(
                    h=config.get('h', 1.0),
                    scale=config.get('kernel_scale', 1.0),
                    graph_embedding_representation=kwargs.graph_embedding_representation)
                    
            else:
                raise ValueError('Invalid kernel identifier')

        # temperature hyperparameters
        def linear_alpha(t):
            return jnp.minimum((config.get('alpha_linear', kwargs.alpha_linear) * t) + config.get('alpha', 0.0), kwargs.ceil_alpha)

        def linear_beta(t):
            return jnp.minimum((config.get('beta_linear', kwargs.beta_linear) * t) + config.get('beta', 0.0), kwargs.ceil_beta)

        def linear_tau(t):
            return jnp.minimum((config.get('tau_linear', kwargs.tau_linear) * t) + config.get('tau', 0.0), kwargs.ceil_tau)

        
        def exponential_alpha(t):
            return jnp.minimum(jnp.exp(config.get('alpha_expo', kwargs.alpha_expo) * t) * config.get('alpha', 1.0), kwargs.ceil_alpha)

        def exponential_beta(t):
            return jnp.minimum(jnp.exp(config.get('beta_expo', kwargs.beta_expo) * t) * config.get('beta', 1.0), kwargs.ceil_beta)

        def exponential_tau(t):
            return jnp.minimum(jnp.exp(config.get('tau_expo', kwargs.tau_expo) * t) * config.get('tau', 1.0), kwargs.ceil_tau)


        if 'alpha_linear' in config.keys():
            alpha_sched = linear_alpha
        elif 'alpha_expo' in config.keys():
            alpha_sched = exponential_alpha
        else:
            alpha_sched = lambda _: jnp.array([config.get('alpha_const', 1.0)])

        if 'beta_expo' in config.keys():
            beta_sched = exponential_beta
        else:
            beta_sched = linear_beta

        tau_sched = lambda _: jnp.array([config.get('tau_const', 1.0)])
        gamma_sched = lambda _: jnp.array([config.get('gamma_const', 1.0)])

        # optimizer
        optimizer = dict(
            name=config.get('opt_name', kwargs.opt_name),
            stepsize=config.get('opt_stepsize', kwargs.opt_stepsize),
        )

        # initliaze svgd
        if kwargs.joint:
            svgd = BatchedJointDotProductGraphSVGD(
                n_vars=kwargs.n_vars,
                n_dim=kwargs.latent_dim,
                optimizer=optimizer,
                kernel=kernel, 
                target_log_prior=log_prior,
                target_log_joint_prob=log_joint_target,
                target_log_joint_prob_no_batch=log_joint_target_no_batch,
                alpha=alpha_sched,
                beta=beta_sched,
                gamma=gamma_sched,
                tau=tau_sched,
                n_grad_mc_samples=config.get('n_grad_mc_samples', kwargs.n_grad_mc_samples),
                n_acyclicity_mc_samples=config.get('n_acyclicity_mc_samples', kwargs.n_acyclicity_mc_samples),
                score_function_baseline=config.get('score_function_baseline', kwargs.score_function_baseline),
                clip=None,
                fix_rotation=kwargs.fix_rotation,
                grad_estimator_x=config.get('grad_estimator_x', kwargs.grad_estimator_x),
                grad_estimator_theta=config.get('grad_estimator_theta', kwargs.grad_estimator_theta),
                repulsion_in_prob_space=False,
                latent_prior_std=latent_prior_std,
                constraint_prior_graph_sampling=kwargs.constraint_prior_graph_sampling,
                graph_embedding_representation=kwargs.graph_embedding_representation,
                verbose=False)

        else:
            svgd = BatchedDotProductGraphSVGD(
                n_vars=kwargs.n_vars,
                n_dim=kwargs.latent_dim,
                optimizer=optimizer,
                kernel=kernel, 
                target_log_prior=log_prior,
                target_log_prob=log_target,
                alpha=alpha_sched,
                beta=beta_sched,
                gamma=gamma_sched,
                tau=tau_sched,
                n_grad_mc_samples=config.get('n_grad_mc_samples', kwargs.n_grad_mc_samples),
                n_acyclicity_mc_samples=config.get('n_acyclicity_mc_samples', kwargs.n_acyclicity_mc_samples),
                score_function_baseline=config.get('score_function_baseline', kwargs.score_function_baseline),
                clip=None,
                fix_rotation=kwargs.fix_rotation,
                grad_estimator='score',
                repulsion_in_prob_space=False,
                latent_prior_std=latent_prior_std,
                constraint_prior_graph_sampling=kwargs.constraint_prior_graph_sampling,
                graph_embedding_representation=kwargs.graph_embedding_representation,
                verbose=False)

        # run
        if kwargs.joint:
            def metrics(params):
                m = compute_tune_metrics_joint(params=params, kwargs=kwargs, svgd=svgd,
                    variants_target=variants_target, eltwise_log_likelihood=eltwise_log_likelihood,
                    batch_g_gt=batch_g_gt, batch_x_train=batch_x_train, batch_x_ho=batch_x_ho, 
                    batch_x_interv=batch_x_interv)
                tune.report(**m)
                return
        else:
            def metrics(params):
                m = compute_tune_metrics(params=params, kwargs=kwargs, svgd=svgd,
                    variants_target=variants_target, eltwise_log_marg_likelihood=eltwise_log_marg_likelihood,
                    batch_g_gt=batch_g_gt, batch_x_train=batch_x_train, batch_x_ho=batch_x_ho, 
                    batch_x_interv=batch_x_interv)
                tune.report(**m)
                return

        # run with config
        if kwargs.joint:
            particles = svgd.sample_particles(
                key=local_key, 
                n_steps=100000, # will be stopped by scheduler 
                init_particles_x=init_particles_x.copy(), 
                init_particles_theta=init_particles_theta.copy(), 
                metric_every=1 if kwargs.smoke_test else kwargs.metric_every,
                tune_metrics=[metrics])
        else:
            particles = svgd.sample_particles(
                key=local_key, 
                n_steps=100000, # will be stopped by scheduler 
                init_particles=init_particles.copy(), 
                metric_every=1 if kwargs.smoke_test else kwargs.metric_every,
                tune_metrics=[metrics])


    '''
    Run hyperparameter tuning
    '''
    objective_mode = "min"

    search_alg = HyperOptSearch(metric=kwargs.objective_metric, mode=objective_mode)

    local_dir = os.path.abspath(os.path.join(
        '..', *STORE_ROOT,
    ))
    exp_dir = os.path.join(local_dir, descr)
    if os.path.exists(exp_dir) and os.path.isdir(exp_dir) and not kwargs.resume:
        print('Deleted existing tuning folder.')
        shutil.rmtree(exp_dir)
        

    ahb = AsyncHyperBandScheduler(
        time_attr="svgd_steps",
        grace_period=1 if kwargs.smoke_test else kwargs.n_grace_period,
        max_t=2 if kwargs.smoke_test else kwargs.n_steps)

    analysis = tune.run(
        train_func,
        config=tuned,
        num_samples=kwargs.num_tune_samples,
        scheduler=ahb,
        local_dir=local_dir,
        search_alg=search_alg,
        progress_reporter=CLIReporter(
            parameter_columns=['alpha_slope'], 
            metric_columns=['unique_graphs'],
            max_report_frequency=30, # seconds
            ),
        name=descr,
        resources_per_trial={
            "cpu": kwargs.cpu_per_trial,
            "gpu": 0
        },
        verbose=1,
        metric=kwargs.objective_metric,
        mode=objective_mode,
        resume=kwargs.resume, # "PROMPT" would ask whether to start new experiment or resume from existing folder
    )

    print(kwargs.objective_metric)

    print('Best configuration:')
    pprint.pprint(analysis.get_best_config(
        metric=kwargs.objective_metric, mode=objective_mode, scope="last-5-avg"))

    pprint.pprint(analysis.get_best_trial(
        metric=kwargs.objective_metric, mode=objective_mode, scope="last-5-avg"))

    # Save a dataframe for analyzing trial results
    df = analysis.results_df
    df_sorted = df.sort_values(by=[kwargs.objective_metric])
    df_sorted.to_csv(os.path.join(local_dir, descr + '.csv'))



