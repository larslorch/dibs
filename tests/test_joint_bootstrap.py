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

from dibs.utils.func import mask_topk, id2bit, bit2id, log_prob_ids, particle_joint_empirical, particle_joint_mixture
from dibs.utils.tree import tree_index, tree_unzip_leading

from dibs.eval.target import make_ground_truth_posterior

from dibs.bootstrap.bootstrap import NonparametricDAGBootstrap
from dibs.bootstrap.learners import GES, PC


import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype")


if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)

    key = random.PRNGKey(0)
    c = 0
    
    '''
    This script tests DAG Boostrap + MLE
    '''

    n_boot_samples = 30
    n_boot_samples = 2
    
    # target
    n_vars = 20
    n_vars = 6

    n_observations = 20
    n_ho_observations = 100
    n_posterior_g_samples = 100
    n_intervention_sets = 10
    perc_intervened = 0.1

    # Linear Gaussian
    obs_noise = 0.1
    mean_edge = 0.0
    sig_edge = 1.0

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
        load=False, verbose=True)

    x = jnp.array(target.x)

    model = LinearGaussianGaussianJAX(**inference_model_kwargs)
   
    no_interv_targets = jnp.zeros(n_vars).astype(bool)

    def log_joint_target(g_mat, theta):
        return (target.g_dist.unnormalized_log_prob(g=mat_to_graph(g_mat))
                + model.log_prob_parameters(theta=theta, w=g_mat)
                + model.log_likelihood(theta=theta, w=g_mat, data=x, interv_targets=no_interv_targets))

    # [L, d, d], PyTree with leading dim [L, ...] -> [L]
    def double_eltwise_log_joint_prob(g_mats, thetas):
        thetas_unzipped = tree_unzip_leading(thetas, g_mats.shape[0])
        return jnp.array([log_joint_target(g_mat, theta) for g_mat, theta in zip(g_mats, thetas_unzipped)])
    

    print('GT theta')
    print(target.theta * target.g)

    '''Run MCMC'''
    # init DAG bootstrap
    # learner = GES()
    learner = PC(ci_test='gaussian', ci_alpha=0.05)
    dag_bootstrap = NonparametricDAGBootstrap(learner=learner, verbose=True)
    print(type(learner).__name__)

    # DAG boostrap
    key, subk = random.split(key)
    boot_samples = dag_bootstrap.sample_particles(key=subk, n_samples=n_boot_samples, x=x)

    # MLE parameters
    mle_kwargs = {
        'type':    'lingauss',
        'cov_mat':  (x.T @ x) / x.shape[0],
        'graphs':   boot_samples,
    }
    mle_parameters = dag_bootstrap.learner.get_mle_params(mle_kwargs)
    
    print('mle_parameters', mle_parameters.shape)

    # joint distribution
    boot_dist = particle_joint_mixture(boot_samples, mle_parameters, double_eltwise_log_joint_prob)



