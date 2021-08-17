import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import collections
import pprint 
import tqdm

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_mul, index_update

import numpy as onp

from dibs.graph.distributions import LowerTriangularDAGDistribution
from dibs.utils.graph import *
from dibs.utils.func import id2bit, bit2id, particle_joint_empirical, particle_joint_mixture
from dibs.utils.tree import tree_shapes
from dibs.utils.graph import mat_to_graph

from dibs.eval.target import make_ground_truth_posterior

from dibs.models.dirichletCategorical import BDeu
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.FCGaussian import FCGaussianJAX

from dibs.mcmc.joint_structure import MHJointStructureMCMC, GibbsJointStructureMCMC

from dibs.utils.func import mask_topk, id2bit, bit2id, log_prob_ids, particle_joint_empirical, particle_joint_mixture
from dibs.utils.tree import tree_unzip_leading, tree_zip_leading, tree_shapes

from dibs.eval.metrics import neg_log_joint_posterior_predictive


if __name__ == '__main__':

    key = random.PRNGKey(0)
    jnp.set_printoptions(precision=6, suppress=True)

    '''This script tests that the empirical (Dirac) distribution of 
    MCMC samples is able to converge to the closed form density for small p(G | D)
    '''

    burnin = 100
    thinning = 10
    n_samples = 30

    theta_prop_sig_mh = 0.001
    theta_prop_sig_gibbs = 0.001

    graph_prior_str = 'sf'


    '''Target observations'''
    n_vars = 50
    n_observations = 100
    
    model_str = 'lingauss'
    model_kwargs = dict(
        obs_noise=0.1, 
        mean_edge=0.0, 
        sig_edge=1.0,
    )

    key, subk = random.split(key)
    target = make_ground_truth_posterior(
        key=subk, c=1, n_vars=n_vars,
        graph_prior_str=graph_prior_str,
        generative_model_str=model_str,
        generative_model_kwargs=model_kwargs,
        inference_model_str=model_str,
        inference_model_kwargs=model_kwargs,
        n_observations=n_observations,
        n_ho_observations=100,
        n_posterior_g_samples=1,
        n_intervention_sets=10,
        perc_intervened=0.1,
        load=False,
        verbose=True)

    
    '''Joint inference model'''

    if model_str == 'lingauss':
        model = LinearGaussianGaussianJAX(**model_kwargs)
    elif model_str == 'fcgauss':
        model = FCGaussianJAX(**model_kwargs)
    else:
        raise ValueError()

    no_interv_targets = jnp.zeros(n_vars).astype(bool)

    def log_joint_target(g_mat, theta):
        return (target.g_dist.unnormalized_log_prob_soft(soft_g=g_mat)
                + model.log_prob_parameters(theta=theta, w=g_mat)
                + model.log_likelihood(theta=theta, w=g_mat, data=jnp.array(target.x), interv_targets=no_interv_targets))

    # evaluates [n_particles, n_vars, n_vars], [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
    eltwise_log_likelihood = jit(vmap(
        lambda w_, theta_, x_: (
            model.log_likelihood(theta=theta_, w=w_, data=x_, interv_targets=no_interv_targets)
        ), (0, 0, None), 0))

    # [L, d, d], PyTree with leading dim [L, ...] -> [L]
    def double_eltwise_log_joint_prob(g_mats, thetas):
        thetas_unzipped = tree_unzip_leading(thetas, g_mats.shape[0])
        return jnp.array([log_joint_target(g_mat, theta) for g_mat, theta in zip(g_mats, thetas_unzipped)])
        

    '''Run MCMC'''
    mh_mcmc = MHJointStructureMCMC(
        n_vars=n_vars,
        theta_prop_sig=theta_prop_sig_mh,
        only_non_covered=False)

    key, subk = random.split(key)
    g_samples_mh, theta_samples_mh = mh_mcmc.sample(key=subk, n_samples=n_samples,
        log_joint_target=log_joint_target, theta_shape=model.get_theta_shape(n_vars=n_vars),
        burnin=burnin, thinning=thinning)

    gibbs_mcmc = GibbsJointStructureMCMC(
        n_vars=n_vars,
        theta_prop_sig=theta_prop_sig_gibbs,
        only_non_covered=False)

    g_samples_gibbs, theta_samples_gibbs = gibbs_mcmc.sample(key=subk, n_samples=n_samples,
        log_joint_target=log_joint_target, theta_shape=model.get_theta_shape(n_vars=n_vars),
        burnin=burnin, thinning=thinning)


    '''Eval'''
    print_steps = 5

    print()
    print('Neg. posterior predictive')
    for desc, g_samples, theta_samples in [
        ('MH', g_samples_mh, theta_samples_mh),
        ('MH-within-Gibbs', g_samples_gibbs, theta_samples_gibbs),
    ]:
        print(desc)
        for cutoff in jnp.linspace(n_samples/print_steps, n_samples, print_steps, endpoint=True).astype(jnp.int32):
            if not cutoff:
                continue
            
            g_samples_cutoff = g_samples[:cutoff, :, :]
            theta_samples_cutoff = tree_zip_leading(tree_unzip_leading(theta_samples, n_samples)[:cutoff])

            empirical = particle_joint_empirical(g_samples_cutoff, theta_samples_cutoff)
            mixture = particle_joint_mixture(g_samples_cutoff, theta_samples_cutoff, double_eltwise_log_joint_prob)

            pp = neg_log_joint_posterior_predictive(
                dist=empirical,
                eltwise_log_joint_target=eltwise_log_likelihood,
                x=jnp.array(target.x_ho)
            )

            pp_mixt = neg_log_joint_posterior_predictive(
                dist=mixture,
                eltwise_log_joint_target=eltwise_log_likelihood,
                x=jnp.array(target.x_ho)
            )

            print(f'Samples: {cutoff:4d}  Unique G: {len(onp.unique(mixture[0], axis=0)):4d}  {pp.item():12.2f} [empirical]   {pp_mixt.item():12.2f} [mixture]')
            # print(f'Samples: {cutoff:4d}  Unique G: {len(onp.unique(mixture[0], axis=0)):4d}   Mixture max p: {jnp.exp(mixture[2]).max().item():6.4f} {pp.item():12.2f} [empirical]   {pp_mixt.item():12.2f} [mixture]')
