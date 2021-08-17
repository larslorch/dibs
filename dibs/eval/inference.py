import os
import ray

import pandas as pd

import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap, vjp, jvp

from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX
from dibs.models.FCGaussian import FCGaussianJAX

from dibs.eval.target import parse_target_str, make_ground_truth_posterior

from dibs.eval.marginal_inference import (
    run_marginal_argmax,
    run_marginal_structMCMC,
    run_marginal_bootstrap,
    run_marginal_svgd,
)

from dibs.eval.joint_inference import (
    run_joint_argmax,
    run_joint_structMCMC,
    run_joint_bootstrap,
    run_joint_svgd,
)

from dibs.eval.result import eval
from dibs.utils.graph import mat_to_graph

STORE_ROOT = ['results']


@ray.remote
def eval_single_target(*, method, c, r, key, n_particles, n_particles_loop, graph_prior_str, kwargs, additional_descr=None, load=True):
    """
    Evaluate a single target based on random key `key` and seed `c`
    for a list of particles `n_particles_loop` using `make_ground_truth_posterior`
    """

    (generative_model_str, generative_model_kwargs), (inference_model_str, inference_model_kwargs) = parse_target_str(kwargs)

    key, subk = random.split(key)
    target = make_ground_truth_posterior(
        key=subk, c=c, n_vars=kwargs.n_vars, 
        graph_prior_str=graph_prior_str,
        generative_model_str=generative_model_str,
        generative_model_kwargs=generative_model_kwargs,
        inference_model_str=inference_model_str,
        inference_model_kwargs=inference_model_kwargs,
        n_observations=kwargs.n_observations,
        n_ho_observations=kwargs.n_ho_observations, 
        n_posterior_g_samples=kwargs.n_posterior_g_samples,
        n_intervention_sets=kwargs.n_intervention_sets,
        perc_intervened=kwargs.perc_intervened,
        load=load,
        real_data=kwargs.real_data,
        real_data_held_out=kwargs.real_data_held_out,
        real_data_normalize=kwargs.real_data_normalize,
        verbose=False)

    # init inference model using target graph dist
    if inference_model_str == 'bge':
        ig_model = BGe(
            **inference_model_kwargs,
            g_dist=target.g_dist,
            verbose=False)
        jax_model = BGeJAX(**inference_model_kwargs)

    elif inference_model_str == 'lingauss':
        ig_model = LinearGaussianGaussian(
            **inference_model_kwargs,
            g_dist=target.g_dist,
            verbose=False)
        jax_model = LinearGaussianGaussianJAX(**inference_model_kwargs)

    else:
        raise NotImplementedError()

    def ig_log_target(g_):
        return ig_model.g_dist.unnormalized_log_prob(g=g_) + \
            ig_model.log_marginal_likelihood_given_g(g=g_, x=jnp.array(target.x))

    def ig_log_target_single(g_, j_):
        return ig_model.g_dist.unnormalized_log_prob_single(g=g_, j=j_) + \
            ig_model.log_marginal_likelihood_given_g_single(g=g_, x=jnp.array(target.x), j=j_)


    def log_prior(single_w_prob):
        ''' p(G) 
            returns shape [1, ]
            will later be vmapped

            single_w_prob : [n_vars, n_vars] in [0, 1]
                encoding probabilities of edges
        '''
        # no batch variable b
        return target.g_dist.unnormalized_log_prob_soft(soft_g=single_w_prob)

    def log_target(single_w):
        ''' p(D | G) 
            returns shape [1, ]
            will later be vmapped
        '''
        score = jax_model.log_marginal_likelihood_given_g(w=single_w, data=jnp.array(target.x))
        return score

    # evaluates [n_particles, n_vars, n_vars], [n_observations, n_vars], [n_vars] in batch for x_ and interv_targets_
    eltwise_log_marg_likelihood = jit(vmap(lambda w_, x_, interv_targets_: jax_model.log_marginal_likelihood_given_g(
        w=w_, data=x_, interv_targets=interv_targets_), (0, None, None), 0))

    '''Run method'''
    
    if method == 'gt':
        dists = run_marginal_argmax(
            r=r, c=c, key=None, kwargs=kwargs, target=target, additional_descr=additional_descr)

    elif method == 'mcmc_structure':
        key, subk = random.split(key)
        dists = run_marginal_structMCMC(r=r, c=c, key=subk, 
            n_particles_loop=n_particles_loop, kwargs=kwargs, target=target, 
            ig_log_target=ig_log_target, ig_log_target_single=ig_log_target_single, additional_descr=additional_descr)

    elif method == 'boot_ges' or method == 'boot_pc':
        key, subk = random.split(key)
        dists = run_marginal_bootstrap(r=r, c=c, key=subk, 
            n_particles_loop=n_particles_loop, kwargs=kwargs, target=target, 
            ig_log_target=ig_log_target, learner_str=method, additional_descr=additional_descr)

    elif method == 'dibs':
        key, subk = random.split(key)
        dists = run_marginal_svgd(r=r, c=c, key=subk, n_particles=n_particles, 
            target=target, log_prior=log_prior, log_target=log_target, 
            kwargs=kwargs, additional_descr=additional_descr)

    # evaluation
    eval_args = dict(
        rollout=r,
        target=target,
        graph_prior_str=graph_prior_str,
        eltwise_log_marg_likelihood=eltwise_log_marg_likelihood,
    )

    result_df_c = eval(dists, eval_args, joint=False)
    return result_df_c


@ray.remote
def eval_single_joint_target(*, method, c, r, key, n_particles, n_particles_loop, graph_prior_str, kwargs, additional_descr=None, load=True):
    """
    Evaluate a single target based on random key `key` and seed `c`
    for a list of particles `n_particles_loop` using `make_ground_truth_posterior`
    """

    (generative_model_str, generative_model_kwargs), (inference_model_str, inference_model_kwargs) = parse_target_str(kwargs)

    key, subk = random.split(key)
    target = make_ground_truth_posterior(
        key=subk, c=c, n_vars=kwargs.n_vars, 
        graph_prior_str=graph_prior_str,
        generative_model_str=generative_model_str,
        generative_model_kwargs=generative_model_kwargs,
        inference_model_str=inference_model_str,
        inference_model_kwargs=inference_model_kwargs,
        n_observations=kwargs.n_observations,
        n_ho_observations=kwargs.n_ho_observations, 
        n_posterior_g_samples=kwargs.n_posterior_g_samples,
        n_intervention_sets=kwargs.n_intervention_sets,
        perc_intervened=kwargs.perc_intervened,
        load=load, 
        real_data=kwargs.real_data,
        real_data_held_out=kwargs.real_data_held_out,
        real_data_normalize=kwargs.real_data_normalize,
        verbose=False)

    # init inference model using target graph dist
    if inference_model_str == 'lingauss':
        model = LinearGaussianGaussianJAX(**inference_model_kwargs)
    elif inference_model_str == 'fcgauss':
        model = FCGaussianJAX(**inference_model_kwargs)
    else:
        raise NotImplementedError()

    # evaluates [n_particles, n_vars, n_vars], [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
    no_interv_targets = jnp.zeros(kwargs.n_vars).astype(bool)
    eltwise_log_likelihood = jit(vmap(
        lambda w_, theta_, x_, interv_targets_: (
            model.log_likelihood(theta=theta_, w=w_, data=x_, interv_targets=interv_targets_)
        ), (0, 0, None, None), 0))

    @jit
    def ig_log_joint_target(g_mat, theta):
        return (target.g_dist.unnormalized_log_prob_soft(soft_g=g_mat)
                + model.log_prob_parameters(theta=theta, w=g_mat)
                + model.log_likelihood(theta=theta, w=g_mat, data=jnp.array(target.x), interv_targets=no_interv_targets))

    @jit
    def log_prior(single_w_prob):
        ''' p(G) 
            returns shape [1, ]
            will later be vmapped

            single_w_prob : [n_vars, n_vars] in [0, 1]
                encoding probabilities of edges
        '''
        # no batch variable b
        return target.g_dist.unnormalized_log_prob_soft(soft_g=single_w_prob)

    @jit
    def log_joint_target(single_w, single_theta, rng):
        ''' p(theta, D | G) 
            returns shape [1, ]
            will later be vmapped

            single_w :      [n_vars, n_vars]
            single_theta:   [n_vars, n_vars]
        '''
        if kwargs.joint_dibs_n_grad_batch_size is not None:
            # minibatch
            idx = random.choice(rng, a=target.x.shape[0], shape=(min(kwargs.joint_dibs_n_grad_batch_size, target.x.shape[0]),), replace=False)
            x_batch = jnp.array(target.x)[idx, :]
        else:
            x_batch = jnp.array(target.x)

        # compute target
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=x_batch, interv_targets=no_interv_targets)
        return log_prob_theta + log_lik

    @jit
    def log_joint_target_no_batch(single_w, single_theta):
        ''' Same as above but using full data; for metrics'''
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=jnp.array(target.x), interv_targets=no_interv_targets)
        return log_prob_theta + log_lik


    '''Run method'''
    
    if method == 'gt':
        dists = run_joint_argmax(r=r, c=c, key=None, kwargs=kwargs, target=target, additional_descr=additional_descr)

    elif method == 'mh_joint_mcmc_structure' or method == 'gibbs_joint_mcmc_structure':
        key, subk = random.split(key)
        dists = run_joint_structMCMC(r=r, c=c, key=subk, 
            n_particles_loop=n_particles_loop, kwargs=kwargs, target=target, 
            model=model, ig_log_joint_target=ig_log_joint_target,
            option_str=method, additional_descr=additional_descr)
    
    elif method == 'joint_boot_ges' or method == 'joint_boot_pc':
        key, subk = random.split(key)
        dists = run_joint_bootstrap(r=r, c=c, key=subk, 
            n_particles_loop=n_particles_loop, kwargs=kwargs, target=target, 
            model=model, ig_log_joint_target=ig_log_joint_target,
            learner_str=method, additional_descr=additional_descr)

    elif method == 'dibs':
        key, subk = random.split(key)
        dists = run_joint_svgd(r=r, c=c, key=subk, n_particles=n_particles, 
            target=target, model=model, log_prior=log_prior, 
            log_joint_target=log_joint_target, log_joint_target_no_batch=log_joint_target_no_batch,
            kwargs=kwargs, additional_descr=additional_descr)

    # evaluation
    eval_args = dict(
        rollout=r,
        target=target,
        graph_prior_str=graph_prior_str,
        eltwise_log_likelihood=eltwise_log_likelihood,
    )

    result_df_c = eval(dists, eval_args, joint=True)
    return result_df_c


def process_incoming_result(result_df, incoming_dfs, args):
    """
    Appends new pd.DataFrame and saves the result in the existing .csv file  
    """

    # concatenate existing and new results
    result_df = pd.concat([result_df, incoming_dfs], ignore_index=True)

    # save to csv
    save_path = os.path.abspath(os.path.join(
        '..', *STORE_ROOT, args.descr + '.csv'
    ))
    result_df.to_csv(save_path)

    return result_df
