import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")

import tqdm
import multiprocessing
import itertools
import ray
import copy

import pandas as pd

import jax.numpy as jnp
from jax import random

from dibs.utils.version_control import get_version_datetime

from dibs.eval.inference import eval_single_target, eval_single_joint_target, process_incoming_result
from dibs.eval.parser import make_evaluation_parser

from config.svgd import marginal_config, joint_config


if __name__ == '__main__':

    parser = make_evaluation_parser()
    kwargs = parser.parse_args()

    jnp.set_printoptions(precision=4, suppress=True)
    key = random.PRNGKey(kwargs.seed)

    # resources
    cpus_avail = multiprocessing.cpu_count()
    cpus_used = min(cpus_avail, int(cpus_avail * kwargs.rel_cpu_usage))
    print(f'CPU available: {cpus_avail}  used: {cpus_used}')

    ray.init(num_cpus=cpus_used)
    # ray.init(num_cpus=cpus_used, local_mode=True)

    # add current timestamp and version
    kwargs.descr = kwargs.descr + get_version_datetime()
    '''
    This script evaluates several methods against the 
    ground truth posterior p(G|D) (for up to 5 nodes)
    '''
    n_particles_loop = jnp.array(kwargs.n_particles_loop)

    # warn if no tuned hyperparameters; extract memory requirements
    if kwargs.joint:
        assert(kwargs.joint_inference_model in joint_config)
        if kwargs.n_vars not in joint_config[kwargs.joint_inference_model].keys():
            print(f'No DiBS config available for {kwargs.n_vars} nodes.')
            print(f'Will use config for {min(joint_config[kwargs.joint_inference_model].keys())} nodes.')
        config_ = joint_config[kwargs.joint_inference_model]
    
    else:
        assert(kwargs.inference_model in marginal_config)
        if kwargs.n_vars not in marginal_config[kwargs.inference_model].keys():
            print(f'No DiBS config available for {kwargs.n_vars} nodes.')
            print(f'Will use config for {min(marginal_config[kwargs.inference_model].keys())} nodes.')
        config_ = marginal_config[kwargs.inference_model]
    
    kwargs_svgd = config_.get(kwargs.n_vars, config_[min(config_.keys())])

    # smoke test
    if kwargs.smoke_test:
        n_particles_loop = jnp.array([2])
        kwargs.n_rollouts = 1
        kwargs.n_variants = 1

    '''Run all method executions and random rollouts in parallel'''
    n_rollouts = kwargs.n_rollouts 
    n_variants = kwargs.n_variants 
    subkeys = random.split(key, n_variants * n_rollouts)

    result_df = pd.DataFrame()  

    #
    # p(G, theta | D) inference 
    #
    if kwargs.joint:

        # first eval ground G or (G, theta)
        # and creates targets
        result_ids_prelim = []
        for graph_prior_str, c in itertools.product(kwargs.graph_prior, range(n_variants)):
            result_ids_prelim.append(
                eval_single_joint_target.remote(
                    method='gt', c=c, r=0, key=subkeys[c], n_particles=None,
                    n_particles_loop=None, graph_prior_str=graph_prior_str, kwargs=kwargs, load=False)
            )
        
        while len(result_ids_prelim):
            done_id, result_ids_prelim = ray.wait(result_ids_prelim)
            if not kwargs.create_targets_only:
                result_df = process_incoming_result(result_df, ray.get(done_id[0]), kwargs)
            
        if kwargs.create_targets_only:
            exit()
        
        # eval all method rollouts over all variants in parallel
        result_ids = []
        for graph_prior_str, c, r in itertools.product(kwargs.graph_prior, range(n_variants), range(n_rollouts)):
            idx = c * n_rollouts + r

            # dibs
            if not kwargs.skip_joint_dibs:
                for n in n_particles_loop:

                    # default
                    runs = [(None, kwargs)]

                    # ablation studies
                    if kwargs.grid_joint_dibs_graph_embedding_representation:

                        kwargs_mod_on = copy.deepcopy(kwargs)
                        kwargs_mod_on.joint_dibs_graph_embedding_representation = True

                        kwargs_mod_off = copy.deepcopy(kwargs)
                        kwargs_mod_off.joint_dibs_graph_embedding_representation = False

                        runs = [
                            ('_embed_on', kwargs_mod_on),
                            ('_embed_off', kwargs_mod_off),
                        ]
                    
                    if kwargs.grid_joint_dibs_latent_dim:

                        runs = []
                        for n_dim_ in {20: [5, 7, 10, 15, 20], 50: [10, 15, 20, 30, 50]}.get(kwargs.n_vars, 20):
                            kwargs_mod = copy.deepcopy(kwargs)
                            kwargs_mod.joint_dibs_latent_dim = n_dim_
                            runs.append((f'_ndim={n_dim_}', kwargs_mod))
                    
                    if kwargs.grid_joint_dibs_steps:

                        runs = []
                        for n_steps in [200, 400, 800, 1600, 3200]:
                            kwargs_mod = copy.deepcopy(kwargs)
                            kwargs_mod.joint_dibs_n_steps = n_steps
                            runs.append((f'_steps={n_steps}', kwargs_mod))

                    # run
                    for descr_run, kwargs_run in runs:
                        remote_kwargs = dict(
                            method='dibs', additional_descr=descr_run, c=c, r=r, key=subkeys[idx],
                            n_particles=n, n_particles_loop=None, graph_prior_str=graph_prior_str,
                            kwargs=kwargs_run,
                        )
                        # memory-aware scheduling
                        result_ids.append(
                            eval_single_joint_target.options(**kwargs_svgd["resources"]).remote(**remote_kwargs)
                            if kwargs.resource_aware else
                            eval_single_joint_target.remote(**remote_kwargs)
                        )

            # Metropolis-Hastings joint structure MCMC
            if not kwargs.skip_mh_joint_mcmc_structure and not kwargs.skip_baselines:
                result_ids.append(
                    eval_single_joint_target.remote(
                        method='mh_joint_mcmc_structure', c=c, r=r, key=subkeys[idx],
                        n_particles=None, n_particles_loop=n_particles_loop,
                        graph_prior_str=graph_prior_str, kwargs=kwargs)
                )

            # Metropolis-Hastings joint structure MCMC
            if not kwargs.skip_gibbs_joint_mcmc_structure and not kwargs.skip_baselines:
                result_ids.append(
                    eval_single_joint_target.remote(
                        method='gibbs_joint_mcmc_structure', c=c, r=r, key=subkeys[idx],
                        n_particles=None, n_particles_loop=n_particles_loop,
                        graph_prior_str=graph_prior_str, kwargs=kwargs)
                )

            # bootstrap
            if kwargs.joint_inference_model == 'lingauss':
                if not kwargs.skip_joint_bootstrap_ges and not kwargs.skip_baselines:
                    result_ids.append(
                        eval_single_joint_target.remote(
                            method='joint_boot_ges', c=c, r=r, key=subkeys[idx],
                            n_particles=None, n_particles_loop=n_particles_loop,
                            graph_prior_str=graph_prior_str, kwargs=kwargs)
                    )
                if not kwargs.skip_joint_bootstrap_pc and not kwargs.skip_baselines:
                    result_ids.append(
                        eval_single_joint_target.remote(
                            method='joint_boot_pc', c=c, r=r, key=subkeys[idx],
                            n_particles=None, n_particles_loop=n_particles_loop,
                            graph_prior_str=graph_prior_str, kwargs=kwargs)
                    )
                            
    else:
        
        # first eval ground G or (G, theta)
        # and creates targets
        result_ids_prelim = []
        for graph_prior_str, c in itertools.product(kwargs.graph_prior, range(n_variants)):
            result_ids_prelim.append(
                eval_single_target.remote(
                    method='gt', c=c, r=0, key=subkeys[c], n_particles=None,
                    n_particles_loop=None, graph_prior_str=graph_prior_str, kwargs=kwargs, load=False)
            )
        
        while len(result_ids_prelim):
            done_id, result_ids_prelim = ray.wait(result_ids_prelim)
            if not kwargs.create_targets_only:
                result_df = process_incoming_result(result_df, ray.get(done_id[0]), kwargs)
            
        if kwargs.create_targets_only:
            exit()

        # eval all method rollouts over all variants in parallel
        result_ids = []
        for graph_prior_str, c, r in itertools.product(kwargs.graph_prior, range(n_variants), range(n_rollouts)):
            idx = c * n_rollouts + r

            # dibs
            if not kwargs.skip_dibs:
                for n in n_particles_loop:

                    # default
                    runs = [(None, kwargs)]

                    # ablation studies
                    if kwargs.grid_dibs_graph_embedding_representation:
                        
                        kwargs_mod_on = copy.deepcopy(kwargs)
                        kwargs_mod_on.dibs_graph_embedding_representation = True

                        kwargs_mod_off = copy.deepcopy(kwargs)
                        kwargs_mod_off.dibs_graph_embedding_representation = False

                        runs = [
                            ('_embed_on', kwargs_mod_on), 
                            ('_embed_off', kwargs_mod_off),
                        ]
                    
                    if kwargs.grid_dibs_latent_dim:

                        runs = []
                        for n_dim_ in {20: [5, 7, 10, 15, 20], 50: [10, 15, 20, 30, 50]}.get(kwargs.n_vars, 20):
                            kwargs_mod = copy.deepcopy(kwargs)
                            kwargs_mod.dibs_latent_dim = n_dim_
                            runs.append((f'_ndim={n_dim_}', kwargs_mod))

                    if kwargs.grid_dibs_steps:

                        runs = []
                        for n_steps in [200, 400, 800, 1600, 3200]:
                            kwargs_mod = copy.deepcopy(kwargs)
                            kwargs_mod.dibs_n_steps = n_steps
                            runs.append((f'_steps={n_steps}', kwargs_mod))

                    # run
                    for descr_run, kwargs_run in runs:
                        remote_kwargs = dict(
                            method='dibs', additional_descr=descr_run, c=c, r=r, key=subkeys[idx],
                            n_particles=n, n_particles_loop=None, graph_prior_str=graph_prior_str,
                            kwargs=kwargs_run,
                        )
                        # memory-aware scheduling
                        result_ids.append(
                            eval_single_target.options(**kwargs_svgd["resources"]).remote(**remote_kwargs)
                            if kwargs.resource_aware else
                            eval_single_target.remote(**remote_kwargs)
                        )

            # MCMC
            if not kwargs.skip_mcmc_structure and not kwargs.skip_baselines:
                result_ids.append(
                    eval_single_target.remote(
                        method='mcmc_structure', c=c, r=r, key=subkeys[idx], 
                        n_particles=None, n_particles_loop=n_particles_loop, 
                        graph_prior_str=graph_prior_str, kwargs=kwargs)
                )

            # bootstrap
            if not kwargs.skip_bootstrap_ges and not kwargs.skip_baselines:
                result_ids.append(
                    eval_single_target.remote(
                        method='boot_ges', c=c, r=r, key=subkeys[idx],
                        n_particles=None, n_particles_loop=n_particles_loop,
                        graph_prior_str=graph_prior_str, kwargs=kwargs)
                )
            if not kwargs.skip_bootstrap_pc and not kwargs.skip_baselines:
                result_ids.append(
                    eval_single_target.remote(
                        method='boot_pc', c=c, r=r, key=subkeys[idx],
                        n_particles=None, n_particles_loop=n_particles_loop,
                        graph_prior_str=graph_prior_str, kwargs=kwargs)
                )


    # run all in parallel
    pbar = tqdm.tqdm(total=len(result_ids))
    while len(result_ids):

        # returns one ready ID at a time, and remaining open tasks
        if kwargs.timeout is None:
            done_id, result_ids = ray.wait(result_ids)
        else:
            done_id, result_ids = ray.wait(result_ids, timeout=kwargs.timeout)
        if done_id:
            result_df = process_incoming_result(result_df, ray.get(done_id[0]), kwargs)
            pbar.update(1)

        else:
            print(f'The timeout limit of {kwargs.timeout} secs was reached. All remaining experiments were cancelled.')
            break

    # print(result_df)
    
    pbar.close()
    ray.shutdown()
    
