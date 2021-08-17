import argparse
import ray
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import collections
import pprint 
import tqdm
import multiprocessing

import jax.numpy as jnp
from jax import random
from jax import jit, vmap

import pandas as pd

from dibs.utils.graph import mat_to_graph
from dibs.utils.version_control import get_version_datetime
from dibs.eval.target import make_ground_truth_posterior
from dibs.mcmc.joint_structure import MHJointStructureMCMC, GibbsJointStructureMCMC
from dibs.models.linearGaussianGaussian import LinearGaussianGaussianJAX
from dibs.models.FCGaussian import FCGaussianJAX
from dibs.utils.tree import tree_unzip_leading, tree_zip_leading, tree_shapes

STORE_ROOT = ['tuning']

if __name__ == '__main__':

    key = random.PRNGKey(0)
    jnp.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", default="tune_joint_mcmc") 
    parser.add_argument("--smoke_test", action="store_true") 
    parser.add_argument("--verbose", action="store_true") 
    parser.add_argument("--joint_inference_model", default="lingauss", choices=["lingauss", "fcgauss"], help="joint inference model")
    parser.add_argument("--n_rollouts", default=5, type=int) 
    parser.add_argument("--n_vars", default=20, type=int) 
    parser.add_argument("--n_observations", default=100, type=int) 
    parser.add_argument("--burnin", default=5e3, type=int)
    parser.add_argument("--ave", default=5e3, type=int)
    parser.add_argument("--rel_cpu_usage", default=1.0, type=float)

    kwargs = parser.parse_args()
    kwargs.descr = kwargs.descr + '_' + kwargs.joint_inference_model + get_version_datetime()
    

    '''This script evaluates joint MCMC methods at different theta proposal states 
    to select theta with ave acceptance rate after burnin of approximately 0.2-0.3
    '''
    # resources
    cpus_avail = multiprocessing.cpu_count()
    cpus_used = min(cpus_avail, int(cpus_avail * kwargs.rel_cpu_usage))
    print(f'CPU available: {cpus_avail}  used: {cpus_used}')

    ray.init(num_cpus=cpus_used)
    # ray.init(num_cpus=cpus_used, local_mode=True)

    scales = [
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
    ]

    # MCMC settings
    if kwargs.smoke_test:
        kwargs.burnin = 2
        kwargs.ave = 2
        kwargs.n_rollouts = 2
        scales = scales[:2]

    if kwargs.joint_inference_model == 'lingauss':
        model_kwargs = dict(
            obs_noise=0.1, 
            mean_edge=0.0, 
            sig_edge=1.0,
        )
    elif kwargs.joint_inference_model == 'fcgauss':
        model_kwargs = dict(
            obs_noise=0.1,
            sig_param=1.0,
            dims=[10, 10],
        )
    else:
        raise NotImplementedError()

   
    '''Run MCMC'''

    @ray.remote
    def run_mcmc_settings(*, c, local_key, theta_prop_sig):

        # target
        local_key, subk = random.split(local_key)
        target = make_ground_truth_posterior(
            key=subk, c=c, n_vars=kwargs.n_vars,
            graph_prior_str='er',
            generative_model_str=kwargs.joint_inference_model,
            generative_model_kwargs=model_kwargs,
            inference_model_str=kwargs.joint_inference_model,
            inference_model_kwargs=model_kwargs,
            n_observations=kwargs.n_observations,
            n_ho_observations=100,
            n_posterior_g_samples=1,
            n_intervention_sets=10,
            perc_intervened=0.1,
            load=False,
            verbose=False)
        
        # inference model
        if kwargs.joint_inference_model == 'lingauss':
            model = LinearGaussianGaussianJAX(**model_kwargs)
        elif kwargs.joint_inference_model == 'fcgauss':
            model = FCGaussianJAX(**model_kwargs)
        else:
            raise NotImplementedError()

        no_interv_targets = jnp.zeros(kwargs.n_vars).astype(bool)

        def log_joint_target(g_mat, theta):
            return (target.g_dist.unnormalized_log_prob(g=mat_to_graph(g_mat))
                    + model.log_prob_parameters(theta=theta, w=g_mat)
                    + model.log_likelihood(theta=theta, w=g_mat, data=jnp.array(target.x), interv_targets=no_interv_targets))

        
        # evaluates [n_particles, n_vars, n_vars], [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
        eltwise_log_likelihood = jit(vmap(
            lambda w_, theta_, x_: (
                model.log_likelihood(theta=theta_, w=w_, data=x_, interv_targets=no_interv_targets)
            ), (0, 0, None), 0))

        # [L, d, d], [L, d, d] -> [L]
        def double_eltwise_log_joint_prob(g_mats, thetas):
            thetas_unzipped = tree_unzip_leading(thetas, g_mats.shape[0])
            return jnp.array([log_joint_target(g_mat, theta) for g_mat, theta in zip(g_mats, thetas_unzipped)])

        # MH
        mh_mcmc = MHJointStructureMCMC(
            n_vars=kwargs.n_vars,
            theta_prop_sig=theta_prop_sig,
            verbose=kwargs.verbose,
            only_non_covered=False)

        local_key, subk = random.split(local_key)
        _, _ = mh_mcmc.sample(key=subk, n_samples=1,
            log_joint_target=log_joint_target,
            theta_shape=model.get_theta_shape(n_vars=kwargs.n_vars),
            burnin=kwargs.burnin, thinning=kwargs.ave,
            return_matrices=False)
        
        # MH-within-Gibbs
        gibbs_mcmc = GibbsJointStructureMCMC(
            n_vars=kwargs.n_vars,
            theta_prop_sig=theta_prop_sig,
            verbose=kwargs.verbose,
            only_non_covered=False)

        local_key, subk = random.split(local_key)
        _, _ = gibbs_mcmc.sample(key=subk, n_samples=1,
            log_joint_target=log_joint_target,
            theta_shape=model.get_theta_shape(n_vars=kwargs.n_vars),
            burnin=kwargs.burnin, thinning=kwargs.ave,
            return_matrices=False)

        return (theta_prop_sig,
                mh_mcmc.ave_alpha_after_burnin, 
                gibbs_mcmc.ave_alpha_after_burnin)

    # run in parallel
    result_ids_prelim = []
    key, *subk = random.split(key, kwargs.n_rollouts + 1)

    mh_rates = collections.defaultdict(list)
    gibbs_rates = collections.defaultdict(list)

    for c in range(kwargs.n_rollouts):
        for scale in scales:
            result_ids_prelim.append(
                run_mcmc_settings.remote(
                    c=c,
                    local_key=subk[c],
                    theta_prop_sig=scale)
            )

    pbar = tqdm.tqdm(total=len(result_ids_prelim))
    while len(result_ids_prelim):
        done_id, result_ids_prelim = ray.wait(result_ids_prelim)
        scale_, mh_rate_, gibbs_rate_ = ray.get(done_id[0])
        mh_rates[scale_].append(mh_rate_)
        gibbs_rates[scale_].append(gibbs_rate_)
        pbar.update(1)

        # process incoming results after every finished run
        mh_rates_ave,    mh_n =    zip(*[(jnp.array(mh_rates[scale_]).mean(),    len(mh_rates[scale_]))    if mh_rates[scale_]    else (jnp.nan, 0) for scale_ in scales])
        gibbs_rates_ave, gibbs_n = zip(*[(jnp.array(gibbs_rates[scale_]).mean(), len(gibbs_rates[scale_])) if gibbs_rates[scale_] else (jnp.nan, 0) for scale_ in scales])

        # to csv
        result = jnp.array([scales, mh_rates_ave, mh_n, gibbs_rates_ave, gibbs_n]).T
        result_df = pd.DataFrame(result, columns=[
            'theta_prop_sig',
            'MHJointStructureMCMC.ave_alpha_after_burnin', 
            'n_MHJointStructureMCMC', 
            'GibbsJointStructureMCMC.ave_alpha_after_burnin',
            'n_GibbsJointStructureMCMC',
            ])
        
        # save to csv
        save_path = os.path.abspath(os.path.join(
            '..', *STORE_ROOT, kwargs.descr + '.csv'
        ))
        result_df.to_csv(save_path)

    print('Saved at: ', save_path)
    ray.shutdown()
