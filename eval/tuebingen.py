import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import ray
import argparse
import pandas as pd 
import csv
import tqdm
import itertools

import matplotlib.pyplot as plt

from sklearn import metrics as sklearn_metrics
from collections import namedtuple

import jax.numpy as jnp
from jax import jit
from jax import random
from jax.tree_util import tree_flatten

from dibs.utils.graph import *

from dibs.models.linearGaussianGaussian import LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGeJAX
from dibs.models.FCGaussian import FCGaussianJAX

from dibs.svgd.bivariate_joint_dot_graph_svgd import BivariateJointDotProductGraphSVGD
from dibs.svgd.joint_dot_graph_svgd import JointDotProductGraphSVGD

from dibs.kernel.joint import JointAdditiveFrobeniusSEKernel

from dibs.utils.version_control import get_version_datetime

from dibs.eval.target import make_ground_truth_posterior, load_tuebingen_metadata
from dibs.eval.inference import process_incoming_result

from config.svgd import joint_config


import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype")

STORE_ROOT = ['results']

fields_TuebingenEvaluation = (
    'key',
    'tuebingen_id',
    'tuebingen_id_numeric',
    'tuebingen_id_weight',
    'c',
    'correct',
    'p_A_to_B',
    'gt_is_A_to_B',
    'n_empty',
    'n_A_to_B',
    'n_B_to_A',
    'n_cyclic',
)


TuebingenEvaluation = namedtuple(
    'TuebingenEvaluation',
    fields_TuebingenEvaluation,
    defaults=[None for _ in fields_TuebingenEvaluation]
)


def make_parser():
    """
    Returns argparse parser to control evaluation from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--bivariate", action="store_true")
    parser.add_argument("--descr", required=True, help="set experiment filename; keep the same to resume in case interrupted")
    parser.add_argument("--load", help="loads result df and just computes metrics")
    parser.add_argument("--n_rollouts", type=int, default=30)

    parser.add_argument("--n_grad_batch_size", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=3000)
    parser.add_argument("--model", default="fcgauss")
    parser.add_argument("--n_particles", type=int, default=30)
    parser.add_argument("--n_latent_dim", type=int, default=2)
    parser.add_argument("--obs_noise", type=float, default=0.1)

    return parser


def count_occurrence(g_array, g_single):
    """
    Counts how many times `g_single` of shape [a, b] 
    occurs inside `g_array` of shape [N, a, b] 
    """
    return jnp.all(jnp.isclose(g_array, g_single[None]), axis=(-1, -2)).sum().item()
   


def compute_metrics_df(result_df, kwargs):
    """
    Compute results aggregated over random restarts
    """

    roc_aucs = []
    roc_aucs_weighted = []
    accuracies = []
    accuracies_weighted = []
    prop_cyclic = []
    prop_empty = []

    for c in range(kwargs.n_rollouts):
         
        # select rollout
        df = result_df[result_df['c'] == c]
        if df.empty:
            continue

        # metrics
        fpr, tpr, _ = sklearn_metrics.roc_curve(df['gt_is_A_to_B'].astype(int).tolist(), df['p_A_to_B'].tolist())
        roc_auc_ = sklearn_metrics.auc(fpr, tpr)

        fpr_weighted, tpr_weighted, _ = sklearn_metrics.roc_curve(df['gt_is_A_to_B'].astype(int).tolist(), df['p_A_to_B'].tolist(), sample_weight=df['tuebingen_id_weight'].tolist())
        roc_auc_weighted_ = sklearn_metrics.auc(fpr_weighted, tpr_weighted)

        accuracy_ = df['correct'].values.sum() / len(df['correct'].values)
        accuracy_weighted_ = df['correct'].values.dot(df['tuebingen_id_weight'].values) / df['tuebingen_id_weight'].values.sum()

        # save
        accuracies.append(accuracy_)
        accuracies_weighted.append(accuracy_weighted_)
        roc_aucs.append(roc_auc_)
        roc_aucs_weighted.append(roc_auc_weighted_)
        prop_cyclic.append(df['n_cyclic'].values.mean() / kwargs.n_particles)
        prop_empty.append(df['n_empty'].values.mean() / kwargs.n_particles)

    # aggregate
    roc_aucs = jnp.array(roc_aucs)
    roc_aucs_weighted = jnp.array(roc_aucs_weighted)
    accuracies = jnp.array(accuracies)
    accuracies_weighted = jnp.array(accuracies_weighted)
    prop_cyclic = jnp.array(prop_cyclic)
    prop_empty = jnp.array(prop_empty)

    ops = [
        ('mean',    jnp.mean), 
        ('std',     jnp.std),
        ('median',  jnp.median),
        ('perc_10', lambda arr: jnp.percentile(arr, 10)),
        ('perc_25', lambda arr: jnp.percentile(arr, 25)),
        ('perc_75', lambda arr: jnp.percentile(arr, 75)),
        ('perc_90', lambda arr: jnp.percentile(arr, 90)),
    ]

    metrics_df = pd.DataFrame.from_dict({
        'roc_auc':           [op[1](roc_aucs).item() for op in ops], 
        'roc_auc_weighted':  [op[1](roc_aucs_weighted).item() for op in ops],
        'accuracy':          [op[1](accuracies).item() for op in ops],
        'accuracy_weighted': [op[1](accuracies_weighted).item() for op in ops],
        'prop_cylic':        [op[1](prop_cyclic).item() for op in ops],
        'prop_empty':        [op[1](prop_empty).item() for op in ops],
    }, orient='index', columns=[op[0] for op in ops])

    return metrics_df, dict(
        roc_auc=roc_aucs,
        roc_auc_weighted=roc_aucs_weighted,
        accuracy=accuracies,
        accuracy_weighted=accuracies_weighted,
        prop_cylic=prop_cyclic,
        prop_empty=prop_empty,
    )




@ray.remote
def eval_single_tuebingen_pair(*, tuebingen_id, tuebingen_id_weight, c, key, kwargs):
    """
    Runs one random restarts of DiBS with on Tuebingen cause-effect pair `tuebingen_id`
    """

    df = pd.DataFrame(columns=TuebingenEvaluation._fields)

    '''Target'''
    if kwargs.model == 'lingauss':
        # Linear Gaussian
        generative_model_str = 'lingauss'
        generative_model_kwargs = dict(
            obs_noise=kwargs.obs_noise,
            mean_edge=0.0,
            sig_edge=1.0,
        )
        inference_model_str = generative_model_str
        inference_model_kwargs = generative_model_kwargs
        singular_dim_theta = 2

    else:
        # Nonlinear Gaussian with fully connected neural net
        generative_model_str = 'fcgauss'
        generative_model_kwargs = dict(
            obs_noise=kwargs.obs_noise,
            sig_param=1.0,
            dims=[5],
        )
        inference_model_str = generative_model_str
        inference_model_kwargs = generative_model_kwargs
        singular_dim_theta = 3

    # inference model
    if inference_model_str == 'bge':
        model = BGeJAX(**inference_model_kwargs)

    elif inference_model_str == 'lingauss':
        model = LinearGaussianGaussianJAX(**inference_model_kwargs)

    elif inference_model_str == 'fcgauss':
        model = FCGaussianJAX(**inference_model_kwargs)

    else:
        raise NotImplementedError()

    # run
    key, subk = random.split(key)
    target = make_ground_truth_posterior(
        key=subk, c=c, n_vars=2,
        graph_prior_str=None,
        generative_model_str=generative_model_str,
        generative_model_kwargs=generative_model_kwargs,
        inference_model_str=inference_model_str,
        inference_model_kwargs=inference_model_kwargs,
        n_observations=None,
        n_ho_observations=None,
        n_posterior_g_samples=None,
        n_intervention_sets=None,
        perc_intervened=None,
        real_data=f'tuebingen-{tuebingen_id}',
        load=False)

    x = jnp.array(target.x)
    
    @jit
    def log_prior(_):
        ''' Uniform prior in this setting'''
        return 0.0

    no_interv_targets = jnp.zeros(2).astype(bool)

    def log_joint_target(single_w, single_theta, rng):
        ''' p(theta, D | G) 
            returns shape [1, ]
            will later be vmapped

            single_w :      [2, 2]
            single_theta:   
            b:              int indicating batch
            rng:            [1,]
        '''

        # minibatch
        idx = random.choice(rng, a=x.shape[0], shape=(min(kwargs.n_grad_batch_size, x.shape[0],),), replace=False)
        x_batch = x[idx, :]

        # compute target
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=x_batch, interv_targets=no_interv_targets)
        return log_prob_theta + log_lik


    def log_joint_target_no_batch(single_w, single_theta):
        ''' Same as above but using full data; for metrics computed on the flu
        '''
        log_prob_theta = model.log_prob_parameters(theta=single_theta, w=single_w)
        log_lik = model.log_likelihood(theta=single_theta, w=single_w, data=x, interv_targets=no_interv_targets)
        return log_prob_theta + log_lik


    '''SVGD'''
    tuned = joint_config[inference_model_str][20]
    optimizer = dict(name='rmsprop', stepsize=tuned.get("opt_stepsize", 0.005))

    kernel = JointAdditiveFrobeniusSEKernel(
        h_latent=tuned['h_latent'],
        h_theta=tuned['h_theta'],
        scale_latent=tuned.get('kernel_scale_latent', 1.0),
        scale_theta=tuned.get('kernel_scale_theta', 1.0),
        soft_graph_mask=False,
        singular_dim_theta=singular_dim_theta,
        graph_embedding_representation=True)

    # temperature hyperparameters
    def alpha_sched(t):
        return tuned['alpha_linear'] * t

    def beta_sched(t):
        return tuned['beta_linear'] * t

    def tau_sched(t):
        return jnp.array([1.0])
    
    # svgd
    if kwargs.bivariate:
        key, subk = random.split(key)
        latent_prior_std = 1.0 / jnp.sqrt(kwargs.n_latent_dim)
        init_particles_x = random.normal(subk, shape=(kwargs.n_particles, kwargs.n_latent_dim, 2)) * latent_prior_std

        key, subk = random.split(key)
        init_particles_theta = model.init_parameters(key=subk, n_particles=kwargs.n_particles, n_vars=2)

        svgd = BivariateJointDotProductGraphSVGD(
            n_dim=kwargs.n_latent_dim,
            optimizer=optimizer,
            kernel=kernel, 
            target_log_prior=log_prior,
            target_log_joint_prob=log_joint_target,
            target_log_joint_prob_no_batch=log_joint_target_no_batch,
            alpha=alpha_sched,
            tau=tau_sched,
            n_grad_mc_samples=128,
            n_grad_batch_size=kwargs.n_grad_batch_size,
            n_acyclicity_mc_samples=32,
            latent_prior_std=latent_prior_std,
            verbose=False)
    else:
        key, subk = random.split(key)
        latent_prior_std = 1.0 / jnp.sqrt(kwargs.n_latent_dim)
        init_particles_x = random.normal(subk, shape=(kwargs.n_particles, 2, kwargs.n_latent_dim, 2)) * latent_prior_std

        key, subk = random.split(key)
        init_particles_theta = model.init_parameters(key=subk, n_particles=kwargs.n_particles, n_vars=2)

        svgd = JointDotProductGraphSVGD(
            n_vars=2,
            n_dim=kwargs.n_latent_dim,
            optimizer=optimizer,
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_joint_prob=log_joint_target,
            target_log_joint_prob_no_batch=log_joint_target_no_batch,
            alpha=alpha_sched,
            beta=beta_sched,
            gamma=lambda _: jnp.array([0.0]),
            tau=tau_sched,
            n_grad_mc_samples=128,
            n_grad_batch_size=kwargs.n_grad_batch_size,
            n_acyclicity_mc_samples=32,
            latent_prior_std=latent_prior_std,
            verbose=False)


    # evaluate
    key, subk = random.split(key)
    particles_x, _ = svgd.sample_particles(
        key=subk,
        n_steps=kwargs.n_steps, 
        init_particles_x=init_particles_x, 
        init_particles_theta=init_particles_theta, 
        metric_every=50,
        eval_metrics=[])

    # true graph; 
    # if A --> B: `gt_A_to_B` is True
    # if B --> A: `gt_A_to_B` is False
    g_gt = target.g  
    gt_is_A_to_B = jnp.allclose(g_gt,  jnp.array([[0, 1], [0, 0]]))

    # pred
    g_samples = svgd.particle_to_hard_g(particles_x)

    n_empty   = count_occurrence(g_samples, jnp.array([[0, 0], [0, 0]]))
    n_A_to_B =  count_occurrence(g_samples, jnp.array([[0, 1], [0, 0]]))
    n_B_to_A =  count_occurrence(g_samples, jnp.array([[0, 0], [1, 0]]))
    n_cyclic  = count_occurrence(g_samples, jnp.array([[0, 1], [1, 0]]))

    if n_A_to_B + n_B_to_A == 0:
        # no prediction; only empty graphs
        p_A_to_B = 0.5
        correct = False
    else:
        # confidence in A --> B
        p_A_to_B = n_A_to_B / (n_A_to_B + n_B_to_A)

        # hard prediction correct if confidence in correct direction > 0.5
        if gt_is_A_to_B:
            correct = (p_A_to_B > 0.5)
        else:
            correct = (p_A_to_B < 0.5)


    # collect info
    res_kwars = dict(
        key=key,
        tuebingen_id=tuebingen_id,
        tuebingen_id_numeric=int(tuebingen_id),
        tuebingen_id_weight=tuebingen_id_weight,
        c=c,
        correct=int(correct),
        p_A_to_B=p_A_to_B,
        gt_is_A_to_B=int(gt_is_A_to_B),
        n_empty=n_empty,
        n_A_to_B=n_A_to_B,
        n_B_to_A=n_B_to_A,
        n_cyclic=n_cyclic,
    )

    # generate result and append to dataframe
    res = TuebingenEvaluation(**res_kwars)
    df = df.append(res._asdict(), ignore_index=True)
    return df



if __name__ == '__main__':

    jnp.set_printoptions(precision=4, suppress=True)

    parser = make_parser()
    kwargs = parser.parse_args()
    kwargs.descr = kwargs.descr + get_version_datetime()
    if kwargs.smoke_test:
        kwargs.n_rollouts = 2
        kwargs.n_steps = 2
        kwargs.n_particles = 10

    # load?
    if kwargs.load:
        print('Loaded')
        result_path = kwargs.load
        result_df = pd.read_csv(result_path, index_col=0)
        print(result_df)

        # save to csv
        metrics_df, data = compute_metrics_df(result_df, kwargs)
        save_path = os.path.abspath(os.path.join(
            '..', 'results', kwargs.descr + '_metrics_loaded.csv'
        ))
        metrics_df.to_csv(save_path)
        print(metrics_df)

        # plot
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        fig.subplots_adjust(top=0.8)
        axs[0].hist(data['roc_auc_weighted'])
        axs[0].set_title('roc_auc_weighted')
        axs[0].set_xlim((0, 1))
        axs[1].hist(data['accuracy_weighted'])
        axs[1].set_title('accuracy_weighted')
        axs[1].set_xlim((0.25, 0.75))

        fig.suptitle(kwargs.descr, y=0.95)

        save_path_plot = os.path.abspath(os.path.join(
            '..', 'results', kwargs.descr + '_metrics_loaded.png'
        ))
        plt.savefig(save_path_plot, format='png', facecolor=None,
            dpi=300, bbox_inches='tight')

        exit(0)

    # run experiments
    key = random.PRNGKey(0)
    ray.init()
    # ray.init(local_mode=True)

    """Eval all rollouts and all cause-effect pairs in parallel"""
    result_df = pd.DataFrame()  
    result_ids = []

    tuebingen_bivariate_pairs = load_tuebingen_metadata()
    n_tuebingen_variants = len(tuebingen_bivariate_pairs.keys())

    # setup each instance
    subkeys = random.split(key, n_tuebingen_variants * kwargs.n_rollouts)
    for j, (tuebingen_id, pair_info) in enumerate(tuebingen_bivariate_pairs.items()):
        for c in range(kwargs.n_rollouts):
            idx = j * kwargs.n_rollouts + c
            result_ids.append(eval_single_tuebingen_pair.remote(
                tuebingen_id=tuebingen_id, tuebingen_id_weight=pair_info['weight'], c=c, key=subkeys[idx], kwargs=kwargs))

        if kwargs.smoke_test and j == 3:
            break
    
    # run all in parallel
    pbar = tqdm.tqdm(total=len(result_ids))
    while len(result_ids):

        # returns one ready ID at a time, and remaining open tasks
        done_id, result_ids = ray.wait(result_ids)
        result_df = process_incoming_result(result_df, ray.get(done_id[0]), kwargs)
        pbar.update(1)

        # also save intermediate metrics for early indication
        if not len(result_ids) % 50:
            metrics_df, _ = compute_metrics_df(result_df, kwargs)
            save_path = os.path.abspath(os.path.join(
                '..', 'results', kwargs.descr + '_metrics.csv'
            ))
            metrics_df.to_csv(save_path)

    pbar.close()
    ray.shutdown()

    print(result_df)

    # save to csv
    metrics_df, _ = compute_metrics_df(result_df, kwargs)
    save_path = os.path.abspath(os.path.join(
        '..', 'results', kwargs.descr + '_metrics.csv'
    ))
    metrics_df.to_csv(save_path)

    print(metrics_df)






    








