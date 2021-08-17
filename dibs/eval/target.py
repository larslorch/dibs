import os
import pickle
from collections import namedtuple

import jax.numpy as jnp
from jax import random

from dibs.graph.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from dibs.utils.graph import graph_to_mat, adjmat_to_str

from dibs.models.linearGaussian import LinearGaussian, LinearGaussianJAX
from dibs.models.linearGaussianEquivalent import BGe, BGeJAX
from dibs.models.nonlinearGaussian import DenseNonlinearGaussianJAX

STORE_ROOT = ['store'] 


Target = namedtuple('Target', (
    'passed_key',               # jax.random key passed _into_ the function generating this object
    'graph_model',
    'generative_model',
    'inference_model',
    'n_vars',
    'n_observations',
    'n_ho_observations',
    'g',                        # [n_vars, n_vars]
    'theta',                    # PyTree
    'x',                        # [n_observation, n_vars]    data
    'x_ho',                     # [n_ho_observation, n_vars] held-out data
    'x_interv',                 # list of (interv dict, held-out interventional data) 
))


def save_pickle(obj, relpath):
    """Saves `obj` to `path` using pickle"""
    save_path = os.path.abspath(os.path.join(
        '..', *STORE_ROOT, relpath + '.pk'
    ))
    with open(save_path, 'wb') as fp:
        pickle.dump(obj, fp)

def load_pickle(relpath):
    """Loads object from `path` using pickle"""
    load_path = os.path.abspath(os.path.join(
        '..', *STORE_ROOT, relpath + '.pk'
    ))
    with open(load_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def options_to_str(**options):
    return '-'.join(['{}={}'.format(k, v) for k, v in options.items()])


def hparam_dict_to_str(d):
    """
    Converts hyperparameter dictionary into human-readable string
    """
    strg = '_'.join([k + '=' + str(v) for k, v, in d.items()
                     if type(v) in [bool, int, float, str, dict]])
    return strg


def make_synthetic_bayes_net(*,
    key,
    n_vars,
    graph_model,
    generative_model,
    inference_model,
    n_observations=100,
    n_ho_observations=100,
    n_intervention_sets=10,
    perc_intervened=0.1,
    verbose=False,
):
    """
    Returns an instance of `Target` for evaluation of a method against 
    a ground truth synthetic Bayesian network

    Args:
        key: rng key
        c (int): seed
        graph_model (GraphDistribution): graph model object 
        generative_model (BasicModel): BN model object for generating the observations
        inference_model (BasicModel): JAX-BN model object for inference
        n_observations (int): number of observations generated for posterior inference
        n_ho_observations (int): number of held-out observations generated for validation
        n_intervention_sets (int): number of different interventions considered overall
            for generating interventional data
        perc_intervened (float): percentage of nodes intervened upon (set to 0) in 
            an intervention.

    Returns:
        `Target` 
    """

    # remember random key
    passed_key = key.copy()

    # generate ground truth observations
    key, subk = random.split(key)
    g_gt = graph_model.sample_G(subk)
    g_gt_mat = jnp.array(graph_to_mat(g_gt))

    key, subk = random.split(key)
    theta = generative_model.sample_parameters(key=subk, g=g_gt)

    key, subk = random.split(key)
    x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta)

    key, subk = random.split(key)
    x_ho = generative_model.sample_obs(key=subk, n_samples=n_ho_observations, g=g_gt, theta=theta)

    # 10 random 0-clamp interventions where `perc_interv` % of nodes are intervened on
    # list of (interv dict, x)
    x_interv = []
    for idx in range(n_intervention_sets):
    
        # random intervention
        key, subk = random.split(key)
        n_interv = jnp.ceil(n_vars * perc_intervened).astype(jnp.int32)
        interv_targets = random.choice(subk, n_vars, shape=(n_interv,), replace=False)
        interv = {k: 0.0 for k in interv_targets}

        # observations from p(x | theta, G, interv) [n_samples, n_vars]
        key, subk = random.split(key)
        x_interv_ = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta, interv=interv)
        x_interv.append((interv, x_interv_))

    if verbose:
        print(f'Sampled BN with {jnp.sum(g_gt_mat).item()}-edge DAG :\t {adjmat_to_str(g_gt_mat)}')

    # return and save generated target object
    obj = Target(
        passed_key=passed_key,
        graph_model=graph_model,
        generative_model=generative_model,
        inference_model=inference_model,
        n_vars=n_vars,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        g=g_gt_mat,
        theta=theta,
        x=x,
        x_ho=x_ho,
        x_interv=x_interv,
    )
    return obj
    

def make_graph_model(*, n_vars, graph_prior_str, edges_per_node=2):
    """
    Instantiates graph model

    Args:
        n_vars: number of variables
        graph_prior_str: specifier (`er`, `sf`)

    Returns:
        `GraphDistribution`
    """
    if graph_prior_str == 'er':
        graph_model = ErdosReniDAGDistribution(
            n_vars=n_vars, 
            n_edges=edges_per_node * n_vars)

    elif graph_prior_str == 'sf':
        graph_model = ScaleFreeDAGDistribution(
            n_vars=n_vars,
            n_edges_per_node=edges_per_node)

    else:
        assert n_vars <= 5 
        graph_model = UniformDAGDistributionRejection(
            n_vars=n_vars)

    return graph_model


def make_linear_gaussian_equivalent_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, n_observations=100,
    n_ho_observations=100):
    """
    Samples a synthetic linear Gaussian BN instance 
    with Bayesian Gaussian equivalent (BGe) marginal likelihood 
    as inference model to weight each DAG in an MEC equally

    By marginalizing out the parameters, the BGe model does not 
    allow inferring the parameters (theta).
    
    Args:
        key: rng key
        n_vars (int): number variables in BN
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        mean_edge (float): edge weight mean
        sig_edge (float): edge weight stddev
    
    Returns:
        `Target` 
    """

    # init models
    graph_model = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = LinearGaussian(
        obs_noise=obs_noise, mean_edge=mean_edge, 
        sig_edge=sig_edge, g_dist=graph_model)

    inference_model = BGeJAX(
        mean_obs=jnp.zeros(n_vars), 
        alpha_mu=1.0, alpha_lambd=n_vars + 2)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    target = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_model=graph_model,
        generative_model=generative_model,
        inference_model=inference_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations)

    return target


def make_linear_gaussian_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, mean_edge=0.0, sig_edge=1.0, n_observations=100,
    n_ho_observations=100):
    """
    Samples a synthetic linear Gaussian BN instance 

    Args:
        key: rng key
        n_vars (int): number variables in BN
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        mean_edge (float): edge weight mean
        sig_edge (float): edge weight stddev
    
    Returns:
        `Target` 
    """

    # init models
    graph_model = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = LinearGaussian(
        obs_noise=obs_noise, mean_edge=mean_edge, 
        sig_edge=sig_edge, g_dist=graph_model)

    inference_model = LinearGaussianJAX(
        obs_noise=obs_noise, mean_edge=mean_edge, 
        sig_edge=sig_edge)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    target = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_model=graph_model,
        generative_model=generative_model,
        inference_model=inference_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations)

    return target


def make_nonlinear_gaussian_model(*, key, n_vars=20, graph_prior_str='sf', 
    obs_noise=0.1, sig_param=1.0, hidden_layers=[5,], n_observations=100,
    n_ho_observations=100):
    """
    Samples a synthetic nonlinear Gaussian BN instance 
    where the local conditional distributions are parameterized
    by fully-connected neural networks

    Args:
        key: rng key
        n_vars (int): number variables in BN
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        sig_param (float): stddev of the BN parameters,
            i.e. here the neural net weights and biases
        hidden_layers (list): list of ints specifying the hidden layer (sizes)
            of the neural nets parameterizatin the local condtitionals
    
    Returns:
        `Target` 
    """

    # init models
    graph_model = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)

    generative_model = DenseNonlinearGaussianJAX(
        obs_noise=obs_noise, sig_param=sig_param,
        hidden_layers=hidden_layers, g_dist=graph_model)

    inference_model = DenseNonlinearGaussianJAX(
        obs_noise=obs_noise, sig_param=sig_param,
        hidden_layers=hidden_layers, g_dist=graph_model)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    target = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_model=graph_model,
        generative_model=generative_model,
        inference_model=inference_model,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations)

    return target
