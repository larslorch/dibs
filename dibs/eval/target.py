import sys, os
import csv
import pickle
from jax.core import Value
import pandas as pd
from collections import namedtuple, defaultdict
import argparse
import tqdm

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_mul, index_update
import jax.lax as lax 
from jax import random

from dibs.graph.distributions import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from dibs.utils.graph import *

from dibs.models.linearGaussianGenerativeModel import GBNSampler
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX
from dibs.models.FCGaussian import FCGaussianJAX

from dibs.utils.func import mask_topk, id2bit, bit2id, log_prob_ids, particle_empirical, kullback_leibler_dist, dist_is_none

from cdt.data.loader import load_dataset

STORE_ROOT = ['store'] 
TUEBINGEN_ROOT = ['data', 'tuebingen-cause-effect-pairs']


Target = namedtuple('Target', (
    'filename',                 # identifier string; object is stored as `filename`.pk
    'c',                        # identifier int to distinguish several random initializations
    'passed_key',               # jax.random key passed _into_ the function generating this object
    'g_dist',
    'graph_prior_str',
    'generative_model_str',
    'generative_model_kwargs',
    'inference_model_str',
    'inference_model_kwargs',
    'n_vars',
    'n_observations',
    'n_ho_observations',
    'n_posterior_g_samples',    
    'g',                        # [n_vars, n_vars]
    'theta',
    'x',                        # [n_observation, n_vars]    data
    'x_ho',                     # [n_ho_observation, n_vars] held-out data
    'x_interv',                 # list of (interv dict, held-out interventional data) 
    'posterior_g_samples',      # [n_posterior_g_samples, n_vars, n_vars]
    'log_posterior',            # distribution tuple of the posterior with log_posterior = (graph_ids, log_probs)
                                # where graph_ids.shape[0] == log_probs.shape[0] is number of non-zero probability graphs (usually number of DAGs) 
                                # logsumexp(log_probs) = 0, i.e. the distribution is normalized
))


def save_pickle(obj, relpath):
    '''Saves `obj` to `path` using pickle'''
    save_path = os.path.abspath(os.path.join(
        '..', *STORE_ROOT, relpath + '.pk'
    ))
    with open(save_path, 'wb') as fp:
        pickle.dump(obj, fp)

def load_pickle(relpath):
    '''Loads object from `path` using pickle'''
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

def load_tuebingen_metadata():
    """
    Cause-Effect benchmark data by 

    J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schoelkopf:
    "Distinguishing cause from effect using observational data: methods and benchmarks",
    Journal of Machine Learning Research 17(32):1-102, 2016

    """

    meta_data_path = os.path.abspath(os.path.join(
        '..', *TUEBINGEN_ROOT, f'pairmeta.txt'
    ))
    bivariate_pairs = dict()
    with open(meta_data_path, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            
            id = row[0].split(' ')[0]
            start_cause = row[0].split(' ')[1]
            end_cause = row[0].split(' ')[2]
            start_effect = row[0].split(' ')[3]
            end_effect = row[0].split(' ')[4]
            weight = row[0].split(' ')[5]

            # filter multivariate pairs
            if start_cause != end_cause or start_effect != end_effect:
                continue 

            bivariate_pairs[id] = {
                'cause' : int(start_cause) - 1,
                'effect': int(start_effect) - 1,
                'weight': float(weight),
            }
    
    return bivariate_pairs



def load_tuebingen_data(tuebingen_id):
    """
    Cause-Effect benchmark data by 

    J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schoelkopf:
    "Distinguishing cause from effect using observational data: methods and benchmarks",
    Journal of Machine Learning Research 17(32):1-102, 2016

    """

    bivariate_pairs = load_tuebingen_metadata()

    # load data
    if tuebingen_id not in bivariate_pairs.keys():
        raise ValueError(f'The tuebingen id `{tuebingen_id}` does not exist, or is not bivariate.')
    
    info = bivariate_pairs[tuebingen_id]
    data_path = os.path.abspath(os.path.join(
        '..', *TUEBINGEN_ROOT, f'pair{tuebingen_id}.txt'
    ))

    # parse row by row
    try:
        with open(data_path, 'r') as fd:
            reader = csv.reader(fd)
            data_rows = []
            for row in reader:
                if row == []:
                    continue
                row_no_tab = row[0].replace('\t', ' ')
                row_split = row_no_tab.split()
                data_rows.append([float(a) for a in row_split])

            data_raw = jnp.array(data_rows)

            # orient all as 0 -> 1
            pair_data = jnp.vstack([
                data_raw[:, info['cause']], 
                data_raw[:, info['effect']]
            ]).T

            data_full =  {
                'data': pair_data,
                'weight': info['weight'],
            }

    except:
        raise ValueError(f'Could not load tuebingen pair `{tuebingen_id}`')


    # # for sanity check of data import check that `cdt` data is the same
    # try:
    #     # load
    #     tuebingen_data, _ = load_dataset("tuebingen")

    #     # offset in the `cdt` data; `cdt` already skips multivariate data but wrongly renames the pairs
    #     j = int(tuebingen_id)
    #     if j > 51 and j <= 74:
    #         j -= 4
    #     elif j > 74:
    #         j -= 5

    #     # check that the data is the same for sanity check
    #     df = tuebingen_data.loc[f'pair{j}']
    #     data_a, data_b = df[-2], df[-1]
    #     test1 = jnp.vstack([data_a, data_b]).T
    #     print(tuebingen_id, jnp.allclose(pair_data, test1), jnp.abs(pair_data - test1).sum())

    # except:
    #     print(tuebingen_id, 'FAILED', pair_data.shape, test1.shape)
 
    return data_full



def parse_target_str(kwargs):
    """
    Returns tuple of dicts for generative model and inference model 
    as specified by ID string
    """

    if kwargs.joint:
        if kwargs.joint_inference_model == 'lingauss':
            # gen: lingauss -- infer: lingauss
            gen_str, infer_str = 'lingauss', 'lingauss'
            generative_model_kwargs = dict(
                obs_noise=kwargs.lingauss_obs_noise,
                mean_edge=kwargs.lingauss_mean_edge,
                sig_edge=kwargs.lingauss_sig_edge,
            )
            inference_model_kwargs = dict(
                obs_noise=kwargs.lingauss_obs_noise,
                mean_edge=kwargs.lingauss_mean_edge,
                sig_edge=kwargs.lingauss_sig_edge,
            )

        elif kwargs.joint_inference_model == 'fcgauss':
            # gen: fcgauss -- infer: fcgauss
            gen_str, infer_str = 'fcgauss', 'fcgauss'
            generative_model_kwargs = dict(
                obs_noise=kwargs.fcgauss_obs_noise,
                sig_param=kwargs.fcgauss_sig_param,
                dims=[kwargs.fcgauss_n_neurons for _ in range(kwargs.fcgauss_hidden_layers)],
                activation=kwargs.fcgauss_activation,
                bias=kwargs.fcgauss_bias,
            )
            inference_model_kwargs = dict(
                obs_noise=kwargs.fcgauss_obs_noise,
                sig_param=kwargs.fcgauss_sig_param,
                dims=[kwargs.fcgauss_n_neurons for _ in range(kwargs.fcgauss_hidden_layers)],
                activation=kwargs.fcgauss_activation,
                bias=kwargs.fcgauss_bias,
            )

        else:
            raise NotImplementedError()

    else:
        if kwargs.inference_model == 'bge':
            # gen: lingauss -- infer: bge
            gen_str, infer_str = 'lingauss', 'bge'
            generative_model_kwargs = dict(
                obs_noise=kwargs.lingauss_obs_noise,
                mean_edge=kwargs.lingauss_mean_edge,
                sig_edge=kwargs.lingauss_sig_edge,
            )
            inference_model_kwargs = dict(
                mean_obs=jnp.zeros(kwargs.n_vars),
                alpha_mu=kwargs.bge_alpha_mu,
                alpha_lambd=kwargs.n_vars + kwargs.bge_alpha_lambd_add,
            )

        elif kwargs.inference_model == 'lingauss':
            # gen: lingauss -- infer: lingauss
            gen_str, infer_str = 'lingauss', 'lingauss'
            generative_model_kwargs = dict(
                obs_noise=kwargs.lingauss_obs_noise,
                mean_edge=kwargs.lingauss_mean_edge,
                sig_edge=kwargs.lingauss_sig_edge,
            )
            inference_model_kwargs = dict(
                obs_noise=kwargs.lingauss_obs_noise,
                mean_edge=kwargs.lingauss_mean_edge,
                sig_edge=kwargs.lingauss_sig_edge,
            )
        else:
            raise NotImplementedError()

    return (gen_str, generative_model_kwargs), (infer_str, inference_model_kwargs)


def make_ground_truth_posterior(*,
    key,
    c,
    n_vars,
    graph_prior_str,
    generative_model_str,
    generative_model_kwargs,
    inference_model_str,
    inference_model_kwargs,
    n_observations,
    n_ho_observations,
    n_posterior_g_samples,
    n_intervention_sets,
    perc_intervened,
    load,
    verbose=True,
    compute_gt=False,
    real_data=None,
    real_data_held_out=False,
    real_data_normalize=True,
):
    '''
    Returns an instance of `Target`
    for evaluation of a method against a ground truth graph posterior
    '''

    # remember random key
    passed_key = key.copy()

    # real data 
    if real_data is not None:
        filename = real_data + '_' + options_to_str(c=c)
    else:
        filename = generative_model_str + '_' + \
            options_to_str(
                d=n_vars, 
                graph=graph_prior_str,
                n=n_observations, 
                n_ho=n_ho_observations,
                generative_model_kwargs=hparam_dict_to_str(generative_model_kwargs),
                inference_model_kwargs=hparam_dict_to_str(inference_model_kwargs),
                c=c,
            )

    if load:
        try:
            obj = load_pickle(filename)

            # check valid distribution
            if not dist_is_none(obj.log_posterior):
                assert(jnp.allclose(logsumexp(obj.log_posterior[1]), 0, atol=1e-3)) 

            # check the same random key is used
            # assert(jnp.allclose(obj.passed_key, passed_key))

            if verbose:
                print('Loaded {}-edge graph with x_max {:>4.01f}:\t{}'.format(jnp.sum(obj.g).item(), jnp.max(obj.x).item(), adjmat_to_str(obj.g)))
            return obj

        except FileNotFoundError:
            if verbose:
                print('Loading failed: ' + filename + ' does not exist.')
                print('Generating from scratch...')

        except AssertionError:
            if verbose:
                print('Loading failed: Random key changed or distribution is not normalized.')
                print('Generating from scratch...')

    '''Generate ground truth observations''' 
    # init graph distribution 
    if graph_prior_str == 'er':
        g_dist = ErdosReniDAGDistribution(
            n_vars=n_vars, 
            n_edges=2 * n_vars,
            verbose=verbose)

    elif graph_prior_str == 'sf':
        g_dist = ScaleFreeDAGDistribution(
            n_vars=n_vars,
            n_edges_per_node=2,
            verbose=verbose)

    else:
        assert n_vars <= 5 or (real_data is not None)
        g_dist = UniformDAGDistributionRejection(
            n_vars=n_vars,
            verbose=verbose)

    # init generative model
    if generative_model_str == 'gbn':
        generative_model = GBNSampler(
            **generative_model_kwargs,
        )
    elif generative_model_str == 'lingauss':
        generative_model = LinearGaussianGaussian(
            **generative_model_kwargs,
            g_dist=g_dist,
            verbose=False,
        )
    elif generative_model_str == 'fcgauss':
        generative_model = FCGaussianJAX(
            **generative_model_kwargs,
            g_dist=g_dist,
            verbose=False,
        )
    else:
        raise NotImplementedError()

    # init inference model
    if inference_model_str == 'bge':
        inference_model = BGe(
            **inference_model_kwargs,
            g_dist=g_dist,
            verbose=False,
        )
    elif inference_model_str == 'lingauss':
        inference_model = LinearGaussianGaussian(
            **inference_model_kwargs,
            g_dist=g_dist,
            verbose=False,
        )
    elif inference_model_str == 'fcgauss':
        inference_model = FCGaussianJAX(
            **generative_model_kwargs,
            g_dist=g_dist,
            verbose=False,
        )
    else:
        raise NotImplementedError()

    """Synthetic or real data """

    if real_data is None:

        # generate observations
        key, subk = random.split(key)
        g_gt = g_dist.sample_G(subk)
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
            print('Generated {}-edge graph with x_max {:>4.01f}:\t{}'.format(
                jnp.sum(g_gt_mat).item(), jnp.max(x).item(), adjmat_to_str(g_gt_mat)))

    elif real_data == 'sachs':
        '''
        11 variables x 7466 samples
        Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan,G. P. (2005). 
        Causal protein-signaling networks derived from multiparameter single-cell data. 
        Science, 308(5721), 523-529.
        ''' 
        sachs_data, sachs_gt_graph = load_dataset("sachs")

        g_gt_mat = jnp.array(nx_adjacency(sachs_gt_graph))
        theta = None

        # discard annotations
        sachs_x_full = jnp.array(pd.DataFrame(data=sachs_data.to_numpy()).values)

        # normalize
        if real_data_normalize:
            sachs_x_full = (sachs_x_full - sachs_x_full.mean(0, keepdims=True)) / sachs_x_full.std(0, keepdims=True)

        # shuffle
        if real_data_held_out:
            key, subk = random.split(key)
            x_full = random.permutation(subk, sachs_x_full)

            cut = int(0.9 * x_full.shape[0])
            x = x_full[:cut]
            x_ho = x_full[cut:]
            x_interv = None

            n_observations, n_vars = x.shape
            n_ho_observations, _ = x_ho.shape
            n_posterior_g_samples = 0
        
        else:
            key, subk = random.split(key)
            x = random.permutation(subk, sachs_x_full)

            x_ho = None
            x_interv = None

            n_observations, n_vars = x.shape
            n_ho_observations = 0
            n_posterior_g_samples = 0

    elif 'tuebingen' in real_data:
        '''
        Dataset of 100 real cause-effect pairs
        '''
        parse_str = real_data.split('-')
        assert len(parse_str) == 2
        tuebingen_id = parse_str[1]       

        # load
        tuebingen_data = load_tuebingen_data(tuebingen_id)
        data_cause, data_effect = tuebingen_data['data'][:, 0], tuebingen_data['data'][:, 1]

        # shuffle cause effect ordering
        key, subk = random.split(key)
        if random.choice(subk, a=jnp.array([True, False])):
            label = 1.0
            data_a, data_b = data_cause, data_effect
        else:
            label = -1.0 
            data_a, data_b = data_effect, data_cause
        
        # normalize
        if real_data_normalize:
            data_a = (data_a - data_a.mean(keepdims=True)) / data_a.std(keepdims=True)
            data_b = (data_b - data_b.mean(keepdims=True)) / data_b.std(keepdims=True)

        # convert to format
        if label == 1.0:
            g_gt_mat = jnp.array([[0, 1], [0, 0]])
        elif label == -1.0:
            g_gt_mat = jnp.array([[0, 0], [1, 0]])
        else:
            raise ValueError('Tuebingen dataset wrongly loaded.')

        theta = None
        x = jnp.array([data_a, data_b]).T

        # randomly permute data rows
        key, subk = random.split(key)
        x = random.permutation(subk, x)
        x_ho = None
        x_interv = None

        n_observations, n_vars = x.shape
        n_ho_observations = 0
        n_posterior_g_samples = 0

    else:
        raise KeyError(f'Invalid real data set requested: `{real_data}`')

    '''Compute true normalized posterior (via exhaustive search) of inference model'''
    if n_vars <= 5 and compute_gt:
        all_dags = make_all_dags(n_vars=n_vars, return_matrices=False)
        z_g = inference_model.g_dist.log_normalization_constant(all_g=all_dags)
        log_marginal_likelihood = inference_model.log_marginal_likelihood(x=x, all_g=all_dags, z_g=z_g)
        
        posterior_ids, posterior_log_probs = [], [] # (unique graph ids, log probs)
        mecs = defaultdict(list)
        for j, g_ in enumerate(tqdm.tqdm(all_dags, desc="p(G|X) log_posterior", disable=not verbose)):
            binary_mat = jnp.array(graph_to_mat(g_))[jnp.newaxis] # [1, d, d]

            id = bit2id(binary_mat).squeeze(0)
            log_posterior_probs = inference_model.log_posterior_graph_given_obs(
                g=g_, x=x, log_marginal_likelihood=log_marginal_likelihood, z_g=z_g)

            posterior_ids.append(id)
            posterior_log_probs.append(log_posterior_probs)

            # for printing purposes
            mecs[round(log_posterior_probs.item(), 6)].append(g_)

        posterior_ids = jnp.array(posterior_ids) # uint8
        posterior_log_probs = jnp.array(posterior_log_probs) # float64

        # check valid distribution
        normalization_const = logsumexp(posterior_log_probs)
        if not jnp.allclose(normalization_const, 0, atol=1e-3):
            print('logsumexp = {}   sumexp = {}'.format(normalization_const, jnp.exp(normalization_const)))
            print('Unnormalized distribution detected.') 
            assert(False)

        if verbose:
            print('possible adjacency matrices: ', 2 ** (n_vars * n_vars))
            print('target top 5 scores:         ', posterior_log_probs[mask_topk(posterior_log_probs, 5)])
            print('target top 5 probs:          ', jnp.exp(posterior_log_probs[mask_topk(posterior_log_probs, 5)]))

            kk = 5
            print(f'Top {kk} MECs:')
            for k in sorted(mecs.keys(), reverse=True)[:kk]:
                p_each = jnp.exp(k)
                print(f'p(G | D) = {p_each:6.4f}  [Total: {len(mecs[k]) * p_each:6.4f}]')
                for g_ in mecs[k]:
                    print(adjmat_to_str(graph_to_mat(g_))
                        + ('  ===> GROUND TRUTH' if jnp.all(graph_to_mat(g_gt) == graph_to_mat(g_)) else ''))
                print()
    

        # sample graphs from posterior for MMD
        key, subk = random.split(key)
        posterior_g_sample_ids_idx = random.categorical(subk, posterior_log_probs, shape=(n_posterior_g_samples,))
        posterior_g_sample_ids = posterior_ids[posterior_g_sample_ids_idx]
        posterior_g_samples = id2bit(posterior_g_sample_ids, n_vars)

    else:
        posterior_ids = None
        posterior_log_probs = None
        posterior_g_samples = None

    # return and save generated target object
    obj = Target(
        c=c,
        passed_key=passed_key,
        filename=filename,
        g_dist=g_dist,
        graph_prior_str=graph_prior_str,
        generative_model_str=generative_model_str,
        generative_model_kwargs=generative_model_kwargs,
        inference_model_str=inference_model_str,
        inference_model_kwargs=inference_model_kwargs,
        n_vars=n_vars,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        n_posterior_g_samples=n_posterior_g_samples,
        g=g_gt_mat,
        posterior_g_samples=posterior_g_samples,
        theta=theta,
        x=x,
        x_ho=x_ho,
        x_interv=x_interv,
        log_posterior=(posterior_ids, posterior_log_probs),
    )
    save_pickle(obj, filename)
    return obj
    
