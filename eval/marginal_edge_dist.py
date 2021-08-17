import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

from collections import defaultdict
import numpy as onp
import time
import tqdm
import pickle
import pandas as pd

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
from jax.scipy.special import logsumexp
from jax.ops import index, index_add, index_update

from cdt.metrics import get_CPDAG

from dibs.utils.func import id2bit, bit2id, mask_topk, log_prob_ids, particle_empirical, particle_empirical_mixture
from dibs.utils.graph import adjmat_to_str, make_all_dags, mat_to_graph, graph_to_mat, elwise_acyclic_constr_nograd
from dibs.eval.metrics import neg_ave_log_marginal_likelihood

from dibs.graph.distributions import UniformDAGDistributionRejection, ErdosReniDAGDistribution
from dibs.models.linearGaussianGaussianEquivalent import BGe, BGeJAX
from dibs.models.linearGaussianGaussian import LinearGaussianGaussian, LinearGaussianGaussianJAX

from dibs.kernel.basic import FrobeniusSquaredExponentialKernel
from dibs.svgd.dot_graph_svgd import DotProductGraphSVGD
from dibs.svgd.batch_dot_graph_svgd import BatchedDotProductGraphSVGD

from config.svgd import marginal_config

import numpy as np

STORE_ROOT = ['store']


def compute_marginal_edge_beliefs(gs, logprobs, pw_list=[], verbose=False):
    """Computes marginal edge beliefs for single edges and provided pairs of edges"""

    n_vars = gs.shape[-1]
    assert(n_vars <= 5)

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        logprobs[..., None, None], b=gs.astype(logprobs.dtype),
        axis=0, return_sign=True)

    p_edge = log_edge_belief_sgn * jnp.exp(log_edge_belief)
    if verbose:
        print('p_edge')
        print(p_edge)
        print()

    pw_edge = jnp.zeros((n_vars, n_vars, n_vars, n_vars))

    for (a, b), (c, d) in pw_list:
        # [N, ]
        indicator_both_exist = (gs[:, a, b] == 1) & (gs[:, c, d] == 1)
        log_indicator_belief, log_indicator_belief_sgn = logsumexp(
            logprobs, b=indicator_both_exist.astype(logprobs.dtype), return_sign=True)
        gt_p_edge = log_indicator_belief_sgn * jnp.exp(log_indicator_belief)

        pw_edge = index_add(pw_edge, index[a, b, c, d], gt_p_edge)

        if verbose:
            print(f'p({a}->{b} and {c}->{d} | D) = ', gt_p_edge)
    
    if verbose:
        print()

    return p_edge, pw_edge



if __name__ == '__main__':

    jnp.set_printoptions(precision=6, suppress=True)
    key = random.PRNGKey(0)

    # example
    n_observations = 100
    verbose = True
    filename = 'mec_toy_4'
    load = False
    load = True

    # dibs
    batch_size = 30
    n_steps = 500
    n_particles = 30
 
    if not load:
        """Save a closed-form distribution computed via exhaustive enumeration"""

        gt = jnp.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])

        theta = jnp.array([
            [   0,    2,   -2,   0],
            [   0,    0,    0,   3],
            [   0,    0,    0,   1],
            [   0,    0,    0,   0],
        ])

        """
        Computes closed form BGe posterior for the graph.
        Then runs DiBS and prints its posterior probabilities
        """

        # igraph.Graph
        print('\nGround truth DAG')
        print(gt)
        print(adjmat_to_str(gt))
        g_gt = mat_to_graph(gt)
        n_vars = gt.shape[-1]

        # # CPDAG
        print('\nGround truth CPDAG')
        gt_cpdag = jnp.array(get_CPDAG(onp.array(gt)), dtype=gt.dtype)
        print(gt_cpdag)
        print(adjmat_to_str(gt_cpdag))
        print()

        """Generate standard observations"""
        generative_model = LinearGaussianGaussian(
            obs_noise=1.0, 
            mean_edge=None,
            sig_edge=None,
            g_dist=None,
            verbose=True,
        )

        if n_vars <= 5:
            print('theta')
            print(theta * gt)

        key, subk = random.split(key)
        x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta)


        """BGe settings"""
        g_dist = UniformDAGDistributionRejection(n_vars=n_vars)
        model = BGe(
            g_dist=g_dist,
            mean_obs=0.0 * jnp.ones(n_vars),
            alpha_mu=1.0, 
            alpha_lambd=n_vars + 2,
            verbose=True,
        )

        score = model.log_marginal_likelihood_given_g(g=g_gt, x=x)
        print('GT score', score)
        if n_vars > 5:
            exit()

        all_dags = make_all_dags(n_vars=n_vars, return_matrices=False)
        z_g = g_dist.log_normalization_constant(all_g=all_dags)
        print('Number of possible DAGs:', len(all_dags), '\n')

        # log p(D)
        log_evidence = model.log_marginal_likelihood(x=x, all_g=all_dags, z_g=z_g)
        print('log p(D)', log_evidence, '\n')

        # log p(G | D)
        posterior_ids, posterior_log_probs = [], [] # (unique graph ids, log probs)
        mecs = defaultdict(list)
        for j, g_ in enumerate(tqdm.tqdm(all_dags, desc="p(G|X) log_posterior", disable=False)):

            log_posterior = model.log_posterior_graph_given_obs(
                g=g_, x=x, log_marginal_likelihood=log_evidence, z_g=z_g)

            posterior_ids.append(bit2id(jnp.array(graph_to_mat(g_))[jnp.newaxis]).squeeze(0))
            posterior_log_probs.append(log_posterior)

            # for printing purposes
            mecs[round(log_posterior.item(), 8)].append(g_)

        posterior_ids = jnp.array(posterior_ids) # uint8
        posterior_log_probs = jnp.array(posterior_log_probs) # float64

        # check valid distribution
        normalization_const = logsumexp(posterior_log_probs)
        if not jnp.allclose(normalization_const, 0, atol=1e-3):
            print('logsumexp = {}   sumexp = {}'.format(normalization_const, jnp.exp(normalization_const)))
            print('Unnormalized distribution detected.') 
            assert(False)

        # save
        save_path = os.path.abspath(os.path.join('..', *STORE_ROOT, filename + '.pk'))
        obj = {
            'gt': gt,
            'theta': theta,
            'x': x,
            'posterior_ids': posterior_ids,
            'posterior_log_probs': posterior_log_probs,
            'mecs': mecs,
        }
        with open(save_path, 'wb') as fp:
            pickle.dump(obj, fp)
            print('Saved at:', save_path)

    else:
        """Load previously generated object"""

        load_path = os.path.abspath(os.path.join('..', *STORE_ROOT, filename + '.pk'))
        with open(load_path, 'rb') as fp:
            obj = pickle.load(fp)
            gt = obj['gt']
            theta = obj['theta']
            x = obj['x']
            posterior_ids = obj['posterior_ids']
            posterior_log_probs = obj['posterior_log_probs']
            mecs = obj['mecs']

            # print info
            print('\nGround truth DAG')
            print(gt)
            print(adjmat_to_str(gt))
            g_gt = mat_to_graph(gt)
            n_vars = gt.shape[-1]

            # # CPDAG
            print('\nGround truth CPDAG')
            gt_cpdag = jnp.array(get_CPDAG(onp.array(gt)), dtype=gt.dtype)
            print(gt_cpdag)
            print(adjmat_to_str(gt_cpdag))
            print()

            print('theta')
            print(theta * gt)

            print('x shape and IQR per var')
            print(x.shape)
            print(jnp.percentile(x, [25, 75], axis=0), '\n')

    """Ground truth posterior"""
    kk = 3
    print(f'Top {kk} MECs:')
    for k in sorted(mecs.keys(), reverse=True)[:kk]:
        p_each = jnp.exp(k)
        print(f'p(G | D) = {p_each:6.4f}  [Total: {len(mecs[k]) * p_each:6.4f}]')
        for g_ in mecs[k]:
            print(adjmat_to_str(graph_to_mat(g_))
                + ('  ===> GROUND TRUTH' if jnp.all(graph_to_mat(g_gt) == graph_to_mat(g_)) else ''))
        print()

    """GT posterior marginals"""
    pairwise_edges = [
        ((1, 0), (0, 2)),
        ((1, 0), (2, 0)),
        ((0, 1), (0, 2)),
        ((0, 1), (2, 0)),
        #
        ((1, 3), (3, 2)),
        ((1, 3), (2, 3)),
        ((3, 1), (3, 2)),
        ((3, 1), (2, 3)),
    ]
    posterior_graphs = id2bit(posterior_ids, n_vars)
    gt_p_edge, gt_pw_edge = compute_marginal_edge_beliefs(posterior_graphs, posterior_log_probs, pw_list=pairwise_edges, verbose=True)

    """SVGD"""
    n_dim = n_vars
    kernel = FrobeniusSquaredExponentialKernel(h=2.0, graph_embedding_representation=True)

    # temperature hyperparameters
    def alpha_sched(t):
        return t

    def beta_sched(t):
        return 2.0 * t
    
    def log_prior(single_w_prob):
        return 0.0

    inference_model = BGeJAX(
        mean_obs=0.0 * jnp.ones(n_vars),
        alpha_mu=1.0,
        alpha_lambd=n_vars + 2,
        verbose=True,
    )

    def log_target(single_w, b):
        ''' p(D | G) 
            returns shape [1, ]
            will later be vmapped

            single_w : [n_vars, n_vars] in {0, 1}
        '''
        score = inference_model.log_marginal_likelihood_given_g(w=single_w, data=x)
        return score

    print('GT score', log_target(jnp.array(gt), 0))

    # evaluates [n_particles, n_vars, n_vars], [n_observations, n_vars] in batch on held-out data 
    eltwise_log_marg_likelihood = jit(vmap(lambda w, x_: inference_model.log_marginal_likelihood_given_g(w=w, data=x_), (0, None), 0))
    
    # assuming dot product representation of graph
    key, subk = random.split(key)
    latent_prior_std = 1.0 / jnp.sqrt(n_dim)
    init_particles = random.normal(subk, shape=(batch_size, n_particles, n_vars, n_dim, 2)) * latent_prior_std

    # svgd
    # svgd = DotProductGraphSVGD(
    svgd = BatchedDotProductGraphSVGD(        
        n_vars=n_vars,
        n_dim=n_dim,
        optimizer=dict(name='rmsprop', stepsize=0.005),
        kernel=kernel, 
        target_log_prior=log_prior,
        target_log_prob=log_target,
        alpha=alpha_sched,
        beta=beta_sched,
        gamma=lambda _: jnp.array([1.0]),
        tau=lambda _: jnp.array([1.0]),
        n_grad_mc_samples=128,
        n_acyclicity_mc_samples=32,
        clip=None,
        fix_rotation='not',
        grad_estimator='score',
        score_function_baseline=0.0,
        repulsion_in_prob_space=False,
        latent_prior_std=latent_prior_std,
        constraint_prior_graph_sampling='soft',
        graph_embedding_representation=True,
        verbose=True)

    # metrics

    # for batch svgd
    def neg_ave_log_marginal_likelihood_hard(params):
        '''
        Computes negative log posterior predictive on held-out data
        '''
        mean_neg_ave_log_marginal_likelihood = jnp.array([
            neg_ave_log_marginal_likelihood(dist=particle_empirical(hard_g), x=x,
                eltwise_log_target=lambda w_, x_: eltwise_log_marg_likelihood(w_, x_))
            for hard_g in svgd.particle_to_hard_g(params['batch_x'])
        ]).mean()

        return 'neg MLL [hard] {:8.04f}'.format(mean_neg_ave_log_marginal_likelihood.item())
        

    # evaluate
    key, subk = random.split(key)
    batch_dibs_latent_particles = svgd.sample_particles(
        key=subk,
        n_steps=n_steps, 
        init_particles=init_particles.copy(),
        metric_every=50,
        eval_metrics=[neg_ave_log_marginal_likelihood_hard])

    """Marginals posterior"""
    batch_dibs_g = svgd.particle_to_hard_g(batch_dibs_latent_particles)

    ave_p_edge = {
        'empirical':  jnp.zeros((n_vars, n_vars)),
        'mixture':  jnp.zeros((n_vars, n_vars)),
    }
    ave_pw_p_edge = {
        'empirical':  jnp.zeros((n_vars, n_vars, n_vars, n_vars)),
        'mixture':  jnp.zeros((n_vars, n_vars, n_vars, n_vars)),
    }
    
    for b, dibs_g in enumerate(batch_dibs_g):

        # filter cyclic
        is_dag = elwise_acyclic_constr_nograd(dibs_g, n_vars) == 0
        if is_dag.sum() == 0:
            print('Only cyclic graphs returned. Increase steps.')
            exit()
        dibs_g = dibs_g[is_dag, :, :]

        dibs_empirical = particle_empirical(dibs_g)
        dibs_empirical_g = id2bit(dibs_empirical[0], n_vars)

        dibs_mixture = particle_empirical_mixture(dibs_g, vmap(lambda g_: log_target(g_, 0), 0, 0))
        dibs_mixture_g = id2bit(dibs_mixture[0], n_vars)

        # empirical
        gt_p_edge_empirical, gt_pw_edge_empirical = compute_marginal_edge_beliefs(dibs_empirical_g, dibs_empirical[1], pw_list=pairwise_edges)
        ave_p_edge['empirical'] += gt_p_edge_empirical / batch_size
        ave_pw_p_edge['empirical'] += gt_pw_edge_empirical / batch_size
       
        # mixture
        gt_p_edge_mixture, gt_pw_edge_mixture = compute_marginal_edge_beliefs(dibs_mixture_g, dibs_mixture[1], pw_list=pairwise_edges)
        ave_p_edge['mixture'] += gt_p_edge_mixture / batch_size
        ave_pw_p_edge['mixture'] += gt_pw_edge_mixture / batch_size

    print('----------------')
    for name in ['empirical', 'mixture']:
        print(name)
        print('ave p_edge')
        print(ave_p_edge[name])

        for (a, b), (c, d) in pairwise_edges:
            print(f'p({a}->{b} and {c}->{d} | D) = ',  ave_pw_p_edge[name][a, b, c, d])
        print()
