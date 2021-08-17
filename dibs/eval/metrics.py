import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")

import os
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, vmap
from jax.ops import index, index_add, index_update
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp

import numpy as onp

from dibs.utils.func import id2bit, bit2id, pairwise_structural_hamming_distance
from dibs.utils.tree import tree_mul, tree_select
from dibs.utils.graph import elwise_acyclic_constr_nograd

from cdt.metrics import get_CPDAG
from sklearn import metrics as sklearn_metrics

#
# marginal posterior p(G | D) metrics
#

def l1_edge_belief(*, dist, g):
    '''
        dist:           log distribution tuple
        g:              [n_vars, n_vars] ground truth graph
    '''
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist 
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as "wrong on every edge"
        return n_vars * (n_vars - 1) / 2

    particles = particles_cyc[is_dag, :, :]
    log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)

    # L1 edge error
    p_edge = log_edge_belief_sgn * jnp.exp(log_edge_belief)
    p_no_edge = 1 - p_edge
    err_connected = jnp.sum(g * p_no_edge)
    err_notconnected = jnp.sum(
        jnp.triu((1 - g) * (1 - g).T * (1 - p_no_edge * p_no_edge.T), k=0))

    err = err_connected + err_notconnected
    return err


def expected_shd(*, dist, g, use_cpdag=False):
    '''
        dist:           log distribution tuple
        g:              [n_vars, n_vars] ground truth graph
    '''
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as "wrong on every edge"
        return n_vars * (n_vars - 1) / 2
    
    particles = particles_cyc[is_dag, :, :]
    log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
    
    # convert to cpdag?
    if use_cpdag:
        particles = jnp.array([get_CPDAG(onp.array(mat)) for mat in particles]).astype(particles.dtype)
        g = jnp.array(get_CPDAG(onp.array(g))).astype(g.dtype)
    
    # compute shd for each graph
    shds = pairwise_structural_hamming_distance(x=particles, y=g[None]).squeeze(1)

    # expected SHD = sum_G p(G) SHD(G)
    log_expected_shd, log_expected_shd_sgn = logsumexp(
        log_weights, b=shds.astype(log_weights.dtype), axis=0, return_sign=True)

    expected_shd = log_expected_shd_sgn * jnp.exp(log_expected_shd)
    return expected_shd


def expected_edges(*, dist, g):
    '''
        dist:           log distribution tuple
        g:              [n_vars, n_vars] ground truth graph
    '''
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # if no acyclic graphs, count the edges of the cyclic graphs; more consistent 
        n_edges_cyc = particles_cyc.sum(axis=(-1, -2))
        log_expected_edges_cyc, log_expected_edges_cyc_sgn = logsumexp(
            log_weights_cyc, b=n_edges_cyc.astype(log_weights_cyc.dtype), axis=0, return_sign=True)

        expected_edges_cyc = log_expected_edges_cyc_sgn * jnp.exp(log_expected_edges_cyc)
        return expected_edges_cyc
    
    particles = particles_cyc[is_dag, :, :]
    log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
    
    # count edges for each graph
    n_edges = particles.sum(axis=(-1, -2))

    # expected edges = sum_G p(G) edges(G)
    log_expected_edges, log_expected_edges_sgn = logsumexp(
        log_weights, b=n_edges.astype(log_weights.dtype), axis=0, return_sign=True)

    expected_edges = log_expected_edges_sgn * jnp.exp(log_expected_edges)
    return expected_edges


def threshold_metrics(*, dist, g, undirected_cpdag_oriented_correctly=False):
    '''
        dist:           log distribution tuple
        g:              [n_vars, n_vars] ground truth graph
    '''
    n_vars = g.shape[0]
    g_flat = g.reshape(-1)

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist 
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as random/junk classifier
        # for AUROC: 0.5
        # for precision-recall: no. true edges/ no. possible edges
        return {
            'roc_auc': 0.5,
            'prc_auc': (g.sum() / (n_vars * (n_vars - 1))).item(),
            'ave_prec': (g.sum() / (n_vars * (n_vars - 1))).item(),
        }

    particles = particles_cyc[is_dag, :, :]
    log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])

    # count undirected cpdag edges in correct orientation IF correct AND only once IF incorrect? 
    if undirected_cpdag_oriented_correctly:
        # find undirected edges of inferred CPDAGs
        particle_cpdags = jnp.array([get_CPDAG(onp.array(mat)) for mat in particles]).astype(particles.dtype)
        particle_cpdag_undir_edge = ((particle_cpdags == 1) & (particle_cpdags.transpose((0, 2, 1)) == 1))
        
        # direct them according to the ground truth IF the ground truth has an edge there
        particle_cpdag_undir_edge_correct = particle_cpdag_undir_edge & ((g[None] == 1) | (g[None].transpose((0, 2, 1)) == 1))
        particles = jnp.where(particle_cpdag_undir_edge_correct, g, particles)

        # direct them one way IF the ground truth does not have an edge here (to only count the mistake once)
        particle_cpdag_undir_edge_incorrect = particle_cpdag_undir_edge & ((g[None] == 0) & (g[None].transpose((0, 2, 1)) == 0))
        particles = jnp.where(particle_cpdag_undir_edge_incorrect, jnp.triu(jnp.ones_like(g, dtype=g.dtype)), particles)

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)

    # L1 edge error
    p_edge = log_edge_belief_sgn * jnp.exp(log_edge_belief)
    p_edge_flat = p_edge.reshape(-1)

    # threshold metrics 
    fpr_, tpr_, _ = sklearn_metrics.roc_curve(g_flat, p_edge_flat)
    roc_auc_ = sklearn_metrics.auc(fpr_, tpr_)
    precision_, recall_, _ = sklearn_metrics.precision_recall_curve(g_flat, p_edge_flat)
    prc_auc_ = sklearn_metrics.auc(recall_, precision_)
    ave_prec_ = sklearn_metrics.average_precision_score(g_flat, p_edge_flat)
    
    return {
        'fpr': fpr_.tolist(),
        'tpr': tpr_.tolist(),
        'roc_auc': roc_auc_,
        'precision': precision_.tolist(),
        'recall': recall_.tolist(),
        'prc_auc': prc_auc_,
        'ave_prec': ave_prec_,
    }


def best_train_neg_log_marginal_likelihood(*, dist, eltwise_log_target, x):
    '''
        dist:                   log distribution tuple
        eltwise_log_target:  [M, n_vars, n_vars] -> [M, ] P(D | G) for held-out D
        x:                      [N, n_vars]
    '''
    n_ho_observations, n_vars = x.shape

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        particles = jnp.zeros((1, n_vars, n_vars), dtype=particles_cyc.dtype)

    else:
        particles = particles_cyc[is_dag, :, :]

    # return best marginal log likelihood achieved by any particle
    log_likelihood = eltwise_log_target(particles, x)
    return - log_likelihood.max()


def neg_ave_log_marginal_likelihood(*, dist, eltwise_log_target, x):
    '''
        dist:                   log distribution tuple
        eltwise_log_target:  [M, n_vars, n_vars] -> [M, ] P(D | G) for held-out D
        x:                      [N, n_vars]
    '''
    n_ho_observations, n_vars = x.shape

   # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        particles = jnp.zeros((1, n_vars, n_vars), dtype=particles_cyc.dtype)
        log_weights = jnp.array([0.0], dtype=log_weights_cyc.dtype)

    else:
        particles = particles_cyc[is_dag, :, :]
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
        
    log_likelihood = eltwise_log_target(particles, x)

     # - sum_G p(G | D) log(p(x | G))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


def neg_log_posterior_predictive(*, dist, eltwise_log_target, x):
    '''
        dist:                   log distribution tuple
        eltwise_log_target:  [M, n_vars, n_vars] -> [M, ] P(D | G) for held-out D
        x:                      [N, n_vars]
    '''
    n_ho_observations, n_vars = x.shape

   # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        particles = jnp.zeros((1, n_vars, n_vars), dtype=particles_cyc.dtype)
        log_weights = jnp.array([0.0], dtype=log_weights_cyc.dtype)

    else:
        particles = particles_cyc[is_dag, :, :]
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
        
    log_likelihood = eltwise_log_target(particles, x)

    # - log ( sum_G p(G | D) p(x | G) )
    log_score = logsumexp(log_weights + log_likelihood, axis=0)
    score = - log_score
    return score


#
# joint posterior p(G, theta | D) metrics
#

def best_train_neg_log_likelihood(*, dist, eltwise_log_joint_target, x):
    '''
        dist:                       log distribution tuple
        eltwise_log_joint_target:   [:, n_vars, n_vars], [:, n_vars, n_vars], [N, n_vars] -> [:, ] P(D | G, theta) for held-out D=x
        x:                          [N, n_vars]
    '''
    n_ho_observations, n_vars = x.shape

    ids_cyc, theta_cyc, log_weights_cyc = dist
    hard_g_cyc = id2bit(ids_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(hard_g_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        hard_g = tree_mul(hard_g_cyc, 0.0)
        theta = tree_mul(theta_cyc, 0.0)
        log_weights = tree_mul(log_weights_cyc, 0.0)

    else:
        hard_g = hard_g_cyc[is_dag, :, :]
        theta = tree_select(theta_cyc, is_dag)
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])

    log_likelihood = eltwise_log_joint_target(hard_g, theta, x)
    return - log_likelihood.max()


def neg_ave_log_likelihood(*, dist, eltwise_log_joint_target, x):
    '''
        dist:        3-tuple of [L, ...] graph ids
                                [L, ...] parameters theta
                                [L,]     log weights

        eltwise_log_joint_target:   [:, n_vars, n_vars], [:, n_vars, n_vars], [N, n_vars] -> [:, ] P(D | G, theta) for held-out D=x
        x:                          [N, n_vars]
    '''
    n_ho_observations, n_vars = x.shape

    ids_cyc, theta_cyc, log_weights_cyc = dist
    hard_g_cyc = id2bit(ids_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(hard_g_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        hard_g = tree_mul(hard_g_cyc, 0.0)
        theta = tree_mul(theta_cyc, 0.0)
        log_weights = tree_mul(log_weights_cyc, 0.0)

    else:
        hard_g = hard_g_cyc[is_dag, :, :]
        theta = tree_select(theta_cyc, is_dag)
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
        
    log_likelihood = eltwise_log_joint_target(hard_g, theta, x)

    # - sum_G p(G, theta | D) log(p(x | G, theta))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


def neg_log_joint_posterior_predictive(*, dist, eltwise_log_joint_target, x):
    '''
        dist:        3-tuple of [L, ...] graph ids
                                [L, ...] parameters theta
                                [L,]     log weights

        eltwise_log_joint_target:   [:, n_vars, n_vars], [:, n_vars, n_vars], [N, n_vars] -> [:, ] P(D | G, theta) for held-out D=x
        x:                          [N, n_vars]
    '''
    n_ho_observations, n_vars = x.shape

    ids_cyc, theta_cyc, log_weights_cyc = dist
    hard_g_cyc = id2bit(ids_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(hard_g_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        hard_g = tree_mul(hard_g_cyc, 0.0)
        theta = tree_mul(theta_cyc, 0.0)
        log_weights = tree_mul(log_weights_cyc, 0.0)

    else:
        hard_g = hard_g_cyc[is_dag, :, :]
        theta = tree_select(theta_cyc, is_dag)
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
        
    log_likelihood = eltwise_log_joint_target(hard_g, theta, x)

    # - log ( sum_G p(G , theta | D) p(x_ho | G, theta) )
    log_score = logsumexp(log_weights + log_likelihood, axis=0)
    score = - log_score
    return score

