
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce
import numpy as onp

from dibs.utils.graph import *
from dibs.utils.func import (
    mask_topk, id2bit, bit2id, log_prob_ids, kullback_leibler_dist, 
    particle_empirical, particle_empirical_mixture,
    particle_joint_empirical, particle_joint_mixture,
)

from dibs.eval.metrics import (
    l1_edge_belief, expected_shd, expected_edges,
    neg_ave_log_marginal_likelihood, neg_ave_log_likelihood, 
    neg_log_posterior_predictive, neg_log_joint_posterior_predictive, 
    threshold_metrics,
)
from dibs.eval.target import make_ground_truth_posterior
from dibs.utils.func import symlog, expand_by
from dibs.utils.tree import tree_unzip_leading

def compute_tune_metrics_graph(*, params, kwargs, svgd, variants_target, batch_g_gt, batch_empirical_graph, batch_mixture_graph, use_cpdag):

    batch_x = params['batch_x']
    batch_keys = params['batch_keys']
    batch_hard_g = svgd.particle_to_hard_g(batch_x)
    batch_size = batch_x.shape[0]
    n_variants = len(variants_target)

    m = {}

    # L1 edge belief
    mean_edge_belief_hard = jnp.array([
        l1_edge_belief(dist=q_, g=g_)
        for q_, g_ in zip(batch_empirical_graph, batch_g_gt)
    ]).mean()

    mean_edge_belief_mixture = jnp.array([
        l1_edge_belief(dist=q_, g=g_)
        for q_, g_ in zip(batch_mixture_graph, batch_g_gt)
    ]).mean()

    m["edge_belief_hard"] = mean_edge_belief_hard.item()
    m["edge_belief_mixture"] = mean_edge_belief_mixture.item()

    m["log_edge_belief_hard"] = jnp.log(mean_edge_belief_hard).item()
    m["log_edge_belief_mixture"] = jnp.log(mean_edge_belief_mixture).item()

    # expected SHD
    mean_shd_hard = jnp.array([
        expected_shd(dist=q_, g=g_, use_cpdag=use_cpdag)
        for q_, g_ in zip(batch_empirical_graph, batch_g_gt)
    ]).mean()

    mean_shd_mixture = jnp.array([
        expected_shd(dist=q_, g=g_, use_cpdag=use_cpdag)
        for q_, g_ in zip(batch_mixture_graph, batch_g_gt)
    ]).mean()

    m["shd_hard"] = mean_shd_hard.item()
    m["shd_mixture"] = mean_shd_mixture.item()

    m["log_shd_hard"] = jnp.log(mean_shd_hard).item()
    m["log_shd_mixture"] = jnp.log(mean_shd_mixture).item()

    # expected SHD
    mean_edges_hard = jnp.array([
        expected_edges(dist=q_, g=g_)
        for q_, g_ in zip(batch_empirical_graph, batch_g_gt)
    ]).mean()

    mean_edges_mixture = jnp.array([
        expected_edges(dist=q_, g=g_)
        for q_, g_ in zip(batch_mixture_graph, batch_g_gt)
    ]).mean()

    m["edges_hard"] = mean_edges_hard.item()
    m["edges_mixture"] = mean_edges_mixture.item()

    # AUROC
    mean_roc_auc_hard = 0.0
    for q_, g_ in zip(batch_empirical_graph, batch_g_gt):
        threshold_metrics_hard_ = threshold_metrics(dist=q_, g=g_)
        mean_roc_auc_hard += threshold_metrics_hard_.get('roc_auc', 0.5) # triggered when all DAGs cyclic
    mean_roc_auc_hard /= len(batch_empirical_graph)

    mean_roc_auc_mixture = 0.0
    for q_, g_ in zip(batch_mixture_graph, batch_g_gt):
        threshold_metrics_mixture_ = threshold_metrics(dist=q_, g=g_)
        mean_roc_auc_mixture += threshold_metrics_mixture_.get('roc_auc', 0.5) # triggered when all DAGs cyclic
    mean_roc_auc_mixture /= len(batch_mixture_graph)

    m["neg_roc_auc_hard"] = - mean_roc_auc_hard
    m["neg_roc_auc_mixture"] = - mean_roc_auc_mixture

    # number of graphs that are cyclic amongst hard graphs implied by x
    mean_dag_count = jnp.array([
        (eltwise_acyclic_constr(hard_g, kwargs.n_vars)[0] == 0).sum()
        for hard_g in batch_hard_g
    ]).mean()
    m["cyclic_graphs"] = kwargs.n_particles - mean_dag_count.item()

    # constraint value
    mean_constraint_value = jnp.array([
        svgd.beta(params["t"]) * svgd.eltwise_constraint(
            particles, 
            random.logistic(subk, shape=(100, kwargs.n_vars, kwargs.n_vars)),
            params["t"]).mean()
        for particles, subk in zip(batch_x, batch_keys)
    ]).mean()
    m["mean_constraint_value"] = mean_constraint_value.item()

    mean_constraint_value_unweighted = jnp.array([
        svgd.eltwise_constraint(
            particles,
            random.logistic(subk, shape=(100, kwargs.n_vars, kwargs.n_vars)),
            params["t"]).mean()
        for particles, subk in zip(batch_x, batch_keys)
    ]).mean()
    m["mean_constraint_value_unweighted"] = mean_constraint_value_unweighted.item()

    # number of unique graphs implied by x
    mean_n_unique = jnp.array([
        len(onp.unique(bit2id(hard_g), axis=0))
        for hard_g in batch_hard_g
    ]).mean()
    m["unique_graphs"] = mean_n_unique.item()

    # least decided node probs for variance investigation
    edge_probs = svgd.edge_probs(batch_x, params["t"])
    edge_probs_deviation = jnp.abs(edge_probs - 0.5) # 0 -> random; 0.5 -> decided
    edge_probs_deviation_sorted = edge_probs_deviation.reshape(*edge_probs_deviation.shape[0:2], -1).sort(axis=-1)
    edge_probs_deviation_top1_ave = edge_probs_deviation_sorted[..., 0].mean().item()
    edge_probs_deviation_top3_ave = edge_probs_deviation_sorted[..., 2].mean().item()
    edge_probs_deviation_top5_ave = edge_probs_deviation_sorted[..., 4].mean().item()
    m["edge_prob_top1_undecided"] = 0.5 - edge_probs_deviation_top1_ave
    m["edge_prob_top3_undecided"] = 0.5 - edge_probs_deviation_top3_ave
    m["edge_prob_top5_undecided"] = 0.5 - edge_probs_deviation_top5_ave

    return m


def compute_tune_metrics(*, params, kwargs, svgd, variants_target, eltwise_log_marg_likelihood, batch_g_gt, batch_x_train, batch_x_ho, batch_x_interv):
    """
    Computes tune metrics for marginal inference case
    """

    batch_x = params['batch_x']
    batch_keys = params['batch_keys']
    batch_hard_g = svgd.particle_to_hard_g(batch_x)
    batch_size = batch_x.shape[0]
    n_variants = len(variants_target)

    batch_empirical = [
        particle_empirical(hard_g)
        for b, hard_g in enumerate(batch_hard_g)
    ]

    batch_mixture = [
        particle_empirical_mixture(hard_g, lambda g: svgd.eltwise_log_prob(g, b))
        for b, hard_g in enumerate(batch_hard_g)
    ]

    '''Graph metrics'''
    m = compute_tune_metrics_graph(params=params, kwargs=kwargs, svgd=svgd, variants_target=variants_target, 
        batch_g_gt=batch_g_gt, batch_empirical_graph=batch_empirical, batch_mixture_graph=batch_mixture, use_cpdag=True)

    '''Likelihood based metrics'''
    no_interv_targets = jnp.zeros(kwargs.n_vars).astype(bool)

    for key, batch_q in [('hard', batch_empirical), ('mixture', batch_mixture)]:

        eltwise_log_marg_likelihood_ = lambda w_, x_: eltwise_log_marg_likelihood(w_, x_, no_interv_targets)

        # average log posterior predictive on train data
        mean_neg_ave_train_log_marginal_likelihood = jnp.array([
            neg_ave_log_marginal_likelihood(dist=q_, x=x_,
                eltwise_log_target=eltwise_log_marg_likelihood_)
            for q_, x_ in zip(batch_q, batch_x_train)
        ]).mean()

        mean_neg_train_log_posterior_pred = jnp.array([
            neg_log_posterior_predictive(dist=q_, x=x_,
                eltwise_log_target=eltwise_log_marg_likelihood_)
            for q_, x_ in zip(batch_q, batch_x_train)
        ]).mean()

        m["neg_train_ave_log_marginal_likelihood_" + key] = mean_neg_ave_train_log_marginal_likelihood.item()
        m["neg_train_log_posterior_pred_" + key] = mean_neg_train_log_posterior_pred.item()

        # average log posterior predictive on held-out data
        mean_neg_ave_test_log_marginal_likelihood = jnp.array([
            neg_ave_log_marginal_likelihood(dist=q_, x=x_ho_,
                eltwise_log_target=eltwise_log_marg_likelihood_)
            for q_, x_ho_ in zip(batch_q, batch_x_ho)
        ]).mean()

        mean_neg_test_log_posterior_pred = jnp.array([
            neg_log_posterior_predictive(dist=q_, x=x_ho_,
                eltwise_log_target=eltwise_log_marg_likelihood_)
            for q_, x_ho_ in zip(batch_q, batch_x_ho)
        ]).mean()

        m["neg_test_ave_log_marginal_likelihood_" + key] = mean_neg_ave_test_log_marginal_likelihood.item()
        m["neg_test_log_posterior_pred_" + key] = mean_neg_test_log_posterior_pred.item()
        m["symlog_neg_test_ave_log_marginal_likelihood_" + key] = symlog(mean_neg_ave_test_log_marginal_likelihood).item()
        m["symlog_neg_test_log_posterior_pred_" + key] = symlog(mean_neg_test_log_posterior_pred).item()

        # log interventional posterior predictive
        mean_neg_ave_interv_log_marginal_likelihood = 0.0
        mean_neg_interv_log_posterior_pred = 0.0
        for q, x_interv in zip(batch_q, batch_x_interv):

            for interv_dict_, x_interv_ in x_interv:

                interv_targets_ = jnp.isin(jnp.arange(kwargs.n_vars), jnp.array(list(interv_dict_.keys())))
                eltwise_log_marg_likelihood_interv_ = lambda w_, x_: eltwise_log_marg_likelihood(w_, x_, interv_targets_)

                mean_neg_ave_interv_log_marginal_likelihood += neg_ave_log_marginal_likelihood(
                    dist=q,
                    eltwise_log_target=eltwise_log_marg_likelihood_interv_,
                    x=jnp.array(x_interv_))

                mean_neg_interv_log_posterior_pred += neg_log_posterior_predictive(
                    dist=q,
                    eltwise_log_target=eltwise_log_marg_likelihood_interv_,
                    x=jnp.array(x_interv_))

        mean_neg_ave_interv_log_marginal_likelihood /= (len(x_interv) * len(batch_x_interv))
        mean_neg_interv_log_posterior_pred /= (len(x_interv) * len(batch_x_interv))

        m["neg_interv_ave_log_marginal_likelihood_" + key] = mean_neg_ave_interv_log_marginal_likelihood.item()
        m["neg_interv_log_posterior_pred_" + key] = mean_neg_interv_log_posterior_pred.item()
        m["symlog_neg_interv_ave_log_marginal_likelihood_" + key] = symlog(mean_neg_ave_interv_log_marginal_likelihood).item()
        m["symlog_neg_interv_log_posterior_pred_" + key] = symlog(mean_neg_interv_log_posterior_pred).item()

    # hyperparameters
    m["svgd_steps"] = params["t"]
    m["alpha"] = params['alpha'].item()
    m["beta"] = params['beta'].item()
    m["tau"] = params['tau'].item()
    m["norm_particles"] = jnp.linalg.norm(params['batch_x'].reshape(batch_size, -1), axis=1).mean(0).item()
    m["norm_phi"] = jnp.linalg.norm(params['batch_phi'].reshape(batch_size, -1), axis=1).mean(0).item()

    return m
    

def compute_tune_metrics_joint(*, params, kwargs, svgd, variants_target, eltwise_log_likelihood, batch_g_gt, batch_x_train, batch_x_ho, batch_x_interv):
    """
    Computes tune metrics for joint inference case
    """

    batch_x = params['batch_x']
    batch_phi_x = params['batch_phi_x']
    batch_size = batch_x.shape[0]

    batch_theta = tree_unzip_leading(params['batch_theta'], batch_size)
    batch_phi_theta = tree_unzip_leading(params['batch_phi_theta'], batch_size)
    
    batch_keys = params['batch_keys']
    batch_hard_g = svgd.particle_to_hard_g(batch_x)
    n_variants = len(variants_target)

    batch_empirical = [
        particle_joint_empirical(hard_g, theta)
        for b, (hard_g, theta) in enumerate(zip(batch_hard_g, batch_theta))
    ]

    batch_mixture = [
        particle_joint_mixture(hard_g, theta, lambda ws_, thetas_: svgd.double_eltwise_log_joint_prob_no_batch(ws_, thetas_, b))
        for b, (hard_g, theta) in enumerate(zip(batch_hard_g, batch_theta))
    ]

    '''Graph metrics'''
    batch_empirical_graph = [(q[0], q[2]) for q in batch_empirical]
    batch_mixture_graph = [(q[0], q[2]) for q in batch_mixture]

    m = compute_tune_metrics_graph(params=params, kwargs=kwargs, svgd=svgd, variants_target=variants_target,
        batch_g_gt=batch_g_gt, batch_empirical_graph=batch_empirical_graph, batch_mixture_graph=batch_mixture_graph, use_cpdag=False)

    '''Likelihood based metrics'''
    no_interv_targets = jnp.zeros(kwargs.n_vars).astype(bool)

    for key, batch_q in [('hard', batch_empirical), ('mixture', batch_mixture)]:

        eltwise_log_joint_target_ = lambda w_, theta_, x_: eltwise_log_likelihood(w_, theta_, x_, no_interv_targets)

        # average log posterior predictive on train data
        mean_neg_ave_train_likelihood = jnp.array([
            neg_ave_log_likelihood(dist=q_, x=x_,
                eltwise_log_joint_target=eltwise_log_joint_target_)
            for q_, x_ in zip(batch_q, batch_x_train)
        ]).mean()

        mean_neg_train_log_posterior_pred = jnp.array([
            neg_log_joint_posterior_predictive(dist=q_, x=x_,
                eltwise_log_joint_target=eltwise_log_joint_target_)
            for q_, x_ in zip(batch_q, batch_x_train)
        ]).mean()

        m["neg_train_ave_log_likelihood_" + key] = mean_neg_ave_train_likelihood.item()
        m["neg_train_log_posterior_pred_" + key] = mean_neg_train_log_posterior_pred.item()

        # average log posterior predictive on held-out data
        mean_neg_ave_test_likelihood = jnp.array([
            neg_ave_log_likelihood(dist=q_, x=x_ho_,
                eltwise_log_joint_target=eltwise_log_joint_target_)
            for q_, x_ho_ in zip(batch_q, batch_x_ho)
        ]).mean()

        mean_neg_test_log_posterior_pred = jnp.array([
            neg_log_joint_posterior_predictive(dist=q_, x=x_ho_,
                eltwise_log_joint_target=eltwise_log_joint_target_)
            for q_, x_ho_ in zip(batch_q, batch_x_ho)
        ]).mean()

        m["neg_test_ave_log_likelihood_" + key] = mean_neg_ave_test_likelihood.item()
        m["neg_test_log_posterior_pred_" + key] = mean_neg_test_log_posterior_pred.item()
        m["symlog_neg_test_ave_log_likelihood_" + key] = symlog(mean_neg_ave_test_likelihood).item()
        m["symlog_neg_test_log_posterior_pred_" + key] = symlog(mean_neg_test_log_posterior_pred).item()

        # log interventional posterior predictive
        mean_neg_ave_interv_likelihood = 0.0
        mean_neg_interv_log_posterior_pred = 0.0
        for q, x_interv in zip(batch_q, batch_x_interv):

            for interv_dict_, x_interv_ in x_interv:

                interv_targets_ = jnp.isin(jnp.arange(kwargs.n_vars), jnp.array(list(interv_dict_.keys())))
                eltwise_log_likelihood_interv_ = lambda w_, theta_, x_: eltwise_log_likelihood(w_, theta_, x_, interv_targets_)

                mean_neg_ave_interv_likelihood += neg_ave_log_likelihood(
                    dist=q,
                    eltwise_log_joint_target=eltwise_log_likelihood_interv_,
                    x=jnp.array(x_interv_))

                mean_neg_interv_log_posterior_pred += neg_log_joint_posterior_predictive(
                    dist=q,
                    eltwise_log_joint_target=eltwise_log_likelihood_interv_,
                    x=jnp.array(x_interv_))

        mean_neg_ave_interv_likelihood /= (len(x_interv) * len(batch_x_interv))
        mean_neg_interv_log_posterior_pred /= (len(x_interv) * len(batch_x_interv))

        m["neg_interv_ave_log_likelihood_" + key] = mean_neg_ave_interv_likelihood.item()
        m["neg_interv_log_posterior_pred_" + key] = mean_neg_interv_log_posterior_pred.item()
        m["symlog_neg_interv_ave_log_likelihood_" + key] = symlog(mean_neg_ave_interv_likelihood).item()
        m["symlog_neg_interv_log_posterior_pred_" + key] = symlog(mean_neg_interv_log_posterior_pred).item()

    # norms
    m["norm_particles_x"] =     jnp.array([jnp.linalg.norm(x_) for x_ in batch_x]).mean().item()
    m["norm_particles_theta"] = jnp.array([tree_reduce(jnp.add, tree_map(jnp.linalg.norm, tr_)) for tr_ in batch_theta]).mean().item()
    m["norm_phi_x"] =           jnp.array([jnp.linalg.norm(x_) for x_ in batch_phi_x]).mean().item()
    m["norm_phi_theta"] =       jnp.array([tree_reduce(jnp.add, tree_map(jnp.linalg.norm, tr_)) for tr_ in batch_phi_theta]).mean().item()

    # hyperparameters
    m["svgd_steps"] = params["t"]
    m["alpha"] = params['alpha'].item()
    m["beta"] = params['beta'].item()
    m["tau"] = params['tau'].item()

    return m
    
