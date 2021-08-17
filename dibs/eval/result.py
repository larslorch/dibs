import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")

from collections import namedtuple, defaultdict
import pandas as pd

import jax.numpy as jnp
from jax import grad, jit, vmap, vjp, jvp
from jax import random
from jax.scipy.special import logsumexp

from dibs.eval.metrics import (
    l1_edge_belief, 
    expected_shd,
    expected_edges,
    threshold_metrics,
    neg_ave_log_marginal_likelihood, 
    neg_ave_log_likelihood, 
    neg_log_posterior_predictive, 
    neg_log_joint_posterior_predictive,
    best_train_neg_log_marginal_likelihood,
    best_train_neg_log_likelihood
)
from dibs.utils.graph import elwise_acyclic_constr_nograd
from dibs.utils.func import dist_is_none, id2bit

MMD_GRID_SEARCH = [('01', 0.1), ('1', 1.0), ('10', 10.0)]


"""
Objects to standardize evaluation results
"""

Distribution = namedtuple('Distribution', (
    'passed_key',
    'descr',            # description
    'descr_hparams',    # description of hyperparameters
    'joint',            # boolean
    'n_particles',
    'target',
    'dist',             # (ids, thetas, log_probs) if joint else (ids, log_probs)
    'g_particles',      # [n_particles, n_vars, n_vars]
    'theta_particles',  # [n_particles, ...]
    'walltime',
    'kwargs',           # eval parser kwargs
    'log',              # additional info such as acceptance rate
    'error',
))

# defines row of the result dataframe
# info
fields_distributionEvaluation_info = (
    'descr',
    'descr_hparams',
    'n_particles',
    'target_filename',
    'graph_dist',
    'n_vars',
    'n_observations',
    'n_ho_observations',
    'passed_key',
    'c',  # random instantiation of test target
    'r',  # random rollout of method      
)

# metrics
fields_distributionEvaluation_metrics = (
    'kl_q_p',
    'kl_p_q',
    'mmd_shd_01_sample',
    'mmd_shd_1_sample',
    'mmd_shd_10_sample',
    'l1_edge_belief',
    'expected_shd',
    'expected_shd_cpdag',
    'expected_edges',
    'best_train_neg_log_marginal_likelihood',
    'best_train_neg_log_likelihood',
    'neg_ave_train_marginal_likelihood',
    'neg_ave_train_likelihood',
    'neg_train_log_posterior_predictive',
    'neg_ave_test_log_marginal_likelihood',
    'neg_ave_test_log_likelihood',
    'neg_test_log_posterior_predictive',
    'neg_ave_interv_log_marginal_likelihood',
    'neg_ave_interv_log_likelihood',
    'neg_interv_log_posterior_predictive',
    'n_unique_graphs',
    'n_cyclic_graphs',
    'walltime',
    'fpr',
    'tpr',
    'roc_auc',
    'precision',
    'recall',
    'prc_auc',
    'ave_prec',
)

# debugging
fields_distributionEvaluation_debug = (
    'log',
    'status',
    'error',
)

fields_distributionEvaluation = (
    fields_distributionEvaluation_info +
    fields_distributionEvaluation_metrics + 
    fields_distributionEvaluation_debug 
)

DistributionEvaluation = namedtuple(
    'DistributionEvaluation',
    fields_distributionEvaluation,
    defaults=[None for _ in fields_distributionEvaluation]
)



def eval_marginal_dist_metrics(q, dist_kwargs, args):
    """
    Computes (normalized) distribution metrics based on `args`
        q:      [n_unique, 2]
        args:   dict containing information used to evaluate (such as e.g. ground truth graph)
    """

    if dist_is_none(q):
        return dict()

    # double check that distribution is normalized
    if not jnp.isclose(logsumexp(q[1]), 0.0, atol=1e-2):
        # atol kept relatively big because error > 1e-4 can occur even for x64 precision
        print('q.shape', q.shape)
        print('logsumexp(q[1])', logsumexp(q[1]))        
        print('q[1]', q[1])
        raise ValueError('Distribution not normalized.')

    # graph metrics
    edge_belief = l1_edge_belief(dist=q, g=jnp.array(args["target"].g))
    shd = expected_shd(dist=q, g=jnp.array(args["target"].g), use_cpdag=False)
    shd_cpdag = expected_shd(dist=q, g=jnp.array(args["target"].g), use_cpdag=True)
    edges = expected_edges(dist=q, g=jnp.array(args["target"].g))
    threshold_metric_dict = threshold_metrics(dist=q, g=jnp.array(args["target"].g),
        undirected_cpdag_oriented_correctly='boot' in dist_kwargs['descr'])

    #### training observational data
    no_interv_targets = jnp.zeros(args["target"].n_vars).astype(bool)
    eltwise_log_marg_likelihood_ = lambda w_, x_: args["eltwise_log_marg_likelihood"](w_, x_, no_interv_targets)

    # best training marginal likelihood achieved with any graph
    best_train_neg_log_lik = best_train_neg_log_marginal_likelihood(
        dist=q,
        eltwise_log_target=eltwise_log_marg_likelihood_,
        x=jnp.array(args["target"].x)).item()    

    # ave log marginal likelihood
    neg_ave_train_marginal_likelihood = neg_ave_log_marginal_likelihood(
        dist=q,
        eltwise_log_target=eltwise_log_marg_likelihood_,
        x=jnp.array(args["target"].x)).item()

    # log posterior predictive 
    neg_train_log_posterior_pred = neg_log_posterior_predictive(
        dist=q,
        eltwise_log_target=eltwise_log_marg_likelihood_,
        x=jnp.array(args["target"].x)).item()    

    #### held-out observational data
    # ave log marginal likelihood
    neg_ave_test_log_marginal_likelihood = neg_ave_log_marginal_likelihood(
        dist=q,
        eltwise_log_target=eltwise_log_marg_likelihood_,
        x=jnp.array(args["target"].x_ho)).item() if args["target"].x_ho is not None else None

    # log posterior predictive
    neg_test_log_posterior_pred = neg_log_posterior_predictive(
        dist=q,
        eltwise_log_target=eltwise_log_marg_likelihood_,
        x=jnp.array(args["target"].x_ho)).item() if args["target"].x_ho is not None else None

    #### held-out interventional data
    # log interventional posterior predictive
    if args["target"].x_interv is not None:
        neg_ave_interv_log_marginal_likelihood = 0.0
        neg_interv_log_posterior_pred = 0.0
        for interv_dict_, x_interv_ in args["target"].x_interv:
            interv_targets_ = jnp.isin(jnp.arange(args["target"].n_vars), jnp.array(list(interv_dict_.keys())))
            eltwise_log_marg_likelihood_interv_ = lambda w_, x_: args["eltwise_log_marg_likelihood"](w_, x_, interv_targets_)

            neg_ave_interv_log_marginal_likelihood += neg_ave_log_marginal_likelihood(
                dist=q,
                eltwise_log_target=eltwise_log_marg_likelihood_interv_,
                x=jnp.array(x_interv_)).item()

            neg_interv_log_posterior_pred += neg_log_posterior_predictive(
                dist=q,
                eltwise_log_target=eltwise_log_marg_likelihood_interv_,
                x=jnp.array(x_interv_)).item()

        neg_ave_interv_log_marginal_likelihood /= len(args["target"].x_interv)
        neg_interv_log_posterior_pred /= len(args["target"].x_interv)
    else:
        neg_ave_interv_log_marginal_likelihood = None
        neg_interv_log_posterior_pred = None

    # count number of unique graphs
    n_unique_graphs = q[0].shape[0]

    # wrap-up
    return dict(
        l1_edge_belief=edge_belief,
        expected_shd=shd,
        expected_shd_cpdag=shd_cpdag,
        expected_edges=edges,
        best_train_neg_log_marginal_likelihood=best_train_neg_log_lik,
        neg_ave_train_marginal_likelihood=neg_ave_train_marginal_likelihood,
        neg_train_log_posterior_predictive=neg_train_log_posterior_pred,
        neg_ave_test_log_marginal_likelihood=neg_ave_test_log_marginal_likelihood,
        neg_test_log_posterior_predictive=neg_test_log_posterior_pred,
        neg_ave_interv_log_marginal_likelihood=neg_ave_interv_log_marginal_likelihood,
        neg_interv_log_posterior_predictive=neg_interv_log_posterior_pred,
        n_unique_graphs=n_unique_graphs,
        fpr=threshold_metric_dict.get('fpr', None),
        tpr=threshold_metric_dict.get('tpr', None),
        roc_auc=threshold_metric_dict.get('roc_auc', None),
        precision=threshold_metric_dict.get('precision', None),
        recall=threshold_metric_dict.get('recall', None),
        prc_auc=threshold_metric_dict.get('prc_auc', None),
        ave_prec=threshold_metric_dict.get('ave_prec', None),
    )


def eval_joint_dist_metrics(q, dist_kwargs, args):
    """
    Computes (normalized) distribution metrics based on `args`
        q:      [n_unique, 2]
        args:   dict containing information used to evaluate (such as e.g. ground truth graph)
    """

    if dist_is_none(q):
        return dict()

    # double check that distribution is normalized
    if not jnp.isclose(logsumexp(q[2]), 0.0, atol=1e-2):
        # atol kept relatively big because error > 1e-4 can occur even for x64 precision
        print('q.shape', q.shape)
        print('logsumexp(q[2])', logsumexp(q[2]))        
        print('q[2]', q[2])
        raise ValueError('Distribution not normalized.')

    # graph metrics
    q_graph = (q[0], q[2])
    edge_belief = l1_edge_belief(dist=q_graph, g=jnp.array(args["target"].g))
    shd = expected_shd(dist=q_graph, g=jnp.array(args["target"].g), use_cpdag=False)
    shd_cpdag = expected_shd(dist=q_graph, g=jnp.array(args["target"].g), use_cpdag=True)
    edges = expected_edges(dist=q_graph, g=jnp.array(args["target"].g))
    threshold_metric_dict = threshold_metrics(dist=q_graph, g=jnp.array(args["target"].g),
        undirected_cpdag_oriented_correctly='boot' in dist_kwargs['descr'])

    #### training observational data
    no_interv_targets = jnp.zeros(args["target"].n_vars).astype(bool)
    eltwise_log_likelihood_ = lambda w_, theta_, x_: args["eltwise_log_likelihood"](w_, theta_, x_, no_interv_targets)

    # best training marginal likelihood achieved with any graph
    best_train_neg_log_lik = best_train_neg_log_likelihood(
        dist=q,
        eltwise_log_joint_target=eltwise_log_likelihood_,
        x=jnp.array(args["target"].x)).item()    

    # ave log marginal likelihood
    neg_ave_train_likelihood = neg_ave_log_likelihood(
        dist=q,
        eltwise_log_joint_target=eltwise_log_likelihood_,
        x=jnp.array(args["target"].x)).item()

    # log posterior predictive 
    neg_train_log_posterior_pred = neg_log_joint_posterior_predictive(
        dist=q,
        eltwise_log_joint_target=eltwise_log_likelihood_,
        x=jnp.array(args["target"].x)).item()    

    #### held-out observational data

    # ave log likelihood
    neg_ave_test_log_likelihood = neg_ave_log_likelihood(
        dist=q,
        eltwise_log_joint_target=eltwise_log_likelihood_,
        x=jnp.array(args["target"].x_ho)).item() if args["target"].x_ho is not None else None

    # log posterior predictive
    neg_test_log_posterior_pred = neg_log_joint_posterior_predictive(
        dist=q,
        eltwise_log_joint_target=eltwise_log_likelihood_,
        x=jnp.array(args["target"].x_ho)).item() if args["target"].x_ho is not None else None

    #### held-out interventional data
    # log interventional posterior predictive
    if args["target"].x_interv is not None:
        neg_ave_interv_log_likelihood = 0.0
        neg_interv_log_posterior_pred = 0.0
        for interv_dict_, x_interv_ in args["target"].x_interv:
            interv_targets_ = jnp.isin(jnp.arange(args["target"].n_vars), jnp.array(list(interv_dict_.keys())))
            eltwise_log_likelihood_interv_ = lambda w_, theta_, x_: args["eltwise_log_likelihood"](w_, theta_, x_, interv_targets_)

            neg_ave_interv_log_likelihood += neg_ave_log_likelihood(
                dist=q,
                eltwise_log_joint_target=eltwise_log_likelihood_interv_,
                x=jnp.array(x_interv_)).item()

            neg_interv_log_posterior_pred += neg_log_joint_posterior_predictive(
                dist=q,
                eltwise_log_joint_target=eltwise_log_likelihood_interv_,
                x=jnp.array(x_interv_)).item()

        neg_ave_interv_log_likelihood /= len(args["target"].x_interv)
        neg_interv_log_posterior_pred /= len(args["target"].x_interv)
    else:
        neg_ave_interv_log_likelihood = None
        neg_interv_log_posterior_pred = None


    # count number of unique graphs
    n_unique_graphs = q[0].shape[0]

    # wrap-up
    return dict(
        l1_edge_belief=edge_belief,
        expected_shd=shd,
        expected_shd_cpdag=shd_cpdag,
        expected_edges=edges,
        best_train_neg_log_likelihood=best_train_neg_log_lik,
        neg_ave_train_likelihood=neg_ave_train_likelihood,
        neg_train_log_posterior_predictive=neg_train_log_posterior_pred,
        neg_ave_test_log_likelihood=neg_ave_test_log_likelihood,
        neg_test_log_posterior_predictive=neg_test_log_posterior_pred,
        neg_ave_interv_log_likelihood=neg_ave_interv_log_likelihood,
        neg_interv_log_posterior_predictive=neg_interv_log_posterior_pred,
        n_unique_graphs=n_unique_graphs,
        fpr=threshold_metric_dict.get('fpr', None),
        tpr=threshold_metric_dict.get('tpr', None),
        roc_auc=threshold_metric_dict.get('roc_auc', None),
        precision=threshold_metric_dict.get('precision', None),
        recall=threshold_metric_dict.get('recall', None),
        prc_auc=threshold_metric_dict.get('prc_auc', None),
        ave_prec=threshold_metric_dict.get('ave_prec', None),
    )



def eval_marginal_particle_metrics(g_particles, dist_kwargs, args):
    """
    Computes (normalized) distribution metrics based on `args`
        g_particles:    [n_particles, n_vars, n_vars]
        args:           dict containing information used to evaluate (such as e.g. ground truth graph)
    """

    if g_particles is None:
        return dict()

    # metrics when exhaustive ground truth is available
    exhaustive_metrics = dict()
    if not dist_is_none(args["target"].log_posterior):

        '''
        To ensure comparability bewteen all methods, the distribution-formulation
        of MMD is used for evaluation. See eval_dist_metrics()
        '''
        pass

    # number of cyclic graphs (duplicates counted individually to keep number of graphs const.)
    graphs = id2bit(g_particles, args["target"].n_vars)
    is_dag = elwise_acyclic_constr_nograd(graphs, args["target"].n_vars) == 0
    n_cyclic_graphs = graphs.shape[0] - is_dag.sum()

    # wrap-up
    return dict(
        n_cyclic_graphs=n_cyclic_graphs,
        **exhaustive_metrics
    )


def eval_joint_particle_metrics(g_particles, theta_particles, dist_kwargs, args):
    """
    Computes (normalized) distribution metrics based on `args`
        g_particles:        [n_particles, n_vars, n_vars]
        theta_particles:    [n_particles, ...]
        args:               dict containing information used to evaluate (such as e.g. ground truth graph)
    """

    if g_particles is None and theta_particles is None:
        return dict()

    # metrics when exhaustive ground truth is available
    exhaustive_metrics = dict()
    if not dist_is_none(args["target"].log_posterior):

        '''
        To ensure comparability bewteen all methods, the distribution-formulation
        of MMD is used for evaluation. See eval_dist_metrics()
        '''
        pass

    # number of cyclic graphs (duplicates counted individually to keep number of graphs const.)
    graphs = id2bit(g_particles, args["target"].n_vars)
    is_dag = elwise_acyclic_constr_nograd(graphs, args["target"].n_vars) == 0
    n_cyclic_graphs = graphs.shape[0] - is_dag.sum()

    # wrap-up
    return dict(
        n_cyclic_graphs=n_cyclic_graphs,
        **exhaustive_metrics
    )


def eval_info_metrics(dist, args):
    """
    Computes information metrics based on `dist`
    dist:   Distribution object
    args:   dict containing information used to evaluate (such as e.g. ground truth graph)
    """

    info_metrics = dict(
        walltime=dist.walltime,
        log=dist.log,
    )

    # wrap-up
    return info_metrics

def eval(dists, args, joint):
    """
    Evaluatuates dict of list of `dists` against several metrics
        dists:  list of `Distribution` objects
        args:   dict containing information used to evaluate (such as e.g. rollout or ground truth graph)
        joint:  boolean 
    """

    df = pd.DataFrame(columns=DistributionEvaluation._fields)

    # evaluate all distributions
    for dist in dists:
        
        # collect info
        dist_kwargs = dict(
            passed_key=dist.passed_key,
            descr=dist.descr,
            descr_hparams=dist.descr_hparams,
            n_particles=dist.n_particles,
            target_filename=dist.target.filename,
            graph_dist=args["graph_prior_str"],
            n_vars=dist.target.n_vars,
            n_observations=dist.target.n_observations,
            n_ho_observations=dist.target.n_ho_observations,
            c=dist.target.c,
            r=args["rollout"],
        )

        # compute evalution metrics
        if dist.error is None:

            # distribution-based metrics
            if joint:
                dist_eval = eval_joint_dist_metrics(dist.dist, dist_kwargs, args)
            else:
                dist_eval = eval_marginal_dist_metrics(dist.dist, dist_kwargs, args)

            # particle-based metrics
            if joint:
                particle_eval = eval_joint_particle_metrics(dist.g_particles, dist.theta_particles, dist_kwargs, args)
            else:
                particle_eval = eval_marginal_particle_metrics(dist.g_particles, dist_kwargs, args)

            # info metrics 
            info_eval = eval_info_metrics(dist, args)

            eval_kwargs = dict(
                # metrics
                **dist_eval,
                **particle_eval,
                **info_eval,

                # debugging
                status=0,
                error=dist.error,
            )
               
        # handle exceptions and errors
        else:
            eval_kwargs = dict(
                # debugging
                status=1,
                error=repr(dist.error),
            )

        # generate result and append to dataframe
        res = DistributionEvaluation(**dist_kwargs, **eval_kwargs)
        df = df.append(res._asdict(), ignore_index=True)

    return df
