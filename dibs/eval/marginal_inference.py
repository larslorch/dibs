import time

import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap, vjp, jvp

from dibs.utils.func import particle_empirical, particle_empirical_mixture, bit2id
from dibs.utils.graph import mat_to_graph

from dibs.kernel.basic import (
    FrobeniusSquaredExponentialKernel,
    AngularSquaredExponentialKernel,
    StructuralHammingSquaredExponentialKernel,
)

from dibs.bootstrap.bootstrap import NonparametricDAGBootstrap
from dibs.bootstrap.learners import GES, PC
from dibs.mcmc.structure import StructureMCMC
from dibs.svgd.dot_graph_svgd import DotProductGraphSVGD

from dibs.eval.result import Distribution

from dibs.eval.target import hparam_dict_to_str

from config.svgd import marginal_config
from config.baselines import struct_mcmc_config


def run_marginal_argmax(*, r, c, key, kwargs, target, additional_descr=None):
    """
    Baseline comprising the ground truth graph as a single particle distribution
    If the ground truth is cyclic (e.g. for the Sachs dataset), 
    the returned metrics are not interpretable/wrong
    """
    g_true_particles = jnp.array(target.g)[jnp.newaxis]
    g_true_dist = particle_empirical(g_true_particles)

    dists = [Distribution(
        passed_key=None,
        descr='gt' + (additional_descr or ''),
        descr_hparams=None,
        joint=False,
        n_particles=1,
        target=target,
        dist=g_true_dist,
        g_particles=bit2id(g_true_particles),
        theta_particles=None,
        walltime=0.0,
        kwargs=kwargs,
        log=None,
        error=None,
    )]

    return dists


def run_marginal_structMCMC(*, r, c, key, n_particles_loop, kwargs, target, ig_log_target, ig_log_target_single, additional_descr=None):
    """
    Single execution of Structure MCMC
    Run once for maximum number of particles, and then truncated for smaller n_particles queries
    """

    dists = []
    n_particles_max = max(n_particles_loop)

    # hyperparams
    config_ = struct_mcmc_config[kwargs.inference_model]
    burnin = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('burnin', kwargs.mcmc_burnin) if not kwargs.smoke_test else 2
    thinning = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('thinning', kwargs.mcmc_thinning) if not kwargs.smoke_test else 2

    structMCMC_init_params = dict(
        n_vars=kwargs.n_vars, 
        only_non_covered=kwargs.mcmc_only_non_covered, 
        verbose=False,
    )
    structMCMC = StructureMCMC(**structMCMC_init_params)

    structMCMC_run_params = dict(
        key=key,
        n_samples=n_particles_max, 
        unnormalized_log_prob_single=ig_log_target_single,
        burnin=burnin,
        thinning=thinning,
    )

    structMCMC_hparams = dict(
        **structMCMC_init_params,
        **structMCMC_run_params,
    )

    # safe execution
    try:
        t_start = time.time()
        structMCMC_samples = structMCMC.sample(
            **structMCMC_run_params,
            verbose_indication=kwargs.verbose if r == 0 and c == 0 else 0,
        )
        t_end = time.time()

        # generate distributions for different n_particles used
        for n_particles in n_particles_loop:
            structMCMC_particles = structMCMC_samples[:n_particles]
            structMCMC_dist = particle_empirical(structMCMC_particles)
            dists.append(Distribution(
                passed_key=key,
                descr='mcmc_structure' + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(structMCMC_hparams),
                joint=False,
                n_particles=n_particles,
                target=target,
                dist=structMCMC_dist,
                g_particles=bit2id(structMCMC_particles),
                theta_particles=None,
                walltime=(t_end - t_start) * n_particles / n_particles_max,
                kwargs=kwargs,
                log=None,
                error=None,
            ))

    # error handling
    except Exception as e:
        print('structMCMC', repr(e))

        for n_particles in n_particles_loop:
            dists.append(Distribution(
                passed_key=key,
                descr='mcmc_structure' + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(structMCMC_hparams),
                joint=False,
                n_particles=n_particles,
                target=target,
                dist=None,
                g_particles=None,
                theta_particles=None,
                walltime=None,
                kwargs=kwargs,
                log=None,
                error=e
            ))

    return dists


def run_marginal_bootstrap(*, r, c, key, n_particles_loop, kwargs, target, ig_log_target, learner_str, no_bootstrap=False, additional_descr=None):
    """
    Single execution of DAG bootstrap with learner `learner`
    Run once for maximum number of particles, and then truncated for smaller n_particles queries
    """
    dists = []

    # if 1 particle is evaluated, do not bootstrap but use all the data
    # run this in advance
    if jnp.isin(n_particles_loop, 1).any() and not no_bootstrap:
        dists += run_marginal_bootstrap(r=r, c=c, key=key,
            n_particles_loop=jnp.array([1]), kwargs=kwargs, 
            target=target, ig_log_target=ig_log_target, 
            learner_str=learner_str, no_bootstrap=True, additional_descr=additional_descr)

        n_particles_loop = n_particles_loop[jnp.where(n_particles_loop != 1)]

    # run with max number of particles and extract other results
    n_particles_max = max(n_particles_loop)

    if learner_str == 'boot_ges':
        boot = NonparametricDAGBootstrap(
            learner=GES(), 
            verbose=False, 
            n_restarts=kwargs.bootstrap_n_error_restarts,
            no_bootstrap=no_bootstrap,
        )

        boot_hparams = dict(
            bootstrap_n_error_restarts=kwargs.bootstrap_n_error_restarts,
        )

    elif learner_str == 'boot_pc':
        boot = NonparametricDAGBootstrap(
            learner=PC(
                ci_test=kwargs.bootstrap_pc_ci_test,
                ci_alpha=kwargs.bootstrap_pc_ci_alpha,
            ), 
            verbose=False, 
            n_restarts=kwargs.bootstrap_n_error_restarts,
            no_bootstrap=no_bootstrap,
        )

        boot_hparams = dict(
            bootstrap_n_error_restarts=kwargs.bootstrap_n_error_restarts,
            ci_test=kwargs.bootstrap_pc_ci_test,
            ci_alpha=kwargs.bootstrap_pc_ci_alpha,
        )

    else:
        raise ValueError('Invalid learner string.')
    
    # safe execution
    try:
        t_start = time.time()        
        boot_samples = boot.sample_particles(
            key=key, n_samples=n_particles_max, x=jnp.array(target.x),
            verbose_indication=kwargs.verbose if r == 0 and c == 0 else 0,
        )
        t_end = time.time()

        def eltwise_log_target(g_array):
            # [N, d, d] -> [N, ]
            return jnp.array([ig_log_target(mat_to_graph(g)) for g in g_array])

        # generate distributions for different n_particles used
        for n_particles in n_particles_loop:
            
            boot_dist = particle_empirical_mixture(
                boot_samples[:n_particles], eltwise_log_target)
            
            dists.append(Distribution(
                passed_key=key,
                descr=learner_str + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(boot_hparams),
                joint=False,
                n_particles=n_particles,
                target=target,
                dist=boot_dist,
                g_particles=bit2id(boot_samples[:n_particles]),
                theta_particles=None,
                walltime=(t_end - t_start) * n_particles / n_particles_max,
                kwargs=kwargs,
                log=None,
                error=None,
            ))

    # error handling
    except Exception as e:
        print(learner_str, repr(e))

        for n_particles in n_particles_loop:
            dists.append(Distribution(
                passed_key=key,
                descr=learner_str + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(boot_hparams),
                joint=False,
                n_particles=n_particles,
                target=target,
                dist=None,
                g_particles=None,
                theta_particles=None,
                walltime=None,
                kwargs=kwargs,
                log=None,
                error=e,
            ))
    
    return dists


def run_marginal_svgd(*, r, c, key, n_particles, kwargs, target, log_prior, log_target, additional_descr=None):

    """
    Single execution of SVGD with `n_particles` particles
    """

    dists = []
    svgd_key = key.copy()

    # warning of other config if not available is issued at the beginning of `eval.py`
    config_ = marginal_config[kwargs.inference_model]
    kwargs_svgd = config_.get(kwargs.n_vars, config_[min(config_.keys())])
    
    # initialize
    if kwargs.dibs_n_steps:
        kwargs_svgd['n_steps'] = kwargs.dibs_n_steps
    n_steps = kwargs_svgd.get('n_steps', 3000) if not kwargs.smoke_test else 2

    optimizer = dict(name='rmsprop', stepsize=kwargs_svgd.get('opt_stepsize', kwargs.dibs_opt_stepsize))

    if kwargs.dibs_latent_dim:
        kwargs_svgd['latent_dim'] = kwargs.dibs_latent_dim
    n_dim = kwargs_svgd.get('latent_dim', int(kwargs.n_vars / 2))

    if kwargs_svgd.get('graph_embedding_representation', kwargs.dibs_graph_embedding_representation):
        latent_prior_std = (1.0 / jnp.sqrt(n_dim)).item()
    else:
        latent_prior_std = 1.0

    # assuming dot product representation of graph
    key, subk = random.split(key)
    if kwargs_svgd.get('graph_embedding_representation', kwargs.dibs_graph_embedding_representation):
        init_particles = random.normal(subk, shape=(n_particles, kwargs.n_vars, n_dim, 2)) * latent_prior_std * kwargs.dibs_rel_init_scale
    else:
        init_particles = random.normal(subk, shape=(n_particles, kwargs.n_vars, kwargs.n_vars)) * latent_prior_std * kwargs.dibs_rel_init_scale

    kernel_hparams = dict(
        h=kwargs_svgd['h'],
        scale=kwargs_svgd.get('kernel_scale', 1.0),
        graph_embedding_representation=kwargs_svgd.get('graph_embedding_representation', kwargs.dibs_graph_embedding_representation),
    )
    kernel = FrobeniusSquaredExponentialKernel(**kernel_hparams)

    # temperature hyperparameters
    def linear_alpha(t):
        return jnp.minimum((kwargs_svgd.get('alpha_linear', kwargs.alpha_linear) * t) + kwargs_svgd.get('alpha', 0.0), kwargs.ceil_alpha)

    def linear_beta(t):
        return jnp.minimum((kwargs_svgd.get('beta_linear', kwargs.beta_linear) * t) + kwargs_svgd.get('beta', 0.0), kwargs.ceil_beta)

    def linear_tau(t):
        return jnp.minimum((kwargs_svgd.get('tau_linear', kwargs.tau_linear) * t) + kwargs_svgd.get('tau', 0.0), kwargs.ceil_tau)

    def exponential_alpha(t):
        return jnp.minimum(jnp.exp(kwargs_svgd.get('alpha_expo', kwargs.alpha_expo) * t) * kwargs_svgd.get('alpha', 1.0), kwargs.ceil_alpha)

    def exponential_beta(t):
        return jnp.minimum(jnp.exp(kwargs_svgd.get('beta_expo', kwargs.beta_expo) * t) * kwargs_svgd.get('beta', 1.0), kwargs.ceil_beta)

    def exponential_tau(t):
        return jnp.minimum(jnp.exp(kwargs_svgd.get('tau_expo', kwargs.tau_expo) * t) * kwargs_svgd.get('tau', 1.0), kwargs.ceil_tau)

    if 'alpha_linear' in kwargs_svgd.keys():
        alpha_sched = linear_alpha
    elif 'alpha_expo' in kwargs_svgd.keys():
        alpha_sched = exponential_alpha
    else:
        alpha_sched = lambda _: jnp.array([kwargs_svgd.get('alpha_const', 1.0)])

    if 'beta_linear' in kwargs_svgd.keys():
        beta_sched = linear_beta
    elif 'beta_expo' in kwargs_svgd.keys():
        beta_sched = exponential_beta
    else:
        beta_sched = lambda _: jnp.array([kwargs_svgd.get('beta_const', 1.0)])

    if 'tau_linear' in kwargs_svgd.keys():
        tau_sched = linear_tau
    elif 'tau_expo' in kwargs_svgd.keys():
        tau_sched = exponential_tau
    else:
        tau_sched = lambda _: jnp.array([kwargs_svgd.get('tau_const', 1.0)])

    gamma_sched = lambda _: jnp.array([kwargs_svgd.get('gamma_const', 1.0)])


    # svgd
    dibs_init_params = dict(
        n_vars=kwargs.n_vars,
        n_dim=n_dim,
        optimizer=optimizer,
        kernel=kernel,
        target_log_prior=log_prior,
        target_log_prob=log_target,
        alpha=alpha_sched,
        beta=beta_sched,
        gamma=gamma_sched,
        tau=tau_sched,
        clip=None,
        fix_rotation=kwargs_svgd.get("fix_rotation", kwargs.dibs_fix_rotation),
        grad_estimator='score',
        repulsion_in_prob_space=False,
        latent_prior_std=latent_prior_std,
        graph_embedding_representation=kwargs_svgd.get('graph_embedding_representation', kwargs.dibs_graph_embedding_representation),
        verbose=False
    )
    svgd = DotProductGraphSVGD(
        **dibs_init_params, 
        n_grad_mc_samples=kwargs_svgd.get('n_grad_mc_samples', kwargs.dibs_n_grad_mc_samples),
        n_acyclicity_mc_samples=kwargs_svgd.get('n_acyclicity_mc_samples', kwargs.dibs_n_acyclicity_mc_samples),
        score_function_baseline=kwargs_svgd.get('score_function_baseline', kwargs.dibs_score_function_baseline),
        constraint_prior_graph_sampling=kwargs_svgd.get('constraint_prior_graph_sampling', kwargs.dibs_constraint_prior_graph_sampling),
    )

    dibs_run_params = dict(
        key=key,
    )

    # all hparams
    svgd_hparams = dict(
        **kwargs_svgd,
        rel_init_scale=kwargs.dibs_rel_init_scale,
        **dibs_init_params, 
        **dibs_run_params,
    )
    
    # safe execution
    try:
        # evaluate
        t_start = time.time()        
        svgd_particles = svgd.sample_particles(
            **dibs_run_params,
            init_particles=init_particles.copy(),
            verbose_indication=kwargs.verbose if r == 0 and c == 0 else 0,
            n_steps=n_steps,
        )
        t_end = time.time()

        # posterior
        svgd_hard_g = svgd.particle_to_hard_g(svgd_particles)
        svgd_empirical = particle_empirical(svgd_hard_g)
        svgd_mixture = particle_empirical_mixture(svgd_hard_g, svgd.eltwise_log_prob)

        dists.append(Distribution(
            passed_key=svgd_key,
            descr='dibs_empirical' + (additional_descr or ''),
            descr_hparams=hparam_dict_to_str(svgd_hparams),
            joint=False,
            n_particles=n_particles,
            target=target,
            dist=svgd_empirical,
            g_particles=bit2id(svgd_hard_g),
            theta_particles=None,
            walltime=t_end - t_start,
            kwargs=kwargs,
            log=None,
            error=None,
        ))
        dists.append(Distribution(
            passed_key=svgd_key,
            descr='dibs_mixture' + (additional_descr or ''),
            descr_hparams=hparam_dict_to_str(svgd_hparams),
            joint=False,
            n_particles=n_particles,
            target=target,
            dist=svgd_mixture,
            g_particles=bit2id(svgd_hard_g),
            theta_particles=None,
            walltime=t_end - t_start,
            kwargs=kwargs,
            log=None,
            error=None,
        ))
    
    # error handling
    except Exception as e:
        print(f'SVGD n_particles={n_particles}', repr(e))
        svgd_error_dist = dict(
            passed_key=svgd_key,
            descr_hparams=hparam_dict_to_str(svgd_hparams),
            joint=False,
            n_particles=n_particles,
            target=target,
            dist=None,
            g_particles=None,
            theta_particles=None,
            walltime=None,
            kwargs=kwargs,
            error=e,
        )

        dists.append(Distribution(descr='dibs_empirical' + (additional_descr or ''), **svgd_error_dist))
        dists.append(Distribution(descr='dibs_mixture' + (additional_descr or ''), **svgd_error_dist))

    return dists

