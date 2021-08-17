import time

import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap, vjp, jvp

from dibs.utils.func import particle_joint_empirical, particle_joint_mixture, bit2id
from dibs.utils.graph import mat_to_graph
from dibs.utils.tree import tree_unzip_leading

from dibs.kernel.joint import (
    JointAdditiveFrobeniusSEKernel,
    JointMultiplicativeFrobeniusSEKernel,
)

from dibs.mcmc.joint_structure import MHJointStructureMCMC, GibbsJointStructureMCMC
from dibs.svgd.joint_dot_graph_svgd import JointDotProductGraphSVGD

from dibs.bootstrap.bootstrap import NonparametricDAGBootstrap
from dibs.bootstrap.learners import GES, PC

from dibs.eval.result import Distribution

from dibs.eval.target import hparam_dict_to_str

from config.svgd import joint_config
from config.baselines import mh_mcmc_config, gibbs_mcmc_config

from dibs.utils.tree import tree_expand_leading_by, tree_unzip_leading, tree_zip_leading


def run_joint_argmax(*, r, c, key, kwargs, target, additional_descr=None):
    """
    Baseline comprising the ground truth graph as a single particle distribution
    """
    g_true = tree_expand_leading_by(target.g, 1)
    theta_true = tree_expand_leading_by(target.theta, 1)

    dists = [Distribution(
        passed_key=None,
        descr='gt' + (additional_descr or ''),
        descr_hparams=None,
        joint=True,
        n_particles=1,
        target=target,
        dist=particle_joint_empirical(g_true, theta_true),
        g_particles=bit2id(g_true),
        theta_particles=theta_true,
        walltime=0.0,
        kwargs=kwargs,
        log=None,
        error=None,
    )]

    return dists


def run_joint_structMCMC(*, r, c, key, n_particles_loop, kwargs, target, model, ig_log_joint_target, option_str, additional_descr=None):
    """
    Single execution of Metropolis Joint Structure MCMC
    Run once for maximum number of particles, and then truncated for smaller n_particles queries
    """
    dists = []
    n_particles_max = max(n_particles_loop)

    # hyperparams
    if option_str == 'mh_joint_mcmc_structure':
        config_ = mh_mcmc_config[kwargs.joint_inference_model]
        theta_prop_sig = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('theta_prop_sig', kwargs.mh_joint_mcmc_scale)
        burnin = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('burnin', kwargs.mh_joint_mcmc_burnin) if not kwargs.smoke_test else 2
        thinning = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('thinning', kwargs.mh_joint_mcmc_thinning) if not kwargs.smoke_test else 2
    
    elif option_str == 'gibbs_joint_mcmc_structure':
        config_ = gibbs_mcmc_config[kwargs.joint_inference_model]
        theta_prop_sig = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('theta_prop_sig', kwargs.gibbs_joint_mcmc_scale)
        burnin = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('burnin', kwargs.gibbs_joint_mcmc_burnin) if not kwargs.smoke_test else 2
        thinning = config_.get(kwargs.n_vars, config_[min(config_.keys())]).get('thinning', kwargs.gibbs_joint_mcmc_thinning) if not kwargs.smoke_test else 2

    else:
        raise KeyError('Invalid `option_str` in `run_joint_structMCMC`')
    

    structMCMC_init_params = dict(
        n_vars=kwargs.n_vars, 
        only_non_covered=kwargs.mcmc_only_non_covered, 
        theta_prop_sig=theta_prop_sig,
        verbose=False,
    )
    if option_str == 'mh_joint_mcmc_structure':
        structMCMC = MHJointStructureMCMC(**structMCMC_init_params)
        
    elif option_str == 'gibbs_joint_mcmc_structure':
        structMCMC = GibbsJointStructureMCMC(**structMCMC_init_params)
    
    else:
        raise KeyError('Invalid `option_str` in `run_joint_structMCMC`')

    structMCMC_run_params = dict(
        key=key,
        n_samples=n_particles_max, 
        theta_shape=model.get_theta_shape(n_vars=kwargs.n_vars),
        log_joint_target=ig_log_joint_target,
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
        g_samples_all, theta_samples_all = structMCMC.sample(
            **structMCMC_run_params,
            verbose_indication=kwargs.verbose if r == 0 and c == 0 else 0,
        )
        ave_acceptance_rate = structMCMC.ave_alpha_after_burnin
        t_end = time.time()

        # generate distributions for different n_particles used
        for n_particles in n_particles_loop:
            g_samples = g_samples_all[:n_particles]
            theta_samples = tree_zip_leading(tree_unzip_leading(theta_samples_all, max(n_particles_loop))[:n_particles])
            structMCMC_dist = particle_joint_empirical(g_samples, theta_samples)
            dists.append(Distribution(
                passed_key=key,
                descr=option_str + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(structMCMC_hparams),
                joint=True,
                n_particles=n_particles,
                target=target,
                dist=structMCMC_dist,
                g_particles=bit2id(g_samples),
                theta_particles=theta_samples,
                walltime=(t_end - t_start) * n_particles / n_particles_max,
                kwargs=kwargs,
                log=ave_acceptance_rate,
                error=None,
            ))

    # error handling
    except Exception as e:
        print(option_str, repr(e))

        for n_particles in n_particles_loop:
            dists.append(Distribution(
                passed_key=key,
                descr=option_str + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(structMCMC_hparams),
                joint=True,
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


def run_joint_bootstrap(*, r, c, key, n_particles_loop, kwargs, target, model, ig_log_joint_target, learner_str, no_bootstrap=False, additional_descr=None):
    """
    Single execution of DAG bootstrap with learner `learner`
    Run once for maximum number of particles, and then truncated for smaller n_particles queries
    """
    dists = []

    # if 1 particle is evaluated, do not bootstrap but use all the data
    # run this in advance
    if jnp.isin(n_particles_loop, 1).any() and not no_bootstrap:
        dists += run_joint_bootstrap(r=r, c=c, key=key,
            n_particles_loop=jnp.array([1]), kwargs=kwargs, 
            target=target, ig_log_joint_target=ig_log_joint_target, 
            learner_str=learner_str, no_bootstrap=True, additional_descr=additional_descr)

        n_particles_loop = n_particles_loop[jnp.where(n_particles_loop != 1)]

    # run with max number of particles and extract other results
    n_particles_max = max(n_particles_loop)

    if learner_str == 'joint_boot_ges':
        boot = NonparametricDAGBootstrap(
            learner=GES(), 
            verbose=False, 
            n_restarts=kwargs.joint_bootstrap_n_error_restarts,
            no_bootstrap=no_bootstrap,
        )

        boot_hparams = dict(
            bootstrap_n_error_restarts=kwargs.joint_bootstrap_n_error_restarts,
        )

    elif learner_str == 'joint_boot_pc':
        boot = NonparametricDAGBootstrap(
            learner=PC(
                ci_test=kwargs.joint_bootstrap_pc_ci_test,
                ci_alpha=kwargs.joint_bootstrap_pc_ci_alpha,
            ), 
            verbose=False, 
            n_restarts=kwargs.joint_bootstrap_n_error_restarts,
            no_bootstrap=no_bootstrap,
        )

        boot_hparams = dict(
            bootstrap_n_error_restarts=kwargs.joint_bootstrap_n_error_restarts,
            ci_test=kwargs.joint_bootstrap_pc_ci_test,
            ci_alpha=kwargs.joint_bootstrap_pc_ci_alpha,
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

        # MLE parameters
        x = jnp.array(target.x)
        cov_mat = (x.T @ x) / x.shape[0]
        mle_kwargs = {
            'type':     kwargs.joint_inference_model,
            'cov_mat':  cov_mat,
            'graphs':   boot_samples,
        }
        theta_particles = boot.learner.get_mle_params(mle_kwargs)
        
        t_end = time.time()
        
        # [L, d, d], PyTree with leading dim [L, ...] -> [L]
        def double_eltwise_log_joint_prob(g_mats, thetas):
            thetas_unzipped = tree_unzip_leading(thetas, g_mats.shape[0])
            return jnp.array([ig_log_joint_target(g_mat, theta) for g_mat, theta in zip(g_mats, thetas_unzipped)])

        # generate distributions for different n_particles used
        for n_particles in n_particles_loop:
            
            boot_dist = particle_joint_mixture(
                boot_samples[:n_particles], theta_particles[:n_particles], double_eltwise_log_joint_prob)
            
            dists.append(Distribution(
                passed_key=key,
                descr=learner_str + (additional_descr or ''),
                descr_hparams=hparam_dict_to_str(boot_hparams),
                joint=True,
                n_particles=n_particles,
                target=target,
                dist=boot_dist,
                g_particles=bit2id(boot_samples[:n_particles]),
                theta_particles=theta_particles,
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
                joint=True,
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


def run_joint_svgd(*, r, c, key, n_particles, kwargs, target, model, log_prior, log_joint_target, log_joint_target_no_batch, additional_descr=None):

    """
    Single execution of SVGD with `n_particles` particles
    """

    dists = []
    svgd_key = key.copy()

    # warning of other config if not available is issued at the beginning of `eval.py`
    config_ = joint_config[kwargs.joint_inference_model]
    kwargs_svgd = config_.get(kwargs.n_vars, config_[min(config_.keys())])
    
    # initialize
    if kwargs.joint_dibs_n_steps:
        kwargs_svgd['n_steps'] = kwargs.joint_dibs_n_steps
    n_steps = kwargs_svgd.get('n_steps', 3000) if not kwargs.smoke_test else 2
    
    optimizer = dict(name='rmsprop', stepsize=kwargs_svgd.get('opt_stepsize', kwargs.joint_dibs_opt_stepsize))

    if kwargs.joint_dibs_latent_dim:
        kwargs_svgd['latent_dim'] = kwargs.joint_dibs_latent_dim
    n_dim = kwargs_svgd.get('latent_dim', int(kwargs.n_vars / 2))

    if kwargs_svgd.get('graph_embedding_representation', kwargs.joint_dibs_graph_embedding_representation):
        latent_prior_std = (1.0 / jnp.sqrt(n_dim)).item()
    else:
        latent_prior_std = 1.0

    # assuming dot product representation of graph
    key, subk = random.split(key)
    if kwargs_svgd.get('graph_embedding_representation', kwargs.joint_dibs_graph_embedding_representation):
        init_particles_x = random.normal(subk, shape=(n_particles, kwargs.n_vars, n_dim, 2)) * latent_prior_std  * kwargs.joint_dibs_rel_init_scale
    else:
        init_particles_x = random.normal(subk, shape=(n_particles, kwargs.n_vars, kwargs.n_vars)) * latent_prior_std  * kwargs.joint_dibs_rel_init_scale

    key, subk = random.split(key)
    init_particles_theta = model.init_parameters(key=subk, n_particles=n_particles, n_vars=kwargs.n_vars)

    kernel_joint_choice = kwargs_svgd.get('kernel_choice', kwargs.joint_dibs_kernel)
    singular_dim_theta = 3 if kwargs.joint_inference_model == 'fcgauss' else 2

    if kernel_joint_choice == 'additive-frob':
        kernel_hparams = dict(
            h_latent=kwargs_svgd.get('h_latent', 1.0),
            h_theta=kwargs_svgd.get('h_theta', 1.0),
            scale_latent=kwargs_svgd.get('kernel_scale_latent', 1.0),
            scale_theta=kwargs_svgd.get('kernel_scale_theta', 1.0),
            soft_graph_mask=kwargs_svgd.get('soft_graph_mask', kwargs.joint_dibs_soft_graph_mask),
            singular_dim_theta=singular_dim_theta,
            graph_embedding_representation=kwargs_svgd.get('graph_embedding_representation', kwargs.joint_dibs_graph_embedding_representation),
        )
        kernel = JointAdditiveFrobeniusSEKernel(**kernel_hparams)

    elif kernel_joint_choice == 'multiplicative-frob':
        kernel_hparams = dict(
            h_latent=kwargs_svgd.get('h_latent', 1.0),
            h_theta=kwargs_svgd.get('h_theta', 1.0),
            scale=kwargs_svgd.get('kernel_scale_joint', 1.0),
            soft_graph_mask=kwargs_svgd.get('soft_graph_mask', kwargs.joint_dibs_soft_graph_mask),
            singular_dim_theta=singular_dim_theta,
            graph_embedding_representation=kwargs_svgd.get('graph_embedding_representation', kwargs.joint_dibs_graph_embedding_representation),
        )
        kernel = JointMultiplicativeFrobeniusSEKernel(**kernel_hparams)

    else:
        raise ValueError('Invalid joint kernel identifier')


    # temperature hyperparameters
    def linear_alpha(t):
        return jnp.minimum((kwargs_svgd.get('alpha_linear', kwargs.joint_alpha_linear) * t) + kwargs_svgd.get('alpha', 0.0), kwargs.joint_ceil_alpha)

    def linear_beta(t):
        return jnp.minimum((kwargs_svgd.get('beta_linear', kwargs.joint_beta_linear) * t) + kwargs_svgd.get('beta', 0.0), kwargs.joint_ceil_beta)

    def linear_tau(t):
        return jnp.minimum((kwargs_svgd.get('tau_linear', kwargs.joint_tau_linear) * t) + kwargs_svgd.get('tau', 0.0), kwargs.joint_ceil_tau)

    def exponential_alpha(t):
        return jnp.minimum(jnp.exp(kwargs_svgd.get('alpha_expo', kwargs.joint_alpha_expo) * t) * kwargs_svgd.get('alpha', 1.0), kwargs.joint_ceil_alpha)

    def exponential_beta(t):
        return jnp.minimum(jnp.exp(kwargs_svgd.get('beta_expo', kwargs.joint_beta_expo) * t) * kwargs_svgd.get('beta', 1.0), kwargs.joint_ceil_beta)

    def exponential_tau(t):
        return jnp.minimum(jnp.exp(kwargs_svgd.get('tau_expo', kwargs.joint_tau_expo) * t) * kwargs_svgd.get('tau', 1.0), kwargs.joint_ceil_tau)

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
        target_log_joint_prob=log_joint_target,
        target_log_joint_prob_no_batch=log_joint_target_no_batch,
        alpha=alpha_sched,
        beta=beta_sched,
        gamma=gamma_sched,
        tau=tau_sched,
        clip=None,
        fix_rotation=kwargs_svgd.get("fix_rotation", kwargs.joint_dibs_fix_rotation),
        grad_estimator_x=kwargs_svgd.get("grad_estimator_x", kwargs.joint_dibs_grad_estimator_x),
        grad_estimator_theta=kwargs_svgd.get("grad_estimator_theta", kwargs.joint_dibs_grad_estimator_theta),
        repulsion_in_prob_space=False,
        latent_prior_std=latent_prior_std,
        graph_embedding_representation=kwargs_svgd.get('graph_embedding_representation', kwargs.joint_dibs_graph_embedding_representation),
        verbose=False,
    )
    svgd = JointDotProductGraphSVGD(
        **dibs_init_params, 
        n_grad_mc_samples=kwargs_svgd.get('n_grad_mc_samples', kwargs.joint_dibs_n_grad_mc_samples),
        n_acyclicity_mc_samples=kwargs_svgd.get('n_acyclicity_mc_samples', kwargs.joint_dibs_n_acyclicity_mc_samples),
        score_function_baseline=kwargs_svgd.get('score_function_baseline', kwargs.joint_dibs_score_function_baseline),
        constraint_prior_graph_sampling=kwargs_svgd.get('constraint_prior_graph_sampling', kwargs.joint_dibs_constraint_prior_graph_sampling),
    )

    dibs_run_params = dict(
        key=key,
    )

    # all hparams
    svgd_hparams = dict(
        **kwargs_svgd,
        rel_init_scale=kwargs.joint_dibs_rel_init_scale,
        joint_dibs_n_grad_batch_size=kwargs.joint_dibs_n_grad_batch_size,
        **dibs_init_params, 
        **dibs_run_params,
    )
    
    # safe execution
    try:
        # evaluate
        t_start = time.time()        
        svgd_g_particles, svgd_theta = svgd.sample_particles(
            **dibs_run_params,
            init_particles_x=init_particles_x.copy(),
            init_particles_theta=init_particles_theta.copy(),
            verbose_indication=kwargs.verbose if r == 0 and c == 0 else 0,
            n_steps=n_steps,
        )
        t_end = time.time()

        # posterior
        svgd_hard_g = svgd.particle_to_hard_g(svgd_g_particles)
        svgd_empirical = particle_joint_empirical(svgd_hard_g, svgd_theta)
        svgd_mixture = particle_joint_mixture(svgd_hard_g, svgd_theta, svgd.double_eltwise_log_joint_prob_no_batch)

        dists.append(Distribution(
            passed_key=svgd_key,
            descr='dibs_empirical' + (additional_descr or ''),
            descr_hparams=hparam_dict_to_str(svgd_hparams),
            joint=True,
            n_particles=n_particles,
            target=target,
            dist=svgd_empirical,
            g_particles=bit2id(svgd_hard_g),
            theta_particles=svgd_theta,
            walltime=t_end - t_start,
            kwargs=kwargs,
            log=None,
            error=None,
        ))
        dists.append(Distribution(
            passed_key=svgd_key,
            descr='dibs_mixture' + (additional_descr or ''),
            descr_hparams=hparam_dict_to_str(svgd_hparams),
            joint=True,
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
            joint=True,
            n_particles=n_particles,
            target=target,
            dist=None,
            g_particles=None,
            theta_particles=None,
            walltime=None,
            kwargs=kwargs,
            log=None,
            error=e,
        )

        dists.append(Distribution(descr='dibs_empirical' + (additional_descr or ''), **svgd_error_dist))
        dists.append(Distribution(descr='dibs_mixture' + (additional_descr or ''), **svgd_error_dist))

    return dists

