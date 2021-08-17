
import argparse
from dibs.utils.system import str2bool

def make_evaluation_parser():
    """
    Returns argparse parser to control evaluation from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--rel_cpu_usage", type=float, default=1.0, help="fraction of available CPUs allocated")
    parser.add_argument("--smoke_test", action="store_true", help="If passed, minimal iterations to see if something breaks")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--descr", required=True, help="set experiment filename; keep the same to resume in case interrupted")
    parser.add_argument("--create_targets_only", action="store_true", help="Only create targets, do not evaluate.")
    parser.add_argument("--timeout", type=float, help="in seconds")
    parser.add_argument("--n_particles_loop", default=[30], nargs='+', type=int, help="number of posterior samples per Method")

    # memory
    parser.add_argument("--resource_aware", type=str2bool, default=True, help="memory aware scheduling")

    # ablation studies/additional analyses
    parser.add_argument("--grid_dibs_graph_embedding_representation", action="store_true", help="Runs DiBS with and without graph embed rep")
    parser.add_argument("--grid_joint_dibs_graph_embedding_representation", action="store_true", help="Runs DiBS with and without graph embed rep")

    parser.add_argument("--grid_dibs_latent_dim", action="store_true", help="Runs DiBS for different latent dims")
    parser.add_argument("--grid_joint_dibs_latent_dim", action="store_true", help="Runs DiBS for different latent dims")

    parser.add_argument("--grid_dibs_steps", action="store_true", help="Runs DiBS for variable number of steps")
    parser.add_argument("--grid_joint_dibs_steps", action="store_true", help="Runs DiBS for variable number of steps")

    # real data
    parser.add_argument("--real_data", default=None, help="If passed, uses real data set") 
    parser.add_argument("--real_data_held_out", type=str2bool, default=False, help="If passed, real data set is not split into train/eval")
    parser.add_argument("--real_data_normalize", type=str2bool, default=True, help="If passed, real data set is normalized")

    '''Target'''
    parser.add_argument("--joint", action="store_true", help="If true, tunes evaluation of /joint/ posterior p(G, theta | D) methods") 

    parser.add_argument("--n_variants", type=int, default=1, help="number of different targets optimized and averaged; equally assigned to batch rollouts")
    parser.add_argument("--n_rollouts", type=int, default=1, help="number of rollouts done per method for a single target")

    # generative model    
    parser.add_argument("--graph_prior", nargs="+", default=["er", "sf"], help="inference model")
    parser.add_argument("--inference_model", default="bge", choices=["bge", "lingauss"], help="inference model")
    parser.add_argument("--joint_inference_model", default="lingauss", choices=["lingauss", "fcgauss"], help="joint inference model")

    parser.add_argument("--n_vars", type=int, default=10, help="number of variables in graph")
    parser.add_argument("--n_observations", type=int, default=100, help="number of observations defining the ground truth posterior")
    parser.add_argument("--n_ho_observations", type=int, default=100, help="number of held out observations for validation")
    parser.add_argument("--n_intervention_sets", type=int, default=10, help="number of sets of observations sampled with random interventions")
    parser.add_argument("--perc_intervened", type=float, default=0.2, help="percentage of nodes intervened upon")

    parser.add_argument("--n_posterior_g_samples", type=int, default=100, help="number of ground truth graph samples")

    # inference model
    parser.add_argument("--gbn_lower", type=float, default=1.0, help="GBN Sampler")
    parser.add_argument("--gbn_upper", type=float, default=3.0, help="GBN Sampler")
    parser.add_argument("--gbn_node_mean", type=float, default=0.0, help="GBN Sampler")
    parser.add_argument("--gbn_node_sig", type=float, default=1.0, help="GBN Sampler")
    parser.add_argument("--gbn_obs_sig", type=float, default=0.1, help="GBN Sampler")

    parser.add_argument("--lingauss_obs_noise", type=float, default=0.1, help="linear Gaussian")
    parser.add_argument("--lingauss_mean_edge", type=float, default=0.0, help="linear Gaussian")
    parser.add_argument("--lingauss_sig_edge", type=float, default=1.0, help="linear Gaussian")

    parser.add_argument("--fcgauss_obs_noise", type=float, default=0.1, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_sig_param", type=float, default=1.0, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_hidden_layers", type=int, default=1, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_n_neurons", type=int, default=5, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_activation", type=str, default="relu", help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_bias", type=str2bool, default=True, help="fully-connected NN Gaussian")

    parser.add_argument("--bge_alpha_mu", type=float, default=1.0, help="BGe")
    parser.add_argument("--bge_alpha_lambd_add", type=float, default=2.0, help="BGe")

    parser.add_argument("--skip_baselines", action="store_true")

    '''
    #
    # Marginal inference methods
    # 
    '''

    # mcmc
    parser.add_argument("--skip_mcmc_structure", action="store_true")
    parser.add_argument("--mcmc_thinning", type=float, default=1e3, help="steps skipped between samples")
    parser.add_argument("--mcmc_burnin", type=float, default=1e4, help="initial steps skipped")
    parser.add_argument("--mcmc_only_non_covered", action="store_true", help="default is False")

    # bootstrap
    parser.add_argument("--skip_bootstrap_ges", action="store_true")
    parser.add_argument("--skip_bootstrap_pc", action="store_true")
    parser.add_argument("--bootstrap_n_error_restarts", type=int, default=20)
    parser.add_argument("--bootstrap_pc_ci_test", default="gaussian", help="conditional independence test",
        choices=["binary", "discrete", "hsic_gamma", "hsic_perm", "hsic_clust", "gaussian", "rcit", "rcot"])
    parser.add_argument("--bootstrap_pc_ci_alpha", type=float, default=0.05, help="conditional independence threshold")

    # dibs
    parser.add_argument("--skip_dibs", action="store_true")
    parser.add_argument("--dibs_n_steps", type=int, help="svgd maximum steps")
    parser.add_argument("--dibs_latent_dim", type=int, help="svgd particles latent dim")
    parser.add_argument("--dibs_rel_init_scale", type=float, default=1.0, help="svgd particles")
    parser.add_argument("--dibs_opt_stepsize", type=float, default=0.005, help="learning rate")
    parser.add_argument("--dibs_constraint_prior_graph_sampling", default="soft", choices=[None, "soft", "hard"], help="acyclicity constraint sampling")
    parser.add_argument("--dibs_n_grad_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--dibs_n_acyclicity_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--dibs_score_function_baseline", type=float, default=0.0, help="gradient estimator baseline; 0.0 corresponds to not using a baseline")
    parser.add_argument("--dibs_fix_rotation", default="not", choices=["not", "parallel", "orthogonal"], help="whether and how to fix u0 = v0")
    parser.add_argument("--dibs_graph_embedding_representation", type=str2bool, default=True, help="whether to use graph embedding representation")

    parser.add_argument("--alpha_linear", type=float, default=0.0, help="alpha linear default")
    parser.add_argument("--beta_linear", type=float, default=0.0, help="beta linear default")
    parser.add_argument("--tau_linear", type=float, default=0.0, help="tau linear default")

    parser.add_argument("--alpha_expo", type=float, default=0.0, help="alpha expo default")
    parser.add_argument("--beta_expo", type=float, default=0.0, help="beta expo default")
    parser.add_argument("--tau_expo", type=float, default=0.0, help="tau expo default")

    parser.add_argument("--ceil_alpha", type=float, default=1e9, help="maximum value for alpha")
    parser.add_argument("--ceil_beta", type=float, default=1e9, help="maximum value for beta")
    parser.add_argument("--ceil_tau", type=float, default=1e9, help="maximum value for tau")

    '''
    #
    # Joint inference methods
    # 
    '''

    # MH joint structure mcmc
    parser.add_argument("--skip_mh_joint_mcmc_structure", action="store_true")
    parser.add_argument("--mh_joint_mcmc_thinning", type=float, default=1e3, help="steps skipped between samples")
    parser.add_argument("--mh_joint_mcmc_burnin", type=float, default=1e4, help="initial steps skipped")
    parser.add_argument("--mh_joint_mcmc_only_non_covered", action="store_true", help="default is False")
    parser.add_argument("--mh_joint_mcmc_scale", type=float, default=1e-3, help="scale of proposal random walk for theta")

    # MH-within-Gibbs joint structure mcmc
    parser.add_argument("--skip_gibbs_joint_mcmc_structure", action="store_true")
    parser.add_argument("--gibbs_joint_mcmc_thinning", type=float, default=1e3, help="steps skipped between samples")
    parser.add_argument("--gibbs_joint_mcmc_burnin", type=float, default=1e4, help="initial steps skipped")
    parser.add_argument("--gibbs_joint_mcmc_only_non_covered", action="store_true", help="default is False")
    parser.add_argument("--gibbs_joint_mcmc_scale", type=float, default=1e-3, help="scale of proposal random walk for theta")

    # bootstrap
    parser.add_argument("--skip_joint_bootstrap_ges", action="store_true")
    parser.add_argument("--skip_joint_bootstrap_pc", action="store_true")
    parser.add_argument("--joint_bootstrap_n_error_restarts", type=int, default=20)
    parser.add_argument("--joint_bootstrap_pc_ci_test", default="gaussian", help="conditional independence test",
        choices=["binary", "discrete", "hsic_gamma", "hsic_perm", "hsic_clust", "gaussian", "rcit", "rcot"])
    parser.add_argument("--joint_bootstrap_pc_ci_alpha", type=float, default=0.05, help="conditional independence threshold")

    # dibs
    parser.add_argument("--skip_joint_dibs", action="store_true")
    parser.add_argument("--joint_dibs_n_steps", type=int, help="svgd maximum steps")
    parser.add_argument("--joint_dibs_latent_dim", type=int, help="svgd particles latent dim")
    parser.add_argument("--joint_dibs_rel_init_scale", type=float, default=1.0, help="svgd particles")
    parser.add_argument("--joint_dibs_opt_stepsize", type=float, default=0.005, help="learning rate")
    parser.add_argument("--joint_dibs_constraint_prior_graph_sampling", default="soft", choices=[None, "soft", "hard"], help="acyclicity constraint sampling")
    parser.add_argument("--joint_dibs_n_grad_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--joint_dibs_n_grad_batch_size", type=int, help="svgd observation minibatch size; if not specificed, uses the whole dataset")
    parser.add_argument("--joint_dibs_n_acyclicity_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--joint_dibs_score_function_baseline", type=float, default=0.0, help="gradient estimator baseline; 0.0 corresponds to not using a baseline")
    parser.add_argument("--joint_dibs_fix_rotation", default="not", choices=["not", "parallel", "orthogonal"], help="whether and how to fix u0 = v0")
    parser.add_argument("--joint_dibs_kernel", default="additive-frob", choices=["additive-frob", "multiplicative-frob"], help="joint kernel")
    parser.add_argument("--joint_dibs_grad_estimator_x", default="reparam_soft", choices=["score", "reparam_soft", "reparam_hard"], help="gradient estimator for x in joint inference")
    parser.add_argument("--joint_dibs_grad_estimator_theta", default="hard", choices=["hard"], help="gradient estimator for theta in joint inference")
    parser.add_argument("--joint_dibs_soft_graph_mask", type=str2bool, default=False, help="whether joint kernel (soft-)masks unused parameters")
    parser.add_argument("--joint_dibs_graph_embedding_representation", type=str2bool, default=True, help="whether to use graph embedding representation")

    parser.add_argument("--joint_alpha_linear", type=float, default=0.0, help="alpha linear default")
    parser.add_argument("--joint_beta_linear", type=float, default=0.0, help="beta linear default")
    parser.add_argument("--joint_tau_linear", type=float, default=0.0, help="tau linear default")

    parser.add_argument("--joint_alpha_expo", type=float, default=0.0, help="alpha expo default")
    parser.add_argument("--joint_beta_expo", type=float, default=0.0, help="beta expo default")
    parser.add_argument("--joint_tau_expo", type=float, default=0.0, help="tau expo default")

    parser.add_argument("--joint_ceil_alpha", type=float, default=1e9, help="maximum value for alpha")
    parser.add_argument("--joint_ceil_beta", type=float, default=1e9, help="maximum value for beta")
    parser.add_argument("--joint_ceil_tau", type=float, default=1e9, help="maximum value for tau")

    return parser
