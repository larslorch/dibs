from dibs.utils.system import gigab


marginal_config = dict()
marginal_config["bge"] = {
    20: {
        "latent_dim":                       20,
        "h":                                2.0,
        "alpha_linear":                     2.0,
        "beta_linear":                      1.0,
        "n_grad_mc_samples":                128,
        "n_steps":                          3000,
        "resources": {
            "num_cpus":                     2,
            "memory":                       gigab(2.0),
        }
    },
    50: {
        "latent_dim":                       50,
        "h":                                50.0,
        "alpha_linear":                     2.0,
        "beta_linear":                      1.0,
        "n_grad_mc_samples":                128,
        "n_steps":                          3000,
        "resources": {
            "num_cpus":                     4,
            "memory":                       gigab(20.0),
        }
    },
}

joint_config = dict()
joint_config["lingauss"] = {
    20: {
        "latent_dim":                       20,
        "h_latent":                         5.0,
        "h_theta":                          500.0,
        "alpha_linear":                     0.2,
        "beta_linear":                      1.0,
        "n_grad_mc_samples":                128,
        "n_steps":                          3000,
        "resources": {
            "num_cpus":                     1,
            "memory":                       gigab(1.0),
        }
    },
    50: {
        "latent_dim":                       50,
        "h_latent":                         15.0,
        "h_theta":                          1000.0,
        "alpha_linear":                     0.02,
        "beta_linear":                      1.0,
        "n_grad_mc_samples":                128,
        "n_steps":                          3000,
        "resources": {
            "num_cpus":                     2,
            "memory":                       gigab(3.0),
        }
    },
}

joint_config["fcgauss"] = {
    20: {
        "latent_dim":                       20,
        "h_latent":                         5.0,
        "h_theta":                          1000.0,
        "alpha_linear":                     0.02,
        "beta_linear":                      1.0,
        "n_grad_mc_samples":                128,
        "n_steps":                          3000,
        "resources": {
            "num_cpus":                     2,
            "memory":                       gigab(10.0),
        }
    },
    50: {
        "latent_dim":                       50,
        "h_latent":                         15.0,
        "h_theta":                          2000.0,
        "alpha_linear":                     0.01,
        "beta_linear":                      1.0,
        "n_grad_mc_samples":                128,
        "n_steps":                          3000,
        "resources": {
            "num_cpus":                     3,
            "memory":                       gigab(30.0),
        }
    },
}



