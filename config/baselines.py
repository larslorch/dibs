

"""
config[inference_model_str][n_vars]
"""


#
# Marginal
#

struct_mcmc_config = dict()
struct_mcmc_config["bge"] = {
    20: {
        "burnin":           1e5,
        "thinning":         1e4,
    },
    50: {
        "burnin":           1e5,
        "thinning":         1e4,
    },
}


#
# Joint
#

mh_mcmc_config = dict()
mh_mcmc_config["lingauss"] = {
    20: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.001,
    },
    50: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.001,
    },
}
mh_mcmc_config["fcgauss"] = {
    20: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.0001,
    },
    50: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.0001,
    },
}



gibbs_mcmc_config = dict()
gibbs_mcmc_config["lingauss"] = {
    20: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.05,
    },
    50: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.01,
    },
}
gibbs_mcmc_config["fcgauss"] = {
    20: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.005,
    },
    50: {
        "burnin":           1e5,
        "thinning":         1e4,
        "theta_prop_sig":   0.001,
    },
}
