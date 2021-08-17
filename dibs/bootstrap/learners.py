import jax.numpy as jnp
from jax.ops import index, index_update
import numpy as onp
import pandas as pd

import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")

from cdt.causality.graph import GES as rGES
from cdt.causality.graph import PC as rPC

from dibs.utils.graph import *


def compute_linear_gaussian_mle_params(mle_kwargs):
    """
    Computes MLE parameters for linear GBN
    See Hauser et al
    https://arxiv.org/pdf/1303.3216.pdf 
    Page 17

    Based on https://github.com/agrawalraj/active_learning
    from the paper Agrawal et al 
    https://arxiv.org/pdf/1902.10347.pdf
    """

    cov_mat = mle_kwargs['cov_mat']
    n_vars = cov_mat.shape[-1]

    graphs = mle_kwargs['graphs']
    coeffs = jnp.zeros_like(graphs, dtype=cov_mat.dtype)

    # for each graph
    for graph_idx, g in enumerate(graphs):
        
        # for each node and its parents
        for j in range(n_vars):
            
            parents = onp.where(g[:, j] == 1)[0]
            if len(parents) > 0:

                cov_j_j = cov_mat[j, j]
                cov_j_pa = cov_mat[j, parents]
                cov_pa_pa = cov_mat[jnp.ix_(parents, parents)]

                if len(parents) > 1:
                    inv_cov_pa_pa = jnp.linalg.inv(cov_pa_pa)
                else:
                    inv_cov_pa_pa = jnp.array(1 / cov_pa_pa)

                mle_coeffs_pa_j = cov_j_pa.dot(inv_cov_pa_pa)

                # jax.numpy way for: coeffs[graph_idx, parents, j] = mle_coeffs_pa_j
                coeffs = index_update(coeffs, index[graph_idx, parents, j], mle_coeffs_pa_j)

    return coeffs



class DAGLearner:
    """
    Class as called by Bootstrap implementing an external DAG learning method
    """

    def __init__(self):
        super(DAGLearner, self).__init__()
        pass


    def learn_dag(self, x, cpdag=None):
        """
        Learns DAG from data
            x :         [n_observations, n_vars]
            cpdag :     [n_vars, n_vars] if provided, returns random consistent extension of CPDAG

        Returns 
            dag:        [n_vars, n_vars] 
        """
        raise NotImplementedError 

    def get_mle_params(self, mle_kwargs):
        """
        Computes MLE parameters for a given distribution
        """

        type = mle_kwargs['type']
        if type == 'lingauss':
            return compute_linear_gaussian_mle_params(mle_kwargs)

        else:
            raise NotImplementedError(f'No MLE parameter implementation available for type `{type}`')


class GES(DAGLearner):
    '''
    Greedy equivalence search

    'obs' : GaussL0penObsScore corresponds to BIC
        l0-penalized Gaussian MLE estimator. By default,
        score = log(L(D)) − k * log(n)/2
        corresponding exactly to BIC
        Specifically, assumes linear structural equation model with Gaussian noise
    
    
    'int' : GaussL0penIntScore is intended for a mixture of data sources
        i.e. observational and interventional data
    
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf
    https://rdrr.io/cran/pcalg/api/

    '''

    def __init__(self):
        super(DAGLearner, self).__init__()

        self.alg = rGES(score='obs', verbose=False)  # uses BIC score

    def reinit(self, *, ci_alpha):
        raise NotImplementedError

    def learn_cpdag(self, *, x):
        """
        Learns CPDAG from data
            x :         [n_observations, n_vars]

        Returns 
            cpdag:        [n_vars, n_vars] 
        """
        data = pd.DataFrame(data=x)
        pred_cpdag = self.alg.predict(data)
        pred_cpdag_mat = nx_adjacency(pred_cpdag)
        return jnp.array(pred_cpdag_mat, dtype=jnp.int32)

    def learn_dag(self, *, key, x, cpdag=None):
        # CPDAG
        if cpdag is None:
            cpdag = self.learn_cpdag(x=x)

        # Random consistent extension (DAG in MEC)
        dag = random_consistent_expansion(key=key, cpdag=cpdag)

        return dag


class PC(DAGLearner):
    '''
    Peter - Clark (PC) algorithm 
    can return a cycle, at least it does for `sachs` in CDT datasets

    ci_test:
        'binary',       # 0 "pcalg::binCItest",
        'discrete',     # 1 "pcalg::disCItest",
        'hsic_gamma',   # 2 "kpcalg::kernelCItest",
        'hsic_perm',    # 3 "kpcalg::kernelCItest",
        'hsic_clust',   # 4 "kpcalg::kernelCItest",
        'gaussian',     # 5 "pcalg::gaussCItest",
        'rcit',         # 6 "RCIT:::CItest",
        'rcot',         # 7 "RCIT:::CItest"}

    ci_alpha:   significance level for the individual CI tests

    '''

    def __init__(self, ci_test='gaussian', ci_alpha=0.01):
        super(DAGLearner, self).__init__()

        self.ci_test = ci_test 
        self.ci_alpha = ci_alpha
        self.alg = rPC(CItest=ci_test, alpha=ci_alpha, njobs=None, verbose=False)

    def reinit(self, *, ci_alpha):
        """
        Re-initializes with different confidence level `ci_alpha`
        """
        self.ci_alpha = ci_alpha
        self.alg = rPC(CItest=self.ci_test, alpha=ci_alpha, njobs=None, verbose=False)

    def learn_cpdag(self, *, x):
        """
        Learns CPDAG from data
            x :         [n_observations, n_vars]

        Returns 
            cpdag:        [n_vars, n_vars] 
        """
        data = pd.DataFrame(data=x)
        pred_cpdag = self.alg.predict(data)
        pred_cpdag_mat = nx_adjacency(pred_cpdag)
        return jnp.array(pred_cpdag_mat, dtype=jnp.int32)

    def learn_dag(self, *, key, x, cpdag=None):
        # CPDAG
        if cpdag is None:
            cpdag = self.learn_cpdag(x=x)

        # Random consistent extension (DAG in MEC)
        dag = random_consistent_expansion(key=key, cpdag=cpdag)

        return dag

