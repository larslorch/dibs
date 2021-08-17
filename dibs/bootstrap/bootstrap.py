import tqdm 
import time
from datetime import datetime

import numpy as onp

from jax.ops import index, index_add, index_update
from jax import random 
from jax.scipy.special import logsumexp
import jax.numpy as jnp

from dibs.utils.graph import *
from dibs.utils.func import bit2id, id2bit

from dibs.exceptions import InvalidCPDAGError, ContinualInvalidCPDAGError, BaselineTimeoutError


class NonparametricDAGBootstrap:
    """
    Nonparametric DAG Bootstrap as proposed by
        https://arxiv.org/abs/1608.04471
    
    and e.g. used in 
        https://arxiv.org/abs/1902.10347 (Algorithm 1)

    Arguments: 

        learner :    
            DAG learning algorithm satisfying the following signature
                in: 
                    x :   [n_data, n_vars] a number of observations used to learn a DAG
                out:
                    mat : [n_vars, n_vars] adjacency matrix of a DAG

        n_restarts: int
            number of restarts with lower confidence thresholds in case an invalid CPDAG is returned

        no_bootstrap: bool
            if true, does not do any bootstrapping and is run once on the full dataset
    """

    def __init__(self, *, learner, verbose, n_restarts=0, no_bootstrap=False):
        super(NonparametricDAGBootstrap, self).__init__()

        self.learner = learner
        self.verbose = verbose
        self.n_restarts = n_restarts
        self.no_bootstrap = no_bootstrap

    def sample_particles(self, *, key, n_samples, x, n_data=None, verbose_indication=0):
        """
        Generates `n_samples` DAGs by bootstrapping (sampling with replacement) `n_data` points from x
        and learning a single DAG using an external DAG learning algorithm, in total `n_samples` times

            key
            x : [n_observations, n_vars]
            n_samples : int

        Returns 
            dags: [n_samples, n_vars, n_vars]
        """
        last_verbose_indication = 1
        t_start = time.time()

        dags = []
        for l in tqdm.tqdm(range(n_samples), desc='NonparametricDAGBootstrap', disable=not self.verbose):

            # sample bootstrap dataset 
            # (`n_data = None` indicates `n_data = n_observations`)
            if self.no_bootstrap:
                boot_sample = x
            else:
                n_observations = x.shape[0]
                n = n_data or n_observations

                key, subk = random.split(key)
                idxs = random.choice(subk, n_observations, shape=(n,), replace=True)
                boot_sample = x[idxs]                

            # learn DAG
            key, subk = random.split(key)

            attempts = 0
            while True:
                try:
                    mat = self.learner.learn_dag(key=subk, x=boot_sample)
                    dags.append(mat)
                    break

                except InvalidCPDAGError as e:
                    # if invalid CPDAG rerun with harder confidence threshold
                    attempts += 1
                    self.learner.reinit(ci_alpha=self.learner.ci_alpha / 2.0)
                    if attempts > self.n_restarts:
                        if self.verbose:
                            print(
                                f'{type(self.learner).__name__} did not return an extendable CPDAG '
                                'likely due to an undirected chain even for high-confidence levels. \n'
                                'Skipping this bootstrap sample.'
                            )

            if self.no_bootstrap:
                break

            # verbose progress
            if verbose_indication > 0:
                if (l + 1) >= (last_verbose_indication * n_samples // verbose_indication):
                    print(
                        f'DAGBootstrap {type(self.learner).__name__}    {l + 1} / {n_samples} [{(100 * (l + 1) / n_samples):3.1f} % '
                        + f'| {((time.time() - t_start)/60):.0f} min | {datetime.now().strftime("%d/%m %H:%M")}]',
                        flush=True
                    )
                    last_verbose_indication += 1

        if not dags:
            raise ContinualInvalidCPDAGError('Could not find a valid DAG for any of the boostrap datasets.')

        return jnp.array(dags)
