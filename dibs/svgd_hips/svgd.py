


import tqdm
import time
import scipy.stats as stats

import autograd.numpy as np
from autograd import grad, jacobian, elementwise_grad
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.core import getval

from dibs.utils.func import pairwise_squared_norm

def logsumexp(x, axis=None):
    """Numerically stable log(sum(exp(x))), also defined in scipy.special"""
    max_x = np.max(x, axis=axis)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis))


class ExampleTarget2D:
    def __init__(self):
        super(ExampleTarget2D, self).__init__()

        cov = 0.7
        self.pi = 0.5
        self.mean = np.array([[1, 0], [-1, -2]])
        self.cov = np.array([[[1, cov], [cov, 1]], [[1, -cov], [-cov, 1]]])

    def log_prob(self, x):
        both_x = np.array([np.log(self.pi) + mvn.logpdf(x, self.mean[0], self.cov[0]),
                           np.log(1 - self.pi) + mvn.logpdf(x, self.mean[1], self.cov[1])])
        return logsumexp(both_x, axis=0)
 


class SVGD:
    """
    Stein variational gradient descent as proposed by
        https://arxiv.org/abs/1608.04471

        kernel :   object satisfying `BasicKernel` signature
        target :   object of `torch.nn.Module`, where forward() evaluates the log probability of our target distribution

    """

    def __init__(self, *, kernel, target, verbose=False):
        super(SVGD, self).__init__()

        self.kernel = kernel
        self.target = target
        self.verbose = verbose

    def step(self, *, x, stepsize):
        """
        SVGD update step
            x : [n_particles, n_dim]
        """
        n_particles, n_dim = x.shape
        h_is_none = self.kernel.h == -1.0
        
        # make sure same bandwith is used for all calls to k(x,x') if the median heuristic is applied
        h = getval(self.kernel.compute_median_heuristic(x=x, y=x)) \
            if h_is_none else self.kernel.h

        # gradient functions
        egrad_log_prob = elementwise_grad(self.target.log_prob)
        egrad_kxx = elementwise_grad((lambda a, b: self.kernel.eval(x=a, y=b, h=h)), 0)

        # d/dx_l log prob (x_l)
        grad_log_prob = egrad_log_prob(x)

        # k(x, x)
        kxx = self.kernel.eval(x=x, y=x, h=h)

        phi = np.zeros_like(x)

        # update all particles 
        for l in range(n_particles):

            # compute all sum terms
            gradient_ascent = kxx[:, l, np.newaxis] * grad_log_prob
            repulsion = egrad_kxx(x, x[l])

            phi[l] = (gradient_ascent + repulsion).mean(axis=0)
            
        # apply transform
        x += stepsize * phi

        return x


    def sample_particles(self, *, n_steps, init_particles=None, stepsize=1e-3, alpha=0.9):
        """
        Deterministically transforms particles as provided by `init_particles`
        (or heuristically set by N(0, I) of shape (`n_particles`, `n_dim`))
        to minimize KL to target using SVGD
        """
        
        assert(init_particles.ndim == 2)
        x = init_particles
        n_particles, n_dim = x.shape
        
        # execute particle updates
        for _ in tqdm.tqdm(range(n_steps), desc='SVGD', disable=not self.verbose):
            x = self.step(x=x, stepsize=stepsize)

        return x
