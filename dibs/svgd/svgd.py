


import tqdm
import time

import jax.numpy as jnp
from jax import jit, vmap, random, grad, vjp, jvp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jax.lax import stop_gradient
from jax.ops import index, index_add, index_update


class ExampleTarget2D:
    def __init__(self, key):
        super(ExampleTarget2D, self).__init__()

        self.key = key
        cov = 0.7
        self.pi = 0.5
        self.mean = jnp.array([[1, 0], [-1, -2]], dtype=jnp.float32)
        self.cov = jnp.array([[[1, cov], [cov, 1]], [[1, -cov], [-cov, 1]]], dtype=jnp.float32)


    def sample(self, n):
        self.key, *subk = random.split(self.key, 4)
        z = random.choice(subk[0], 2, p=jnp.array([self.pi, 1 - self.pi]), shape=(n, 1))
        return (1 - z) * random.multivariate_normal(key=subk[1], mean=self.mean[0], cov=self.cov[0], shape=(n,)) \
                   + z * random.multivariate_normal(key=subk[2], mean=self.mean[1], cov=self.cov[1], shape=(n,))

    def log_prob(self, x):
        both_x = jnp.array([jnp.log(self.pi) + multivariate_normal.logpdf(x, self.mean[0], self.cov[0]),
                           jnp.log(1 - self.pi) + multivariate_normal.logpdf(x, self.mean[1], self.cov[1])])
        return logsumexp(both_x, axis=0)
 

class SVGD:
    """
    Stein variational gradient descent as proposed by
        https://arxiv.org/abs/1608.04471

        kernel :            object satisfying `BasicKernel` signature
        target_log_prob :   function whose evaluation returns the log probability of our target distribution

    """

    def __init__(self, *, kernel, target_log_prob, key, verbose=False):
        super(SVGD, self).__init__()

        self.kernel = kernel
        self.target_log_prob = target_log_prob
        self.key = key
        self.verbose = verbose
        self.has_init_core_functions = False
        self.init_core_functions()


    def init_core_functions(self):
        '''Defines functions needed for SVGD and uses jit'''
        
        # log prob grad
        grad_log_prob = jit(grad(self.target_log_prob, 0))
        self.eltwise_grad_log_prob = jit(vmap(grad_log_prob, 0, 0))

        # kernel eval and grad
        def f_kernel_(a, b, h):
            return self.kernel.eval(x=a, y=b, h=h)

        self.f_kernel = jit(f_kernel_)
        grad_kernel = jit(grad(self.f_kernel, 0))
        eltwise_grad_kernel = jit(vmap(grad_kernel, (0, None, None), 0))

        # define single particle update for jit and vmap
        def x_update(single_x, kxx_for_x, x, grad_log_prob, h):

            # compute terms in sum
            weighted_gradient_ascent = kxx_for_x[..., jnp.newaxis] * grad_log_prob
            repulsion = eltwise_grad_kernel(x, single_x, h)

            # average
            return (weighted_gradient_ascent + repulsion).mean(axis=0)

        self.parallel_update = jit(vmap(x_update, (0, 1, None, None, None), 0))

        self.has_init_core_functions = True



    def sample_particles(self, *, n_steps, init_particles=None, stepsize=1e-3, eval_metric=None):
        """
        Deterministically transforms particles as provided by `init_particles`
        (or heuristically set by N(0, I) of shape (`n_particles`, `n_dim`))
        to minimize KL to target using SVGD
        """
        h_is_none = self.kernel.h == -1.0

        assert(init_particles.ndim == 2)
        x = init_particles
        n_particles, n_dim = x.shape
        
        # jit core functions
        if not self.has_init_core_functions:
            self.init_core_functions()

        '''execute particle updates'''
        it = tqdm.tqdm(range(n_steps), desc='SVGD', disable=not self.verbose)
        for _ in it:

            # make sure same bandwith is used for all calls to k(x,x') if the median heuristic is applied
            h = stop_gradient(self.kernel.compute_median_heuristic(x=x, y=x)) \
                if h_is_none else self.kernel.h

            # d/dx log prob(x)
            dx_log_prob = self.eltwise_grad_log_prob(x)

            # k(x, x)
            kxx = self.f_kernel(x, x, h)

            # transformation phi(x)
            phi = self.parallel_update(x, kxx, x, dx_log_prob, h)
            x += stepsize * phi

            # evaluate
            if eval_metric:
                it.set_description('SVGD | metric: {:10.06f}'.format(eval_metric(x)))

        return x
