from jax.ops import index, index_add, index_mul, index_update
import jax.lax as lax
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp


import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':

    def f(x):
        b = 3.5 
        mask = x > 1/4
        return jnp.where(mask, jnp.sqrt(x), x + b)


    xlim = [-1e3, 1e3]
    xs = jnp.linspace(xlim[0], xlim[1], 50)
    plt.plot(xs, f(xs))
    plt.show()


