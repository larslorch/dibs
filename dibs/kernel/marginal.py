
import jax.numpy as jnp
import jax.lax as lax


class FrobeniusSquaredExponentialKernel:
    """
    Squared exponential kernel
    Computes the exponentiated quadratic of the difference in Frobenius norms

    k(W, W') = scale * exp(- 1/h ||W - W'||^2_F )
    """

    def __init__(self, *, h, scale=1.0):
        super(FrobeniusSquaredExponentialKernel, self).__init__()

        self.h = h
        self.scale = scale

    def eval(self, *, x, y, h=-1.0):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x: [...] 
            y: [...] 
            h (float): bandwidth; h == -1 indicates class setting is used
        
        Returns: 
            [1,] 
        """

        # bandwidth (jax-consistent checking which h is used)
        h_ = lax.cond(
            h == -1.0,
            lambda _: self.h,
            lambda _: h,
            operand=None)

        # compute norm
        squared_norm = jnp.sum((x - y) ** 2.0)

        # compute kernel
        return self.scale * jnp.exp(- squared_norm / h_)
