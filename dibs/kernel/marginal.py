import jax.numpy as jnp

class AdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel
    Computes the exponentiated quadratic of the difference in Frobenius norms

    k(W, W') = scale * exp(- 1/h ||W - W'||^2_F )
    """

    def __init__(self, *, h=20.0, scale=1.0):
        super(AdditiveFrobeniusSEKernel, self).__init__()

        self.h = h
        self.scale = scale

    def eval(self, *, x, y):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x: [...] 
            y: [...] 

        Returns: 
            [1,] 
        """
        return self.scale * jnp.exp(- jnp.sum((x - y) ** 2.0) / self.h)
