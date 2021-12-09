import jax.numpy as jnp
from dibs.utils.func import squared_norm_pytree

class JointAdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, theta], [Z', theta']) = 
        scale_z     * exp(- 1/h_z  ||Z - Z'||^2_F)
      + scale_theta * exp(- 1/h_th ||theta - theta'||^2_F )

    """

    def __init__(self, *, h_latent=5.0, h_theta=500.0, scale_latent=1.0, scale_theta=1.0):
        super(JointAdditiveFrobeniusSEKernel, self).__init__()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.scale_latent = scale_latent
        self.scale_theta = scale_theta
       

    def eval(self, *, x_latent, x_theta, y_latent, y_theta):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x_latent: [...]
            x_theta: PyTree 
            y_latent: [...]
            y_theta: PyTree 

        Returns: 
            [1, ]
        """

        # compute norm
        latent_squared_norm = jnp.sum((x_latent - y_latent) ** 2.0)
        theta_squared_norm = squared_norm_pytree(x_theta, y_theta)

        # compute kernel
        return (self.scale_latent * jnp.exp(- latent_squared_norm / self.h_latent)
              + self.scale_theta  * jnp.exp(- theta_squared_norm  / self.h_theta ))

