
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import stop_gradient
import jax.lax as lax
from jax.nn import sigmoid, log_sigmoid
from jax.ops import index, index_add, index_update, index_mul

from dibs.utils.func import (
    median_heuristic_from_squared_norm, 
    pairwise_squared_norm, 
    pairwise_squared_norm_mat,
    pairwise_structural_hamming_distance, 
    pairwise_angular_distance, 
    pairwise_squared_norm_pytree,
)

from dibs.kernel.basic import BasicKernel

class JointMultiplicativeFrobeniusSEKernel(BasicKernel):
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, theta], [Z', theta']) = 
        scale * exp(- 1/h_z   ||Z - Z'||^2_F  
                    - 1/h_th  ||theta - theta'||^2_F )

    """

    def __init__(self, *, h_latent=1.0, h_theta=1.0, scale=1.0, singular_dim_theta=2, soft_graph_mask=False,
                 graph_embedding_representation=True):
        super(BasicKernel, self).__init__()

        if soft_graph_mask:
            raise NotImplementedError()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.scale = scale
        self.soft_graph_mask = soft_graph_mask
        self.singular_dim_theta = singular_dim_theta
        self.graph_embedding_representation = graph_embedding_representation

    # '''Computes squared exponential kernel'''
    def eval(self, *, x_latent, x_theta, y_latent, y_theta, h_latent=1.0, h_theta=1.0, alpha=1.0, axis=None):
        """Evaluates kernel function k(x, y) in batches
        where
            x_latent:   [N, d, k, 2]
            x_theta:    PyTree with leading dimensions [N, ...] 

            y_latent:   [N, d, k, 2]
            y_theta:    PyTree with leading dimensions [N, ...] 

        returns 
            pairwise kernel values over `axis` dimensions, 
            i.e. returns shape [N, M] where elt i,j is kernel of x[i] and y[j]
        """
        n_vars = x_latent.shape[1]

        # compute norms
        if self.graph_embedding_representation:
            latent_squared_norm = pairwise_squared_norm(x=x_latent, y=y_latent)
        else:
            latent_squared_norm = pairwise_squared_norm_mat(x=x_latent, y=y_latent)
            
        theta_squared_norm = pairwise_squared_norm_pytree(x_theta, y_theta, self.singular_dim_theta)

        # bandwidth (jax-consistent checking whether h is None in eval or object)
        h_latent_ = lax.cond(
            h_latent == -1.0,
            lambda _: self.h_latent,
            lambda _: h_latent,
            operand=None)

        h_latent_ = jnp.maximum(h_latent_, 1e-5)  # to avoid convergence to zero for median trick
        h_latent_ = stop_gradient(h_latent_)  # no grad from h

        h_theta_ = lax.cond(
            h_theta == -1.0,
            lambda _: self.h_theta,
            lambda _: h_theta,
            operand=None)

        h_theta_ = jnp.maximum(h_theta_, 1e-5)  # to avoid convergence to zero for median trick
        h_theta_ = stop_gradient(h_theta_)  # no grad from h

        # compute kernel
        return self.scale * jnp.exp(- (latent_squared_norm / h_latent_) - (theta_squared_norm / h_theta_))


class JointAdditiveFrobeniusSEKernel(BasicKernel):
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, theta], [Z', theta']) = 
        scale_z     * exp(- 1/h_z  ||Z - Z'||^2_F)
      + scale_theta * exp(- 1/h_th ||theta - theta'||^2_F )

    """

    def __init__(self, *, h_latent=1.0, h_theta=1.0, scale_latent=1.0, scale_theta=1.0, singular_dim_theta=2, soft_graph_mask=False,
                 graph_embedding_representation=True):
        super(BasicKernel, self).__init__()

        if soft_graph_mask:
            raise NotImplementedError()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.scale_latent = scale_latent
        self.scale_theta = scale_theta
        self.soft_graph_mask = soft_graph_mask
        self.singular_dim_theta = singular_dim_theta
        self.graph_embedding_representation = graph_embedding_representation

    # '''Computes squared exponential kernel'''
    def eval(self, *, x_latent, x_theta, y_latent, y_theta, h_latent=1.0, h_theta=1.0, alpha=1.0, axis=None):
        """Evaluates kernel function k(x, y) in batches
        where
            x_latent:   [N, d, k, 2]
            x_theta:    PyTree with leading dimensions [N, ...] 

            y_latent:   [N, d, k, 2]
            y_theta:    PyTree with leading dimensions [N, ...] 
        
        returns 
            pairwise kernel values over `axis` dimensions, 
            i.e. returns shape [N, M] where elt i,j is kernel of x[i] and y[j]
        """
        n_vars = x_latent.shape[1]
        
        # compute norms
        if self.graph_embedding_representation:
            latent_squared_norm = pairwise_squared_norm(x=x_latent, y=y_latent)
        else:
            latent_squared_norm = pairwise_squared_norm_mat(x=x_latent, y=y_latent)

        theta_squared_norm = pairwise_squared_norm_pytree(x_theta, y_theta, self.singular_dim_theta)

        # bandwidth (jax-consistent checking whether h is None in eval or object)
        h_latent_ = lax.cond(
            h_latent == -1.0,
            lambda _: self.h_latent,
            lambda _: h_latent,
            operand=None)

        h_latent_ = jnp.maximum(h_latent_, 1e-5)  # to avoid convergence to zero for median trick
        h_latent_ = stop_gradient(h_latent_)  # no grad from h

        h_theta_ = lax.cond(
            h_theta == -1.0,
            lambda _: self.h_theta,
            lambda _: h_theta,
            operand=None)

        h_theta_ = jnp.maximum(h_theta_, 1e-5)  # to avoid convergence to zero for median trick
        h_theta_ = stop_gradient(h_theta_)  # no grad from h

        # compute kernel
        return (self.scale_latent * jnp.exp(- latent_squared_norm / h_latent_)
              + self.scale_theta  * jnp.exp(- theta_squared_norm  / h_theta_ ))

