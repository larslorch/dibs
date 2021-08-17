
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import stop_gradient
import jax.lax as lax

from dibs.utils.func import (
    median_heuristic_from_squared_norm, 
    pairwise_squared_norm, 
    pairwise_squared_norm_mat,
    pairwise_structural_hamming_distance, 
    pairwise_angular_distance, 
)

class BasicKernel:
    """
    Basic kernel model structure 
    """

    def __init__(self):
        super(BasicKernel, self).__init__()
        pass

    def eval(self, *, x, y):
        """Evaluates kernel function k(x, y) for graphs g, h
        where
            x:  [n_vars, n_vars] adjacency matrix of graph G
            y:  [n_vars, n_vars] adjacency matrix of graph H
        
        """
        raise NotImplementedError


class FrobeniusSquaredExponentialKernel(BasicKernel):
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k(W, W') = scale * exp(- 1/h ||W - W'||^2_F )

    """

    def __init__(self, *, h=-1.0, scale=1.0, graph_embedding_representation=True):
        super(BasicKernel, self).__init__()

        # h = -1.0 indicates the median heuristic is used to set the bandwith
        self.h = h
        self.scale = scale
        self.graph_embedding_representation = graph_embedding_representation

    def compute_median_heuristic(self, x=None, y=None, squared_norms=None, norms=None, axis=None):
        """
        Computes `h` according to the median heuristic given the inputs
        """
        
        if squared_norms is not None:
            return median_heuristic_from_squared_norm(squared_norms, False)
        elif norms is not None:
            return median_heuristic_from_squared_norm(norms, True)
        else:
            squared_norms = pairwise_squared_norm(x=x, y=y, axis=axis)
            return median_heuristic_from_squared_norm(squared_norms, False)


    # '''Computes squared exponential kernel'''
    def eval(self, *, x, y, h=-1.0, axis=None):
        """Evaluates kernel function k(x, y) in batches
        where
            x:  [N, ...] 
            y:  [M, ...] 
            h:  float       (h = -1 indicates median trick is applied)
        
        returns 
            pairwise kernel values over `axis` dimensions, 
            i.e. returns shape [N, M] where elt i,j is kernel of x[i] and y[j]
        """

        # compute norm
        if self.graph_embedding_representation:
            squared_norm = pairwise_squared_norm(x=x, y=y)
        else:
            squared_norm = pairwise_squared_norm_mat(x=x, y=y)

        # bandwidth (jax-consistent checking whether h is None in eval or object)
        h_ = lax.cond(
            h == -1.0,
            lambda _: self.h,
            lambda _: h,
            operand=None)

        h__ = lax.cond(
            h_ == -1.0,
            lambda _: self.compute_median_heuristic(squared_norms=squared_norm),
            lambda _: h_,
            operand=None)

        h__ = jnp.maximum(h__, 1e-5)  # to avoid convergence to zero for median trick
        h__ = stop_gradient(h__)  # no grad from h

        # compute kernel
        return self.scale * jnp.exp(- squared_norm / h__)


class StructuralHammingSquaredExponentialKernel(BasicKernel):
    """
    Squared exponential kernel, that computes the 
    exponentiated quadratic of the Structural Hamming distance

    k(W, W') = exp(- 1/h HD(W, W'))

    """

    def __init__(self, *, h=-1.0):
        super(BasicKernel, self).__init__()

        # h = -1.0 indicates the median heuristic is used to set the bandwith
        self.h = h

    def compute_median_heuristic(self, x=None, y=None, squared_norms=None, norms=None, axis=None):
        """
        Computes `h` according to the median heuristic given the inputs
        """
        
        if squared_norms is not None:
            return median_heuristic_from_squared_norm(squared_norms, False)
        elif norms is not None:
            return median_heuristic_from_squared_norm(norms, True)
        else:
            norms = pairwise_structural_hamming_distance(x=x, y=y, axis=axis)
            return median_heuristic_from_squared_norm(norms, True)

    # '''Computes squared exponential kernel'''
    def eval(self, *, x, y, h=-1.0, axis=None):
        """Evaluates kernel function k(x, y) in batches
        where
            x:  [N, ...] 
            y:  [M, ...] 
        
        returns 
            pairwise kernel values over `axis` dimensions, 
            i.e. returns shape [N, M] where elt i,j is kernel of x[i] and y[j]
        """

        # compute norm
        hamming = pairwise_structural_hamming_distance(x=x, y=y, axis=axis)

        # bandwidth
        h_ = lax.cond(
            h == -1.0,
            lambda _: self.h,
            lambda _: h,
            operand=None)

        h__ = lax.cond(
            h_ == -1.0,
            lambda _: self.compute_median_heuristic(norms=hamming),
            lambda _: h_,
            operand=None)

        h__ = stop_gradient(h__)  # no grad from h

        # compute kernel
        return jnp.exp(- hamming / h__)


class AngularSquaredExponentialKernel(BasicKernel):
    """
    Exponential kernel that uses angle as distance measure 
    Assumes particles of the form used for `DotProductGraphSVGD`
    """

    def __init__(self, *, h=-1.0, scale=1.0):
        super(BasicKernel, self).__init__()

        # h = -1.0 indicates the median heuristic is used to set the bandwith
        self.h = h
        self.scale = scale

    def compute_median_heuristic(self, x=None, y=None, squared_norms=None, norms=None, axis=None):
        """
        Computes `h` according to the median heuristic given the inputs
        """

        if squared_norms is not None:
            return median_heuristic_from_squared_norm(squared_norms, False)
        elif norms is not None:
            return median_heuristic_from_squared_norm(norms, True)
        else:
            angular = pairwise_angular_distance(x=x, y=y)
            return median_heuristic_from_squared_norm(angular, True)

    # '''Computes squared exponential kernel'''
    def eval(self, *, x, y, h=-1.0):
        """Evaluates kernel function k(x, y) in batches
        where
            x:  [N, d, k, 2]
            y:  [M, d, k, 2]
            h:  float       (h = -1 indicates median trick is applied)

        returns
            pairwise kernel values over `axis` dimensions,
            i.e. returns shape [N, M] where elt i,j is kernel of x[i] and y[j]
        """

        # compute norm
        angular = pairwise_angular_distance(x=x, y=y)

        # bandwidth
        h_ = lax.cond(
            h == -1.0,
            lambda _: self.h,
            lambda _: h,
            operand=None)

        h__ = lax.cond(
            h_ == -1.0,
            lambda _: self.compute_median_heuristic(norms=angular),
            lambda _: h_,
            operand=None)

        h__ = stop_gradient(h__)  # no grad from h

        # compute kernel
        return self.scale * jnp.exp(- angular / h__)

