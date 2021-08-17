# import numpy as np
import autograd.numpy as np
from autograd import grad
from autograd.core import getval

import torch 
import torch.nn as nn


def pairwise_squared_norm(*, x, y, axis=None):
    """Computes pairwise squared euclidean norm
    where
        x:  [N, ...]
        y:  [M, ...]

    returns
        returns shape [N, M] where elt i,j is  ||x[i] - y[j]||_2^2
    """
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    assert(x.ndim == y.ndim and x.ndim >= 2)

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    if axis is None:
        axis = tuple(i for i in range(1, x.ndim))
    if not (len(axis) == 1 or len(axis) == 2):
        raise NotImplementedError

    # pairwise sum of squares along `axis` [N x M]
    pointwise_sq = np.sum(x ** 2, axis=axis).reshape(-1, 1) + np.sum(y ** 2, axis=axis)

    # pairwise product, i.e. outer product [N x M]
    if len(axis) == 2:
        outer_product = np.einsum('Nij,Mij->NM', x, y)
    else:
        outer_product = np.einsum('Ni,Mi->NM', x, y)

    # (x - y)^2 = x^2 + y^2 - 2 x y
    squared_norm = pointwise_sq - 2 * outer_product
    return squared_norm


def median_heuristic_from_squared_norm(x, is_norm=False):
    """Computes median heuristic based on _squared_ distance as used by SVGD
    where
        x :  squared distances
        is_norm: if `is_norm` is `True`, assumes sqrt(squard distance), i.e. actual norm, is passed

    returns
        h :  scaling factor in SE kernel as:     exp( - 1/h d(x, x'))
                where h = med(squared dist) / log n
    """

    # make sure all dimensions have the same length
    N_ = x.shape[0]

    med_square = np.median(x)
    if is_norm:
        med_square = med_square ** 2

    return med_square / np.log(N_ + 1)


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

    k(W, W') = exp(- 1/h ||W - W'||^2_F )

    """

    def __init__(self, *, h=-1.0):
        super(BasicKernel, self).__init__()

        # h = None indicates the median heuristic is used to set the bandwith
        self.h = h

    def compute_median_heuristic(self, x=None, y=None, squared_norms=None, norms=None, axis=None):
        """
        Computes `h` according to the median heuristic given the inputs
        """
        
        if squared_norms is not None:
            return median_heuristic_from_squared_norm(squared_norms)
        elif norms is not None:
            return median_heuristic_from_squared_norm(norms, is_norm=True)
        else:
            squared_norms = pairwise_squared_norm(x=x, y=y, axis=axis)
            return median_heuristic_from_squared_norm(squared_norms)


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
        squared_norm = pairwise_squared_norm(x=x, y=y, axis=axis)

        # bandwidth
        h_ = self.h if h == -1.0 else h 
        h__ = getval(self.compute_median_heuristic(squared_norms=squared_norm)) if h_ == -1.0 else h_
        h__ = getval(h__)

        # compute kernel
        return np.exp(- squared_norm / h__)


class StructuralHammingSquaredExponentialKernel(BasicKernel):
    """
    Squared exponential kernel, that computes the 
    exponentiated quadratic of the Hamming distance

    k(W, W') = exp(- 1/h HD(W, W'))

    """

    def __init__(self, *, h=-1.0):
        super(BasicKernel, self).__init__()

        # h = None indicates the median heuristic is used to set the bandwith
        self.h = h

    def compute_median_heuristic(self, x=None, y=None, squared_norms=None, norms=None, axis=None):
        """
        Computes `h` according to the median heuristic given the inputs
        """
        
        if squared_norms is not None:
            return median_heuristic_from_squared_norm(squared_norms)
        elif norms is not None:
            return median_heuristic_from_squared_norm(norms, is_norm=True)
        else:
            norms = pairwise_structural_hamming_distance(x=x, y=y, axis=axis)
            return median_heuristic_from_squared_norm(norms, is_norm=True)

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
        h_ = self.h if h == -1.0 else h 
        h__ = getval(self.compute_median_heuristic(norms=hamming)) if h_ == -1.0 else h_
        h__ = getval(h__)

        # compute kernel
        return np.exp(- hamming / h__)

