import random
import numpy as np
import igraph as ig
import scipy.optimize as sopt
from graph import *
from plot import *


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, progress_rate=0.25):
    """Solve 
    
        min_W L(W; X) + lambda1 ‖W‖_1 
        s.t. h(W) = 0 
        
    using augmented Lagrangian.

    Args:
        X (np.ndarray):         [n_samples, n_vars] data matrix
        lambda1 (float):        l1 sparsity penalty parameter
        loss_type (str):        loss type
        max_iter (int):         max num of dual ascent steps
        h_tol (float):          exit if |h(w_est)| <= htol 
        rho_max (float):        exit if rho >= rho_max
        w_threshold (float):    drop edge if |weight| < threshold
        progress_rate (float):  enforcing relative primal constraint improvement of h(W)

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    n_samples, n_vars = X.shape

    def _loss(W):
        """Evaluate value and gradient of loss."""
        if loss_type == 'l2':
            R = X - X @ W
            loss = (R ** 2).sum() / (2 * n_samples)
            gradient_loss = - X.T @ R / n_samples
        else:
            raise ValueError('unknown loss type')
        return loss, gradient_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        alpha = 1.0 / n_vars

        # (Yu et al. 2019: DAG-GNN; reformulation of original NOTEARS constraint)
        M = np.eye(n_vars) + alpha * W * W
        M_mult = np.linalg.matrix_power(M, n_vars - 1) # one less power, to have gradient readily available

        # h = (M_mult.T * M).sum() - n_vars # bit faster, but correctness less obvious
        h = np.trace(M_mult @ M) - n_vars

        gradient_h = M_mult.T * W * 2
        return h, gradient_h

    def _unflatten(w):
        """Reshapes doubled variables of shape (2 * n_vars * n_vars,) into (n_vars, n_vars)"""
        return (w[:n_vars * n_vars] - w[n_vars * n_vars:]).reshape((n_vars, n_vars))

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian."""
        W = _unflatten(w)
        loss, gradient_loss = _loss(W)
        h, gradient_h = _h(W)

        # augmented Lagrangian objective
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()

        # gradient of objective
        gradient_smooth = gradient_loss + (rho * h + alpha) * gradient_h

        gradient_obj = np.concatenate(
            [gradient_smooth + lambda1, 
            - gradient_smooth + lambda1], axis=None)

        return obj, gradient_obj

    def _checksol(sol):
        """Check L-BFGS went well"""
        pass 

    # center data
    X = X - np.mean(X, axis=0, keepdims=True)

    # initialization
    w_est = np.zeros(2 * n_vars * n_vars) # split into w_pos, w_neg for smooth l1-penalty gradient
    rho, alpha, h = 1.0, 0.0, np.inf

    # no self-loops not per so enforced (but should be learned via h(w) = 0 -> test for SGD later)
    bnds = [(0, None) for _ in range(2) for i in range(n_vars) for j in range(n_vars)]

    print(f't = {-1}  rho = {rho}  alpha = {alpha}')

    # main dual ascent loop
    for tt in range(max_iter):

        w_new, h_new = None, None

        # minimize primal objective with lowest possible rho 
        # s.t. h(W) is by factor `progress_rate` smaller than for previous iteration
        while rho < rho_max:
            
            # L-BFGS on augmented Lagrangian
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            _checksol(sol)
            w_new = sol.x
            obj = sol.fun
            h_new, _ = _h(_unflatten(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break

        w_est, h = w_new, h_new

        # dual ascent step
        alpha += rho * h

        print(f't = {tt}  rho = {rho}  alpha = {alpha} obj = {obj} h = {h}')

        # exit?
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _unflatten(w_est)

    # thresholding step
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

if __name__ == '__main__':

    n_samples, n_vars, n_edges, graph, noise_scale = 500, 20, 30, 'erdos', 0.5

    # binary adjacency marix of DAG
    B = simulate_dag(n_vars, n_edges, graph)

    # real weighted adjacency matrix as linear SEM parameters
    W_true = simulate_parameters(B, w_ranges=((-2, -0.5), (0.5, 2)))

    # sample from SEM
    X = simulate_linear_sem(W_true, n_samples, noise_scale=noise_scale)

    # NOTEARS
    lambda1, loss_type, max_dual_iter = 0.01, 'l2', 100
    h_tol, rho_max, w_threshold = 1e-8, 1e+16, 0.2

    W_est = notears_linear(X, lambda1, loss_type, max_iter=max_dual_iter, h_tol=h_tol, rho_max=rho_max, w_threshold=w_threshold)

    # plot
    # print('Ground truth')
    # print(W_true.round(1))
    # print('Estimated')
    # print(W_est.round(1))

    plot_matrix_comparison(W_true, W_est)

    # is DAG?
    print('DAG?: ', ig.Graph.Adjacency(W_est.tolist()).is_dag())
    print('No self-loops?: ', np.trace(W_est) == 0)


    
