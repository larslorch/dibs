import random
import numpy as np
import igraph as ig
import scipy.optimize as sopt
import torch
import torch.nn as nn
from graph import *
from plot import *


class NotearsLinearSEM(nn.Module):
    def __init__(self, *, n_vars, lambda1=0.0, bias=True):
        super(NotearsLinearSEM, self).__init__()
        self.n_vars = n_vars
        self.lambda1 = lambda1

        self.fc1 = nn.Linear(n_vars, n_vars, bias=bias)
        nn.init.zeros_(self.fc1.weight)

    def forward(self, x):  # [n, n_vars] -> [n, n_vars]
        """Forward pass, i.e. matrix multiplication with adjacency matrix."""
        x = self.fc1(x) # [n, n_vars]
        return x

    def get_adjacency_matrix(self):
        """Computes or simply returns (causal) adjacency matrix implied by model."""
        return self.fc1.weight.t()

    def h(self):
        """Compute value of acyclicity constraint."""
        alpha = 1.0 / self.n_vars
        W = self.get_adjacency_matrix()

        # (Yu et al. 2019: DAG-GNN; reformulation of original NOTEARS constraint)
        M = torch.eye(n_vars) + alpha * W * W
        h = torch.matrix_power(M, self.n_vars).trace() - self.n_vars 
        return h

    def reg(self):
        """Compute regularization."""
        W = self.get_adjacency_matrix()
        return self.lambda1 * W.abs().sum()


def opt_augmented_lagrangian(*, model, optimizer, loss_fct, X, alpha, rho, h_old, n_steps):

    # for early stopping
    exitbuff, exit_thres, min_steps = 20, 1e-6, 200
    last = torch.zeros(exitbuff)

    for ii in range(n_steps):

        optimizer.zero_grad()

        # model predictions
        pred = model(X)
        
        # objective components
        loss = loss_fct(X, pred)
        h = model.h()
        reg = model.reg()

        # augmented Lagrangian objective
        obj = loss + 0.5 * rho * h * h + alpha * h + reg

        # optimize
        obj.backward()
        optimizer.step()

        # exit?
        if ii > max(exitbuff, min_steps) and (last.mean() - obj.detach()).abs() < exit_thres:
            break
        last[ii % exitbuff] = obj.detach()

    print(f'rho = {rho} ii = {ii}  loss = {loss}  h =  {h.detach().item()} alpha = {alpha}')

    return model, optimizer, h.detach()


def notears(*, model, optimizer, X, loss_fct, n_dual_steps=100, max_primal_steps=1000, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, progress_rate=0.25):
    
    n_samples, n_vars = X.shape

    # center data
    X = X - X.mean(dim=0, keepdim=True)

    # initialization
    state = {
        'model': model.state_dict(),
		'optimizer': optimizer.state_dict(),
    }
    rho, alpha, h = 1.0, 0.0, np.inf

    # main dual ascent loop
    for tt in range(n_dual_steps):

        # minimize primal objective with lowest possible rho 
        # s.t. h(W) is by factor `progress_rate` smaller than for previous iteration
        while rho < rho_max:

            # re-init model and optimizer to previous setting
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
                    
            # optimize augmented Lagrangian (primal)
            model, optimizer, h_new = opt_augmented_lagrangian(
                model=model, 
                optimizer=optimizer,
                loss_fct=loss_fct, 
                X=X, 
                alpha=alpha, 
                rho=rho, 
                h_old=h, 
                n_steps=max_primal_steps)

            if h_new > progress_rate * h:
                rho *= 10
            else:
                break
        
        # update new state
        h = h_new
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # dual ascent step
        alpha += rho * h

        # exit?
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.get_adjacency_matrix().detach()

    # thresholding step
    W_est[W_est.abs() < w_threshold] = 0
    return W_est

if __name__ == '__main__':

    torch.set_default_tensor_type(torch.DoubleTensor)

    n_samples, n_vars, n_edges, graph, noise_scale = 500, 20, 30, 'erdos', 0.5

    # binary adjacency marix of DAG
    B = simulate_dag(n_vars, n_edges, graph)

    # real weighted adjacency matrix as linear SEM parameters
    W_true = simulate_parameters(B, w_ranges=((-2, -0.5), (0.5, 2)))

    # sample from SEM
    X = simulate_linear_sem(W_true, n_samples, noise_scale=noise_scale)

    # to torch
    W_true = torch.from_numpy(W_true)
    X = torch.from_numpy(X)

    # model
    lambda1 = 0.01
    max_dual_iter = 20
    max_primal_steps = 10000
    loss_fct = nn.MSELoss()

    model = NotearsLinearSEM(n_vars=n_vars, lambda1=lambda1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
    
    
    h_tol, rho_max, w_threshold, progress_rate = 1e-8, 1e+16, 0.2, 0.25

    W_est = notears(
        model=model, 
        optimizer=optimizer, 
        X=X, 
        loss_fct=loss_fct, 
        n_dual_steps=max_dual_iter, 
        max_primal_steps=max_primal_steps, 
        h_tol=h_tol, 
        rho_max=rho_max, 
        w_threshold=w_threshold, 
        progress_rate=progress_rate)

    # plot
    # print('Ground truth')
    # print(W_true.round())
    # print('Estimated')
    # print(W_est.round())

    plot_matrix_comparison(W_true.numpy(), W_est.numpy())

    # is DAG?
    print('DAG?: ', ig.Graph.Adjacency(W_est.tolist()).is_dag())
    print('No self-loops?: ', np.trace(W_est) == 0)


    
