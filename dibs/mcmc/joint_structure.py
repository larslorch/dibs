import time
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import igraph as ig
import itertools
import torch
import tqdm
import collections
from datetime import datetime

import jax.numpy as jnp
from jax import random 
from jax.tree_util import tree_map, tree_multimap

from dibs.utils.graph import get_mat_idx, mat_to_graph, graph_to_mat
from dibs.utils.tree import tree_key_split, tree_zip_leading

from dibs.mcmc.structure import StructureMCMC

class MHJointStructureMCMC(StructureMCMC):
    """
    Modified Structure MCMC as by Giudici and Castelo (2003), also inferring parameters jointly
    """

    def __init__(self, *, n_vars, only_non_covered=False, verbose=True, theta_prop_sig=1.0, ave_weight=0.99):
        super(MHJointStructureMCMC, self).__init__(n_vars=n_vars, only_non_covered=only_non_covered, verbose=verbose)

        # additional setup
        self.theta_prop_sig = theta_prop_sig
        self.ave_weight = ave_weight

    def propose_theta(self, *, key, theta):
        """
        Proposal q(theta' | theta) 
        For each theta leaf, add Gaussian random walk propsal
        """
        
        subkey_tree = tree_key_split(key, theta)
        return tree_multimap(lambda subk, leaf: leaf + self.theta_prop_sig * random.normal(subk, shape=leaf.shape), subkey_tree, theta)

    def sample(self, *, key, n_samples, log_joint_target, theta_shape, burnin=1e4, thinning=10, g_init=None, 
        verbose_indication=0, return_matrices=True):
        """
        Generates `n_samples` graph samples  after `burning` steps, sampling every `thinning` steps.
        If `g_init` is not provided, starts with empty graph.

        log_joint_target:   (igraph.Graph, [n_vars, n_vars]) -> propto log p(G, theta | D) 
        theta_shape:        PyTree of param.shape sat leaves

        """
        last_verbose_indication = 1
        t_start = time.time()

        # I[i, j] = 1 iff directed edge j -> i exists
        # A[i, j] = 1 iff there exists a directed path j ~> ... ~> i
        inc = np.zeros((self.n_vars, self.n_vars))
        anc = np.zeros((self.n_vars, self.n_vars))
        g = ig.Graph.Weighted_Adjacency(inc.T.tolist())
        theta = tree_map(lambda leaf_shape: jnp.zeros(leaf_shape), theta_shape)
        
        if g_init is not None:
            # build matrices for provided graph
            g = g_init
            for e in g.es:
                i, j = e.tuple 
                inc, anc = self.apply_mat_move(inc=inc, anc=anc, m=('add', i, j))

        # assert(g.is_dag())

        log_prob = log_joint_target(graph_to_mat(g), theta)
        
        moves = None
        g_samples = []
        theta_samples = []
        ave_alpha = 0.5
        cumul_alpha = 0.0
        n_steps = int(burnin + thinning * n_samples)
        titer = tqdm.tqdm(range(n_steps),
                          desc=f'MHJointStructureMCMC | ave accept: {ave_alpha:1.4f}',
                          disable=not self.verbose)

        for t in titer:

            '''Graph proposal'''
            # compute valid moves (neighborhood) only if accepted last time
            if moves is None:
                moves = self.get_valid_moves(g=g, inc=inc, anc=anc)

            # propose move
            key, subk = random.split(key)
            u = random.uniform(subk)

            key, subk = random.split(key)
            i = random.choice(subk, len(moves))

            move = moves[i]
            g_new = self.apply_move(g=g.copy(), m=move)

            '''Theta proposal'''
            key, subk = random.split(key)
            theta_new = self.propose_theta(key=subk, theta=theta)
            
            '''Metropolis hastings'''
            log_prob_new = log_joint_target(graph_to_mat(g_new), theta_new)
            alpha = min(0, log_prob_new - log_prob)

            if np.log(u) < alpha:

                # accept move
                g = g_new
                theta = theta_new
                log_prob = log_prob_new
                inc, anc = self.apply_mat_move(inc=inc, anc=anc, m=move)
                moves = None

            # acceptance rate average
            ave_alpha = self.ave_weight * ave_alpha + (1 - self.ave_weight) * np.exp(alpha).item()
            if t >= burnin:
                cumul_alpha += np.exp(alpha).item()
            if not t % min(thinning, 200) and self.verbose:
                titer.set_description(f'MHJointStructureMCMC | ave accept: {ave_alpha:1.4f}')

            # collect samples
            if not t % thinning and t >= burnin:
                g_samples.append(g.copy())
                theta_samples.append(theta.copy())

            # verbose progress
            if verbose_indication > 0:
                if t >= (last_verbose_indication * n_steps// verbose_indication):
                    print(
                        f'MHJointStructureMCMC   {t} / {n_steps} [{(100 * t / n_steps):3.1f} % ' + 
                        f'| {((time.time() - t_start)/60):.0f} min | {datetime.now().strftime("%d/%m %H:%M")}]',
                        flush=True
                    )
                    last_verbose_indication += 1

        # randomly permute
        assert(len(g_samples) == n_samples and len(theta_samples) == n_samples)
        key, subk = random.split(key)
        perm = random.permutation(subk, n_samples)
        g_samples = [g_samples[i] for i in perm]

        self.ave_alpha_after_burnin = cumul_alpha / (thinning * n_samples)
        if self.verbose:
            print(f'Ave. acceptance rate after burnin {self.ave_alpha_after_burnin:1.6f}')

        # convert to matrices
        if return_matrices:
            mats = []
            for g_ in g_samples:
                mats.append(np.array(g_.get_adjacency().data))
            return jnp.array(mats), tree_zip_leading(theta_samples)

        else:
            return g_samples, tree_zip_leading(theta_samples)



class GibbsJointStructureMCMC(StructureMCMC):
    """
    Modified Structure MCMC as by Giudici and Castelo (2003), also inferring parameters jointly
    """

    def __init__(self, *, n_vars, only_non_covered=False, verbose=True, theta_prop_sig=1.0, ave_weight=0.99):
        super(GibbsJointStructureMCMC, self).__init__(n_vars=n_vars, only_non_covered=only_non_covered, verbose=verbose)

        # additional setup
        self.theta_prop_sig = theta_prop_sig
        self.ave_weight = ave_weight

    def propose_theta(self, *, key, theta):
        """
        Proposal q(theta' | theta) 
        For each theta leaf, add Gaussian random walk propsal
        """
        
        subkey_tree = tree_key_split(key, theta)
        return tree_multimap(lambda subk, leaf: leaf + self.theta_prop_sig * random.normal(subk, shape=leaf.shape), subkey_tree, theta)

    def sample(self, *, key, n_samples, log_joint_target, theta_shape, burnin=1e4, thinning=10, g_init=None, 
        verbose_indication=0, return_matrices=True):
        """
        Generates `n_samples` graph samples  after `burning` steps, sampling every `thinning` steps.
        If `g_init` is not provided, starts with empty graph.

        log_joint_target: (igraph.Graph, [n_vars, n_vars]) -> propto log p(G, theta | D) 

        """
        last_verbose_indication = 1
        t_start = time.time()

        # I[i, j] = 1 iff directed edge j -> i exists
        # A[i, j] = 1 iff there exists a directed path j ~> ... ~> i
        inc = np.zeros((self.n_vars, self.n_vars))
        anc = np.zeros((self.n_vars, self.n_vars))
        g = ig.Graph.Weighted_Adjacency(inc.T.tolist())
        theta = tree_map(lambda leaf_shape: jnp.zeros(leaf_shape), theta_shape)

        if g_init is not None:
            # build matrices for provided graph
            g = g_init
            for e in g.es:
                i, j = e.tuple 
                inc, anc = self.apply_mat_move(inc=inc, anc=anc, m=('add', i, j))

        # assert(g.is_dag())

        log_prob = log_joint_target(graph_to_mat(g), theta)
        
        moves = None
        g_samples = []
        theta_samples = []
        ave_alpha = 0.5
        cumul_alpha = 0.0
        n_steps = int(burnin + thinning * n_samples)
        titer = tqdm.tqdm(range(n_steps), 
                          desc=f'GibbsJointStructureMCMC | ave accept: {ave_alpha:1.4f}',
                          disable=not self.verbose)

        for t in titer:

            '''Graph proposal'''
            # compute valid moves (neighborhood) only if accepted last iteration
            if moves is None:
                moves = self.get_valid_moves(g=g, inc=inc, anc=anc)

            # propose move
            key, subk = random.split(key)
            u_g = random.uniform(subk)

            key, subk = random.split(key)
            i = random.choice(subk, len(moves))

            move = moves[i]
            g_new = self.apply_move(g=g.copy(), m=move)

            # Metropolis-Hastings accept/reject
            log_prob_new = log_joint_target(graph_to_mat(g_new), theta)
            alpha_g = min(0, log_prob_new - log_prob)

            if np.log(u_g) < alpha_g:
                # accept move
                g = g_new
                log_prob = log_prob_new
                inc, anc = self.apply_mat_move(inc=inc, anc=anc, m=move)
                moves = None

            '''Theta proposal'''
            key, subk = random.split(key)
            theta_new = self.propose_theta(key=subk, theta=theta)

            key, subk = random.split(key)
            u_theta = random.uniform(subk)

            # Metropolis-Hastings accept/reject
            log_prob_new = log_joint_target(graph_to_mat(g), theta_new)
            alpha_theta = min(0, log_prob_new - log_prob)

            if np.log(u_theta) < alpha_theta:
                # accept move
                theta = theta_new
                log_prob = log_prob_new

            # acceptance rate average
            ave_alpha = self.ave_weight * ave_alpha + (1 - self.ave_weight) * np.exp(alpha_theta).item()
            if t >= burnin:
                cumul_alpha += np.exp(alpha_theta).item()
            if not t % min(thinning, 200) and self.verbose:
                titer.set_description(f'GibbsJointStructureMCMC | ave accept: {ave_alpha:1.4f}')

            # collect samples
            if not t % thinning and t >= burnin:
                g_samples.append(g.copy())
                theta_samples.append(theta.copy())

            # verbose progress
            if verbose_indication > 0:
                if t >= (last_verbose_indication * n_steps// verbose_indication):
                    print(
                        f'GibbsJointStructureMCMC   {t} / {n_steps} [{(100 * t / n_steps):3.1f} %' + 
                        f'| {((time.time() - t_start)/60):.0f} min | {datetime.now().strftime("%d/%m %H:%M")}]',
                        flush=True
                    )
                    last_verbose_indication += 1

        # randomly permute
        assert(len(g_samples) == n_samples and len(theta_samples) == n_samples)
        key, subk = random.split(key)
        perm = random.permutation(subk, n_samples)
        g_samples = [g_samples[i] for i in perm]
        theta_samples = [theta_samples[i] for i in perm]

        self.ave_alpha_after_burnin = cumul_alpha / (thinning * n_samples)
        if self.verbose:
            print(f'Ave. acceptance rate after burnin {self.ave_alpha_after_burnin:1.6f}')

        # convert to matrices
        if return_matrices:
            mats = []
            for g_ in g_samples:
                mats.append(np.array(g_.get_adjacency().data))
            return jnp.array(mats), tree_zip_leading(theta_samples)

        else:
            return g_samples, tree_zip_leading(theta_samples)
