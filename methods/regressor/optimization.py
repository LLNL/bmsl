# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2019-06-27 09:49:00
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2019-06-27 10:21:26
"""
ADMM module.
WARNING: the code is a little coupled with agtlm and
proximal_operator.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

# import codes.optimization.proximal_operator as prxopt
# from sksparse.cholmod import cholesky as spcholesky

ABS_TOL = 1e-4
REL_TOL = 1e-3
VERBOSE = False
PLOT_COST = False

MAX_ITER = 130
TOLERANCE = 10e-6


class ADMM:
    """
    ADMM will optimize a constrained lasso problem where w >= 0.
        f(x) = 1/2||y - Xw||^2 + lamb||w||_1
        where w >= 0

    Args:
        lamb (float): regularization parameter.
        rho (int): augmented Lagrangian parameter.
        alpha (float): ADMM relaxation parameter.
        cache (bool) : use cholesky decomposition and keep cache
                        ** vary ** must be false.
        vary (bool):  vary rho according to ADMM paper.
        max_iter (int) : max number of iterations.
        sparse (bool): use sparse implementation.

    Attributes:
        tau_incr
        tau_decr
        mu
        hist

    Methods:
        fit

    References:
    Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011)
    Distributed optimization and statistical learning via the alternating
    direction method of multipliers. Foundations and Trends in Machine
    Learning, 3(1), 1-122.
    """

    def __init__(self, lamb, rho=1.5, alpha=1.2, cache=True, vary=False,
                 max_iter=100, sparse=False):
        assert lamb >= 0
        self.lamb = lamb
        assert rho >= 0
        self.rho = rho
        self.max_iter = max_iter
        self.cache = cache
        self.vary = vary  # leave it false when using cache factorization
        self.sparse = sparse
        self.tau_incr = 2
        self.tau_decr = 2
        self.mu = 10
        self.alpha = alpha  # over-relaxation parameter
        self.hist = []

    def fit(self, X, y, w=None):
        """
            Run Forest!

            Args:
                X (m x n): dataset
                y (n, 1): labels (real)

            Returns:
                w (n): estimated parameters
                t (float): elapsed time
        """
        assert X.shape[0] == y.shape[0]
        y.reshape(len(y),)

        # initializing
        if w is None:
            w = np.zeros((X.shape[1]))
        z_1 = np.zeros((X.shape[1], 1))
        z_2 = np.zeros((X.shape[1], 1))
        u_1 = np.zeros((X.shape[1], 1))  # w - z_1
        u_2 = np.zeros((X.shape[1], 1))  # w - z_2

        # ADMM loop
        keep_going = True
        n_iter = 0
        hist = list()
        prim_res = list()
        start = time.time()

        # cache factorization and other pre-computable terms
        if self.cache:
            if self.sparse:
                temp = scipy.sparse.csc_matrix(np.dot(X.T, X))
                L = spcholesky(temp, beta=self.rho)
            else:
                L = cholesky(np.dot(X.T, X) + self.rho * np.eye(X.shape[1]),
                             lower=True)
            q = np.dot(X.T, y)
            q.shape = (q.size, 1)

        while keep_going and (n_iter <= self.max_iter):
            n_iter += 1
            # w-update step
            if self.cache:
                g = 0.5 * self.rho * (z_1 + z_2 - u_1 - u_2) + q
                # back-solving
                if self.sparse:
                    w = L.solve_Lt(L.solve_A(g))
                else:
                    w = scipy.linalg.solve(L.T, scipy.linalg.solve(L, g))
            else:
                def fun(var):
                    temp_1 = (0.5 * self.rho) * np.linalg.norm(w - z_1 + u_1)
                    temp_2 = (0.5 * self.rho) * np.linalg.norm(w - z_2 + u_2)
                    return f(X, y, var) + temp_1 + temp_2
                opt_resul = minimize(fun, w, method='bfgs')
                w = opt_resul.x
                w.shape = (w.shape[0], 1)

            # z_i update step - these sub-steps can be done in parallel
            z_1_old = z_1.copy()
            z_2_old = z_2.copy()

            x1_hat = self.alpha * w + (1 - self.alpha) * z_1_old
            z_1 = shrinkage(x1_hat + u_1, self.lamb / self.rho)

            x2_hat = self.alpha * w + (1 - self.alpha) * z_2_old
            z_2 = proximal_constrained(x2_hat + u_2)  # proj em C

            # dual variables update step
            u_1 = u_1 + (x1_hat - z_1)  # dual variable for the 1st constraint
            u_2 = u_2 + (x2_hat - z_2)  # dual variable for the 2nd constraint

            # monitor costs over time
            if PLOT_COST:
                hist.append(((np.dot(X, w) - y) ** 2).sum() +
                            self.lamb * (abs(w).sum()))

            # critério de parada
            primal_res = 0.5 * ((w - z_1) + (w - z_2))
            primal_res_norm = np.linalg.norm(primal_res)

            # monitor primal residual over time
            prim_res.append(primal_res_norm)

            dual_res = 0.5 * ((z_1 - z_1_old) + (z_2 - z_2_old))
            dual_res_norm = np.linalg.norm(self.rho * dual_res)
            eps_pri = np.sqrt(X.shape[1]) * ABS_TOL + REL_TOL * \
                max(np.linalg.norm(w),
                    0.5 * (np.linalg.norm(-z_1) + np.linalg.norm(-z_2)))
            eps_dual = np.sqrt(X.shape[1]) * ABS_TOL + REL_TOL * 0.5 * \
                (np.linalg.norm(u_1) + np.linalg.norm(u_2))
            # variação de rho, baseado em Boyd, 2011, ADMM, pg. 20
            if self.vary:
                if primal_res_norm > self.mu * dual_res_norm:
                    self.rho *= self.tau_incr
                    u_1 = u_1 / self.tau_incr
                    u_2 = u_2 / self.tau_incr
                elif dual_res_norm > self.mu * primal_res_norm:
                    self.rho *= (1 / self.tau_decr)
                    u_1 = u_1 * self.tau_decr
                    u_2 = u_2 * self.tau_decr

            # print informações de convergência?
            if VERBOSE:
                message = """|ADMM it.{} |r_norm: {:8.3f} |eps_pri: {:1.3f}
                          |s_norm: {:8.3f} |eps_dual: {:1.3f} |obj: {:8.3f}"""

                value = ((1 / 2) * np.linalg.norm(np.dot(X, w) - y) ** 2 +
                         self.lamb * np.sum(abs(w)))
                print(message.format(n_iter,
                                     primal_res_norm, eps_pri, dual_res_norm,
                                     eps_dual,
                                     value))

            if primal_res_norm <= eps_pri and dual_res_norm <= eps_dual:
                keep_going = False
                if VERBOSE:
                    print('Primal dual stopping criterion met.')

        # make sure it's feasible (w_i >= 0, i=1,...,d)
        # verify whether the constraint satisfiability is required when doing
        # early stop (few max_iters). If not, maybe is better to comment it out
        # as this project might be to harsh for early-stage solutions
        self.hist = hist
        w_final = z_2.copy()

        if PLOT_COST:
            # cost function
            plt.subplot(2, 1, 1)
            plt.plot(hist)
            plt.xlabel('iterations')
            plt.ylabel('Cost function')

            # primal residual: average of both residuals (one for constraint)
            plt.subplot(2, 1, 2)
            plt.semilogy(prim_res)
            plt.xlabel('iterations')
            plt.ylabel('Primal residual - Avg')
            plt.show(block=False)
        return w_final.flatten(), time.time() - start


class ADMM_Lasso:
    """
    ADMM for Lasso, which minimizes the function:
        f(x) = 1/2||y - Xw||^2 + lamb||w||_1

    Args:
        lamb (float): regularization parameter.
        rho (int): augmented Lagrangian parameter.
        alpha (float): ADMM relaxation parameter.
        cache (bool) : use cholesky decomposition and keep cache
                        ** vary ** must be false.
        vary (bool):  vary rho according to ADMM paper.
        max_iter (int) : max number of iterations.

    Attributes:
        tau_incr
        tau_decr
        mu
        hist

    Methods:
        fit

    References:
    Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011)
    Distributed optimization and statistical learning via the alternating
    direction method of multipliers. Foundations and Trends in Machine
    Learning, 3(1), 1-122.
    """

    def __init__(self, lamb, rho=1.5, alpha=1.2, cache=True, vary=False, max_iter=300):
        self.lamb = lamb
        self.rho = rho
        self.alpha = alpha
        self.history = {'objval': np.zeros(max_iter),
                        'r_norm': np.zeros(max_iter),
                        's_norm': np.zeros(max_iter),
                        'eps_pri': np.zeros(max_iter),
                        'eps_dual': np.zeros(max_iter)}
        self.max_iter = max_iter
        self.cache = cache
        self.vary = vary  # leave it false if cache factorization is enabled
        self.tau_incr = 2
        self.tau_decr = 2
        self.mu = 10
        self.alpha = alpha  # over-relaxation parameter

    def fit(self, X, y, w=None):
        """
        Run Forest!

        Args:
            X (m x n): dataset
            y (m, 1): labels (real)
            w (n): warm start

        Returns:
            w (n): estimated parameters
            t (float): elapsed time
        """
        assert X.shape[0] == y.shape[0]
        n = X.shape[1]
        if w is None:
            w = np.random.randn(n)

        # save matrix-vector multiply
        Xty = np.dot(X.T, y)

        # x = np.random.randn(n)
        # z = np.random.randn(n)
        # u = np.random.randn(n)
        x = np.ones(n)
        z = np.ones(n)
        u = np.ones(n)

        if self.cache:
            L = cholesky(np.dot(X.T, X) + self.rho * np.eye(X.shape[1]), lower=True)

        if VERBOSE:
            print('{:3}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\n'.format(
                'iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective'))

        # iterações do algoritmo
        for k in range(self.max_iter):
            # atualização do x
            if self.cache:
                q = Xty.flatten() + self.rho * (z - u)
                x = spsolve(L.T, spsolve(L, q.T))
            else:
                def fun(var):
                    return f(X, y, var) + (0.5 * self.rho) * np.linalg.norm(x + z + u)
                opt_resul = minimize(fun, w, method='bfgs')
                if not opt_resul.success:
                    print(opt_resul.message)
                    raise RuntimeException('bfgs division by zero: {}'.format(opt_resul.message))
                x = opt_resul.x

            # atualização do z
            z_old = z.copy()
            x_hat = self.alpha * x + (1 - self.alpha) * z_old
            z = shrinkage(x_hat + u, self.lamb / self.rho)

            # atualização do u
            u = u + (x_hat - z)

            # diagnóstico, relatórios e verificação de critérios de parada
            self.history['objval'][k] = objective(X, y.flatten(), x, z, self.lamb)
            self.history['r_norm'][k] = np.linalg.norm(0.5 * (x - z))
            self.history['s_norm'][k] = np.linalg.norm(-self.rho * (z - z_old))
            self.history['eps_pri'][k] = np.sqrt(n) * ABS_TOL + REL_TOL * \
                max(np.linalg.norm(x), np.linalg.norm(-z))
            self.history['eps_dual'][k] = np.sqrt(n) * ABS_TOL + REL_TOL * \
                np.linalg.norm(self.rho * u)

            # variação de rho, baseado em Boyd, 2011, ADMM, pg. 20
            primal_res_norm = self.history['r_norm'][k]
            dual_res_norm = self.history['s_norm'][k]
            # np.linalg.norm(self.rho * 0.5 * (z - z_old))
            if self.vary:
                if primal_res_norm > self.mu * dual_res_norm:
                    self.rho *= self.tau_incr
                    u = u / self.tau_incr
                elif dual_res_norm > self.mu * primal_res_norm:
                    self.rho *= (1 / self.tau_decr)
                    u = u * self.tau_decr

            if VERBOSE:
                print('{:3}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(k,
                                                                                      self.history['r_norm'][k],
                                                                                      self.history['eps_pri'][k],
                                                                                      self.history['s_norm'][k],
                                                                                      self.history['eps_dual'][k],
                                                                                      self.history['objval'][k]))

            if (self.history['r_norm'][k] < self.history['eps_pri'][k] and
                    self.history['s_norm'][k] < self.history['eps_dual'][k]):
                break

        if PLOT_COST:
            # cost function
            plt.subplot(2, 1, 1)
            plt.plot(self.history['objval'])
            plt.xlabel('iterations')
            plt.ylabel('Cost function')

            # primal residual: average of both residuals (one for constraint)
            plt.subplot(2, 1, 2)
            plt.semilogy(self.history['r_norm'])
            plt.xlabel('iterations')
            plt.ylabel('Primal residual')
            plt.show(block=False)

        return z  # x_hat


def objective(A, b, x, z, lambda_1):
    """
    Objective function.

    :param A: 
    :param b: 
    :param x: 
    :param z: 
    :param lambda_1: 
    """
    term1 = np.power(np.linalg.norm(np.dot(A, x) - b, ord=2), 2)
    term2 = np.linalg.norm(z, ord=1)
    p = (1 / (2 * A.shape[0])) * term1 + lambda_1 * term2
    return p


def f(A, b, x):
    """
    f function

    :param A: 
    :param b: 
    :param x: 
    :returns: 
    :rtype: 

    """
    term1 = np.power(np.linalg.norm(np.dot(A, x) - b, ord=2), 2)
    p = (1 / (2 * A.shape[0])) * term1
    return p


def factor(A, rho):
    """
    Factorization method.

    :param A: 
    :param rho: 
    :returns: 
    :rtype: 
    """
    [m, n] = A.shape
    L = None
    if m >= n:  # if skinny
        L = cholesky(np.dot(A.T, A) + rho * np.eye(n), lower=True)
    else:  # if flat
        L = cholesky(np.eye(m) + 1 / rho * np.dot(A, A.T), lower=True)
    U = L.T.copy()
    return (L, U)


"""
Description: Implements gradient descent with inexact backtracking line search for the
step size.
"""


class GradientDescent(object):
    """
    Fista will optimize a group lasso problem with the given cost_function,
    cost_grad, and prox in the object cv_fun.

    Args:
        alpha
        beta

    Attributes:
        history

    Methods:
        fit
    """

    def __init__(self, alpha=0.15, beta=0.8):
        assert alpha > 0
        assert alpha <= 0.5
        assert beta > 0
        assert beta <= 1
        self.alpha = alpha
        self.beta = beta
        self.history = {'obj_val': list(), 'step_vals': list()}

    def fit(self, X, y, cost, grad, w=None, max_iter=MAX_ITER):
        """Fitsssss
        Args:
            :X: Samples
            :y: Labels
            :cost: Cost function
            :grad: Grad of cost function
            :w: Initial weights (Default: None, which implies random)
            :max_iter: default set in MAX_ITER

        Returns:
            :returns: w
        """
        assert callable(cost)
        assert callable(grad)
        assert X.shape[0] == y.shape[0]
        assert max_iter > 3
        _, n = X.shape
        if w is None:
            w = np.random.randn(n)

        step = 1
        start = time.time()
        for k in range(max_iter):
            direction = -grad(X, y, w)
            # line search
            backtracking = True
            while backtracking:
                left = cost(X, y, w + step * direction)
                right = cost(X, y, w) + self.alpha * step \
                    * np.dot(grad(X, y, w), direction)
                if left > right:
                    step *= self.beta
                else:
                    backtracking = False
            w = w + step * direction
            self.history['obj_val'].append(cost(X, y, w))
            norm_grad = np.linalg.norm(grad(X, y, w))
            if VERBOSE:
                print('Iter {} - obj: {:.4f} norm_grad: {:.12f}'.format(k, self.history['obj_val'][-1], norm_grad))
            if norm_grad < TOLERANCE:
                if VERBOSE:
                    print('BREAK')
                break
        elapsed = time.time() - start
        return w, elapsed


def proximal_group(w, ind, thres):
    """
    Args:
        w (np.array): parameter vector.
        ind (np.array): group indication with weights.
        thres (float): parameter of the projection.

    Returns:
        w
    """
    ngroups = ind.shape[1]  # number of groups
    w_new = w.copy()
    # Group Lasso penalization
    for i in range(0, ngroups):
        temp = ind[:, i]
        ids_group = np.arange(temp[0], temp[1], dtype=np.int)
        twoNorm = np.sqrt(np.dot(w_new[ids_group], w_new[ids_group]))
        if twoNorm > thres * temp[2]:
            fac = (twoNorm - (thres * temp[2])) / float(twoNorm)
            w_new[ids_group] = w_new[ids_group] * fac
        else:
            w_new[ids_group] = 0.
    return w_new


def proximal_constrained(w):
    """
    Projection on positive cone, for the restricted Lasso optimization.

    Args:
        :param w: 
    Returns
        w
    """
    return np.maximum(w, 0.)


def shrinkage(a, kappa):
    """
    Shrinkage

    Args:
        :param a: 
        :param kappa: 

    Returns
        res
    """
    #res = np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)
    res = np.multiply(np.sign(a), np.maximum(np.abs(a) - kappa, 0.))
    """
    a = a.flatten()
    print('kappa: {:.4f}'.format(kappa))
    res = np.zeros((a.shape[0], 1))
    for ind, val in enumerate(a):
        if val > kappa:
            res[ind] = val - kappa
        elif val < -kappa:
            res[ind] = val + kappa
        else:
            res[ind] = 0
    #print('kappa: {:.4f}, l1: {:.4f}'.format(kappa, np.sum(abs(res))))
    print(np.linalg.norm(res - res2))
    input()
    """
    return res


def compute_largest_group_norm(v, ind, dim, ntasks):
    """
    Largest group norm

    :param v: 
    :param ind: 
    :param dim: 
    :param ntasks: 
    :returns: 
    :rtype: 
    """
    lambda2_max = 0
    ngroups = ind.shape[1]  # number of groups
    w2D = np.reshape(v, (dim, ntasks), order='F')

    for t in range(ntasks):
        for i in range(ngroups):
            temp = ind[:, i]
            ids = np.arange(temp[0], temp[1], dtype=np.int)
            twoNorm = np.linalg.norm(w2D[ids, t]) / float(temp[2])
            if twoNorm > lambda2_max:
                lambda2_max = twoNorm
    return lambda2_max


def proximal_composition(v, ind, dim, ntasks):
    '''
    Args:
        v: vector weights
        ind: index of the features' groups
        dim: problem dimension
        ntasks: number of tasks

    Results:
        w_new: update vector weights
    '''

    w_new = np.zeros((dim, ntasks))
    ngroups = ind.shape[1] - 1  # number of groups

    # convert vector weights to a 2D representation (column format - Fortran)
    w2D = np.reshape(v, (dim, ntasks), order='F')

    # L21 + LG
    if ind[0, 0] == -1:
        for i in range(dim):
            w_new[i, :] = epp2(w2D[i, :], ntasks, ind[2, 0])

    for t in range(ntasks):
        for i in range(ngroups):
            temp = ind[:, i + 1]  # .astype(int)
            ids_group = np.arange(temp[0], temp[1], dtype=np.int)
            twoNorm = np.sqrt(np.dot(w_new[ids_group, t], w_new[ids_group, t]))

            if twoNorm > temp[2]:
                w_new[ids_group, t] = w_new[ids_group, t] * (twoNorm - temp[2]) / float(twoNorm)
            else:
                w_new[ids_group, t] = 0

    # reshape it back to a vector representation and return it
    return np.reshape(w_new, (dim * ntasks,), order='F')


def proximal_average(w, ind, dim, ntasks):
    """
    Proximal average

    :param w: 
    :param ind: 
    :param dim: 
    :param ntasks: 
    :returns: 
    :rtype: 

    """

    ngroups = ind.shape[1] - 1  # number of groups
    w2D = np.reshape(w, (dim, ntasks), order='F')
    w1_new = np.zeros(w2D.shape)
    w2_new = w2D.copy()

    # L2,1-norm penalization
    if ind[0, 0] == -1:
        for i in range(dim):  # applies l2,1-norm on matrix W
            w1_new[i, :] = epp2(w2D[i, :], ntasks, ind[2, 0])

    # Group Lasso penalization
    for t in range(ntasks):  # applies group lasso for each tas independently
        for i in range(ngroups):
            temp = ind[:, i + 1]  # +1 because there was a -1 column added as the first column
            ids_group = np.arange(temp[0], temp[1], dtype=np.int)
            twoNorm = np.sqrt(np.dot(w2D[ids_group, t], w2D[ids_group, t]))

            # print np.linalg.norm(w2D[ids_group,t])
            if twoNorm > temp[2]:
                w2_new[ids_group, t] = w2_new[ids_group, t] * (twoNorm - temp[2]) / float(twoNorm)
            else:
                w2_new[ids_group, t] = 0

    # average of L2,1 and Group lasso norms
    w_new = (w1_new + w2_new) / 2.0

    return np.reshape(w_new, (dim * ntasks,), order='F')


def epp2(v, n, rho):
    """
    epp2

    :param v: 
    :param n: 
    :param rho: 
    :returns: 
    :rtype: 

    """
    v2 = np.sqrt(np.dot(v, v))
    if rho >= v2:
        xk = np.zeros(n)
    else:
        ratio = (v2 - rho) / float(v2)
        xk = v * ratio
    return xk
