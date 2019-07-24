#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:37:18 2018

L21 Joint Feature Learning with Least Squares Loss.

OBJECTIVE:
argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
            + opts.rho_L2 * \|W\|_2^2 + rho1 * \|W\|_{2,1} }

INPUT:
 X: {n * d} * t - input matrix
 Y: {n * 1} * t - output matrix
 rho1: L2,1-norm group Lasso parameter.
 optional:
   opts.rho_L2: L2-norm parameter (default = 0).

OUTPUT:
 W: model: d * t
 funcVal: function value vector.

RELATED PAPERS:
   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
       Report, 2010.

@author: goncalves1
"""
#import sys
import os
import pickle
import numpy as np
#import matplotlib.pyplot as plt
from ..base import BaseMTLEstimator


class JFSMTLRegressor(BaseMTLEstimator):
    """
    Implement the L21 Joint Feature Learning regressor.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """

    def __init__(self, rho_L21=0.1, rho_L2=0, name='JFSMTL-Reg',
                 fit_intercept=True, normalize=False):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.rho_L21 = rho_L21
        self.rho_L2 = rho_L2
        self.W = None

        self.max_iters = 3000
        self.tol = 1e-5  # minimum tolerance: eps * 100
        self.tFlag = 3

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (list): list of pandas.DataFrame input data matrix (covariates).
            y (list): list of pandas.DataFrame label vector (outcome).
        Returns:
            None.
        """
        funcVal = list()

        # x has to be features x samples
        x = [x[t].T for t in range(self.nb_tasks)]

        # initialize a starting point
        W0 = np.ones((self.nb_dims, self.nb_tasks))

        # this flag tests whether the gradient step only changes a little
        bFlag = False

        Wz = W0
        Wz_old = W0

        t = 1.0
        t_old = 0

        itrn = 0
        gamma = 1.0
        gamma_inc = 2.0
        # self.logger.info('{:5} | {:13} | {:13}'.format('Iter.',
        #                                                'FuncVal',
        #                                                'Delta-FuncVal'))

        while itrn < self.max_iters:
            alpha = (t_old - 1) / float(t)
            Ws = (1 + alpha) * Wz - alpha * Wz_old

            # compute function value and gradients of the search point
            gWs = self.__gradVal_eval(Ws, x, y)
            Fs = self.__funVal_eval(Ws, x, y)
            while True:
                Wzp = self.__FGLasso_projection(Ws - gWs / gamma,
                                                self.rho_L21 / gamma)
                Fzp = self.__funVal_eval(Wzp, x, y)

                delta_Wzp = Wzp - Ws
                r_sum = np.linalg.norm(delta_Wzp, 'fro')**2
                try:
                    zzz = np.multiply(delta_Wzp, gWs)
                except FloatingPointError:
                    print('delta_Wzp: '.format(delta_Wzp))
                    print('gWs: '.format(gWs))
                    print(e)
                Fzp_gamma = (Fs + np.multiply(delta_Wzp, gWs).sum() +
                             gamma / 2.0 * np.linalg.norm(delta_Wzp, 'fro')**2)

                if r_sum <= 1e-20:
                    # gradient step makes little improvement
                    bFlag = True
                    break
                if Fzp <= Fzp_gamma:
                    break
                else:
                    gamma *= gamma_inc
            Wz_old = Wz
            Wz = Wzp

            funcVal.append(Fzp + self.__nonsmooth_eval(Wz, self.rho_L21))
            # if itrn > 1:
            #     self.logger.info('{:^5} | {} | {}'.format(itrn,
            #                                               funcVal[-1],
            #                                               abs(funcVal[-1] -
            #                                                   funcVal[-2])))
            if bFlag:
                # The program terminates as the gradient step
                # changes the solution very small
                # self.logger.info(('The program terminates as the gradient step'
                #                   'changes the solution very small'))
                break
            # test stop condition.
            if self.tFlag == 0:
                if itrn >= 2:
                    if abs(funcVal[-1] - funcVal[-2]) <= self.tol:
                        break
            elif self.tFlag == 1:
                if itrn >= 2:
                    if (abs(funcVal[-1] - funcVal[-2]) <
                            (self.tol * funcVal[-2])):
                        break
            elif self.tFlag == 2:
                if funcVal[-1] <= self.tol:
                    break
            elif self.tFlag == 3:
                if itrn >= self.max_iters:
                    break
            else:
                raise ValueError('Unknown termination flag')

            itrn = itrn + 1
            t_old = t
            t = 0.5 * (1 + (1 + 4 * t**2)**0.5)

        self.W = Wzp.copy()
        # save model into pickle file
        filename = os.path.join(self.output_directory,
                                '{}.model'.format(self.__str__()))
        with open(filename, "wb") as fh:
            pickle.dump(self.W, fh)

    def _predict(self, x):
        """ Predict regression value for the input x.
        Args:
            x (pandas.DataFrame): list of pandas.DataFrame input data matrix.
        Returns:
            list of numpy.array with the predicted values.
        """
        y_hats = list()
        for t in range(self.nb_tasks):
            y_hats.append(np.dot(x[t], self.W[:, t]))
        return y_hats

    def __FGLasso_projection(self, W, rho):
        """ Solve it in row wise (L_{2,1} is row coupled).
         for each row we need to solve the proximal operator
         argmin_w { 0.5 \|w - v\|_2^2 + lambda_3 * \|w\|_2 }
        """
        Wp = np.zeros(W.shape)
        for i in range(W.shape[0]):
            nm = np.linalg.norm(W[i, :], 2)
            if nm == 0:
                w = np.zeros((W.shape[1], 1))
            else:
                w = max(nm - rho, 0) / nm * W[i, :]
            Wp[i, :] = w.T
        return Wp

    def __gradVal_eval(self, W, X, Y):
        """smooth part gradient
        """
        grad_W = list()
        for i in range(self.nb_tasks):
            grad_W.append(np.dot(X[i], np.dot(X[i].T, W[:, i]) - Y[i]))
        grad_W = np.vstack(grad_W).T
        grad_W += self.rho_L2 * 2 * W
        return grad_W

    def __funVal_eval(self, W, X, Y):
        """ smooth part function value."""
        funcVal = 0
        for i in range(self.nb_tasks):
            funcVal += 0.5 * np.linalg.norm(Y[i] - np.dot(X[i].T, W[:, i]))**2
        funcVal += self.rho_L2 * np.linalg.norm(W, 'fro')**2
        return funcVal

    def __nonsmooth_eval(self, W, rho_1):
        """ non-smooth part function value. """
        non_smooth_value = 0
        for i in range(self.nb_dims):
            non_smooth_value += rho_1 * np.linalg.norm(W[i, :], 2)
        return non_smooth_value

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.rho_L21 = params['rho_L21']
        self.rho_L2 = params['rho_L2']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'rho_L21': self.rho_L21,
               'rho_L2': self.rho_L2}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        rho_L21 = np.logspace(-3, 5, 10)
        rho_L2 = np.logspace(-3, 5, 10)
        for r0 in rho_L21:
            for r1 in rho_L2:
                yield {'rho_L21': r0,
                       'rho_L2': r1}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
