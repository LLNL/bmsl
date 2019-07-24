#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:24:54 2018

@author: goncalves1
"""
import numpy as np
import scipy.stats as spst
import pickle
import os
from ..base import BaseMTLEstimator
import math

np.seterr(all='raise')

from scipy import stats

_MINIMUM_BETA_P_ = 1e-2
_MIN_EXP = -100  # minimum value for exp


class BJFSRegressor(BaseMTLEstimator):
    """ Implement the Bayesian Joint Feature Selection regressor. """

    def __init__(self, tau_a=1, tau_b=1, sig_inv_a=1, sig_inv_b=1, theta_a=1, theta_b=1,
                 gibbs_iters=10000, gibbs_burnin=5000,
                 debug_mode=False, fit_intercept=True, normalize=False, name='BJFS'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            tau_a (float): shape parameter for gamma prior
            tau_b (float): scale parameter for gamma prior
            phi_eta (int): degrees of freedom for Wishart prior
            theta_a (float): shape-alpha parameter for beta prior
            theta_b (float): shape-beta parameter for beta prior
            sigma_step_method (str): related to W's prior (BGL or Wishart)
            gibbs_iters (int): number of steps for the Gibbs sampler
            gibbs_burnin (int): number of burn-in steps for the Gibbs sampler
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.hyper_params = {'tau_a': tau_a,
                             'tau_b': tau_b,
                             'sig_inv_a': sig_inv_a,
                             'sig_inv_b': sig_inv_b,
                             'theta_a': theta_a,
                             'theta_b': theta_b}
        self.gibbs_iters = gibbs_iters
        self.gibbs_burnin = gibbs_burnin
        self.prediction_method = 'bayesian-prediction'  # 'point-prediction'

        self.debug_mode = debug_mode
        self.posterior_samples = list()

    def _fit(self, x, y, **kwargs):

        self.posterior_samples = self._inference(x, y)
        fname = os.path.join(self.output_directory,
                             '{}.posterior'.format(self.__str__()))
        with open(fname, 'wb') as fh:
            pickle.dump(self.posterior_samples, fh)

        if self.debug_mode:
            fname = os.path.join(self.output_directory,
                                 '{}.norm'.format(self.__str__()))
            with open(fname, 'wb') as fh:
                pickle.dump(self.norm_params, fh)

            fname = os.path.join(self.output_directory,
                                 '{}.loglik'.format(self.__str__()))
            with open(fname, 'wb') as fh:
                pickle.dump(self.loglik, fh)

    def _predict(self, x):
        """ make prediction for a new input sample """
        yhat = [None] * self.nb_tasks
        for task in range(self.nb_tasks):
            yhat[task] = np.zeros((x[task].shape[0], ))
            if self.prediction_method == 'point-prediction':
                yhat[task] = np.dot(x[task], self.W[:, task])
            elif self.prediction_method == 'bayesian-prediction':
                yhat_post = np.zeros((x[task].shape[0], len(self.posterior_samples)))
                for k, sample in enumerate(self.posterior_samples):
                    wb = np.multiply(sample['w'], sample['beta'][:, np.newaxis])
                    yhat_post[:, k] = np.dot(x[task], wb[:, task])
                # for i in range(x[task].shape[0]):
                    # yhat[task][i] = np.mean(remove_outliers(yhat_post[i, :]))

                # yhat[task] = np.median(yhat_post, axis=1)
                yhat[task] = np.mean(yhat_post, axis=1)
        return yhat

    def _inference(self, x, y):
        """ Gibbs sampler """
        dim = self.nb_dims
        T = self.nb_tasks

        # for t in range(T):
        # initial_w[:, t] = np.dot(np.linalg.pinv(x[t]), y[t])

        # priors = {'w': initial_w,
        #           'tau': {  'a': self.hyper_params['tau_a'],   'b': self.hyper_params['tau_b']},
        #           'sig_inv': { 'a': self.hyper_params['phi_a'],   'b': self.hyper_params['phi_b']},
        #           'theta': {'c': self.hyper_params['theta_a'], 'd': self.hyper_params['theta_b']}}

        # initial values
        w_i = np.random.randn(dim, T)  # random initial value for w
        beta_i = np.ones((dim, ))  # initially all variables are important
        tau_i = np.ones(T)  # noise prior equals to 1

        posterior_samples = list()

        if self.debug_mode:
            self.norm_params = {'theta': list(),
                                'beta': list(),
                                'tau': list(),
                                'phi': list(),
                                'w': list(),
                                'beta_a': list(),
                                'beta_b': list(),
                                'beta_p': list()}
            self.loglik = {'w': list()}

        for i in range(self.gibbs_iters):
            if i % 1000 == 0:
                self.logger.info('Gibbs iteration {}'.format(i))

            theta_i = self._sample_posterior_theta(beta_i)

            beta_i, b_a, b_b, b_p = self._sample_posterior_beta(beta_i, w_i, tau_i, theta_i, x, y)

            tau_i = self._sample_posterior_tau(w_i, beta_i, x, y)

            sig_inv_i = self._sample_posterior_sigma_inverse(w_i, beta_i)

            w_i, llk = self._sample_posterior_w(w_i, x, y, tau_i, sig_inv_i, beta_i)

            if self.debug_mode:

                self.norm_params['theta'].append(np.mean(theta_i))
                self.norm_params['beta'].append(np.mean(beta_i))
                self.norm_params['tau'].append(1 / tau_i[0])
                self.norm_params['phi'].append(np.mean(sig_inv_i))
                self.norm_params['w'].append(np.mean(w_i))
                self.loglik['w'].append(llk)

            if i > self.gibbs_burnin:
                # start collecting samples
                posterior_samples.append({'w': w_i, 'tau': tau_i,
                                          'sig_inv_i': sig_inv_i, 'beta': beta_i,
                                          'theta': theta_i})

        return posterior_samples

    def _sample_posterior_w(self, w_p, x, y, tau, phi, beta):
        """Draw from p(w|X,Y,\Tau,\beta)"""
        dim, T = w_p.shape
        w = w_p.copy()

        """ phi is a vector of precision scalars. not a precision matrix. each element of phi is associated with a task. """
        # cov = np.linalg.inv(phi+1e-8*np.eye(T))  # covariance matrix

        # pre-compute some values to save time
        sig2 = [None] * T
        mu1 = [None] * T
        #cov = [None]*T

        for k in range(T):

            _k = np.ones((T, ), dtype=bool)
            _k[k] = False  # exclude k-th element
            #cov[k] = (1.0/phi[k])*np.eye(dim)
            mu1[k] = 0.  # np.dot(cov[k, _k], np.linalg.inv(cov[_k, :][:, _k] + 1e-8*np.eye(T-1)))
            sig2[k] = 1.0 / phi[k]  # - np.dot(mu1[k], cov[_k, k])

        for j in range(w.shape[0]):  # for each row (dimension)
            for k in range(w.shape[1]):  # for each column (task)
                _k = np.ones((T,), dtype=bool)
                _k[k] = False  # exclude k-th element
                if beta[j] == 0:
                    a_jk = 0.  # np.dot(mu1[k], w[j, _k])
                    b_jk = 1.0 / sig2[k]
                else:
                    # mu_jk = 0. #np.dot(mu1[k], w[j, _k])
                    _j = np.ones((w.shape[0],), dtype=bool)
                    _j[j] = False  # exclude k-th element

                    wb_jk = w[_j, k] * beta[_j]  # element-wise multiplication
                    alpha_jk = np.dot(x[k][:, _j], wb_jk)

                    # compute once and save it to avoid computing multiple times
                    term0 = tau[k] * np.dot(x[k][:, j], x[k][:, j])

                    # mean of the posterior distribution
                    ajk_num = tau[k] * np.dot(y[k] - alpha_jk, x[k][:, j])  # (mu_jk/sig2[k]) + beta[j]*tau[k]*np.dot(y[k]-alpha_jk, x[k][:, j])
                    ajk_den = 1.0 / sig2[k] + term0  # 1.0/sig2[k] + term0
                    a_jk = ajk_num / ajk_den
                    # If the above ratio is numerically unstable use this next line.
                    # a_jk = np.exp( np.log(ajk_num) - np.log(ajk_den) )

                    # precision of the posterior distribution
                    b_jk = (1.0 / sig2[k]) + term0

                w[j, k] = np.random.normal(a_jk, np.sqrt(1.0 / b_jk), size=1)[0]

        # print('Sampling w took {} secs'.format(time.time()-t0))
        if self.debug_mode:
            # compute loglik of the just sampled w
            # wb = np.multiply(w, beta[:, np.newaxis])
            loglik = 0
            # for t in range(T):
            #    loglik += -0.5*tau[t]*np.sum((y[t]-np.dot(x[t], wb[:, t]))**2)
            # loglik += np.std(y[t]-np.dot(x[t], wb[:, t]))
            return w, loglik
        else:
            return w, loglik

    def _sample_posterior_tau(self, w, beta, x, y):
        """Draw from p(tau|y,x,w,beta) """
        T = len(x)  # number of task
        tau = list()
        for t in range(T):
            u = np.dot(x[t], np.multiply(w[:, t], beta))
            shape = self.hyper_params['tau_a'] + 0.5 * x[t].shape[0]
            rate = self.hyper_params['tau_b'] + 0.5 * np.sum((y[t] - u)**2)
            scale = 1 / rate
            tau.append(np.random.gamma(shape, scale, size=1)[0])
        return tau

    def _sample_posterior_sigma_inverse(self, w, beta):
        """Draw from p(sig_inv|w) """

        dim, T = w.shape
        sig_inv = [None] * T  # np.zeros(dim)
        for k in range(T):
            shape = self.hyper_params['sig_inv_a'] + 0.5 * dim
            rate = self.hyper_params['sig_inv_b'] + 0.5 * np.dot(w[:, k], w[:, k])
            sig_inv[k] = np.random.gamma(shape=shape, scale=1.0 / rate)
        return sig_inv

    def _sample_posterior_beta(self, p_beta, w, tau, theta, x, y):
        """Sampling from p(beta|y,x,w,tau,theta)"""

        def rexp(a):
            """ Robust implementation of exponential function """
            return np.exp(np.maximum(a, _MIN_EXP)).astype(np.float64)

        def compute_loglikelihood(beta, j, x, y, w, theta, tau, T):
            wb = w * beta  # it's faster than np.multply
            loglik = 0
            for t in range(T):
                res = y[t] - np.dot(x[t], wb[:, t])
                loglik += tau[t] * np.dot(res.T, res)
            return -0.5 * loglik + (math.log(theta) if beta[j] == 1 else math.log(1 - theta))

        dim, T = w.shape
        beta = p_beta.copy()[:, np.newaxis]

        a_vec = None  # np.zeros((dim, 1))
        b_vec = None  # np.zeros((dim, 1))
        p_vec = np.zeros((dim, 1))
        for j in range(dim):
            # for beta_j=0
            beta[j] = 0
            beta_0 = compute_loglikelihood(beta, j, x, y, w, theta, tau, T)

            # for beta_j=1
            beta[j] = 1
            beta_1 = compute_loglikelihood(beta, j, x, y, w, theta, tau, T)

            # exp-normalize trick
            # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            b = np.maximum(beta_0, beta_1)
            p = rexp(beta_1 - b) / (rexp(beta_0 - b) + rexp(beta_1 - b))

            p = np.maximum(_MINIMUM_BETA_P_, p)
            p_vec[j] = p

            beta[j] = np.random.binomial(n=1, p=p, size=1)

        return beta[:, 0], a_vec, b_vec, p_vec

    def _sample_posterior_theta(self, beta):
        """Sampling from p(theta|beta,c,d) """
        theta = np.random.beta(self.hyper_params['theta_a'] + beta.sum(),
                               self.hyper_params['theta_b'] + (1 - beta).sum(), size=1)
        return theta
