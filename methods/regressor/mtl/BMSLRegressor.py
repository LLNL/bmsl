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


_MINIMUM_BETA_P_ = 1e-2
_MIN_EXP = -100  # minimum value for exp


class BMSLRegressor(BaseMTLEstimator):
    """ Implement the Bayesian Multitask Learning regressor. """

    def __init__(self, tau_a=1, tau_b=1, phi_eta=10, theta_a=1, theta_b=1,
                 gibbs_iters=10000, gibbs_burnin=5000, sigma_step_method='wishart',
                 debug_mode=False, fit_intercept=True, normalize=False, name='BMTL'):
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
        # make sure informed sigma_step_method is correct
        assert sigma_step_method in ['wishart', 'bgl']

        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.hyper = {'tau_a': tau_a, 'tau_b': tau_b,
                      'phi_eta': phi_eta, 'theta_a': theta_a,
                      'theta_b': theta_b}
        self.gibbs_iters = gibbs_iters
        self.gibbs_burnin = gibbs_burnin
        self.prediction_method = 'bayesian-prediction'  # 'point-prediction'
        self.sigma_step_method = sigma_step_method  # 'wishart'  or 'bgl' (bayesian graphical lasso)

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

        # def remove_outliers(x):
        #     perc_5 = np.percentile(x, 5)
        #     perc_95 = np.percentile(x, 95)
        #     x = x[(x > perc_5) & (x < perc_95)]
        #     return x

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

        initial_w = np.random.randn(dim, T)
        # for t in range(T):
        # initial_w[:, t] = np.dot(np.linalg.pinv(x[t]), y[t])

        priors = {'w': initial_w,
                  'tau': {'a': self.hyper['tau_a'], 'b': self.hyper['tau_b']},
                  'phi': {'V': np.eye(T), 'eta': self.hyper['phi_eta']},
                  'theta': {'c': self.hyper['theta_a'],
                            'd': self.hyper['theta_b']}}

        phi_params = {'p_phi': priors['phi'],
                      'C': np.eye(T),
                      'Sig': np.eye(T),
                      'a_lambda': 1,
                      'b_lambda': 0.1}

        # initial values
        w_i = priors['w'].copy()
        beta_i = np.ones((dim, ))  # np.random.binomial(n=1, p=0.5, size=dim)
        tau_i = np.ones(T)

        # post_samp_size = (self.gibbs_iters-self.gibbs_burnin)
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

    #        theta_i = data['theta']  # true value
            theta_i = self._sample_posterior_theta(priors['theta'], beta_i)

    #        beta_i = data['beta']  # true value
            beta_i, b_a, b_b, b_p = self._sample_posterior_beta(beta_i, w_i, tau_i, theta_i, x, y)

    #        tau_i = data['tau']  # true value
            tau_i = self._sample_posterior_tau(priors['tau'], w_i, beta_i, x, y)

    #        phi_i = data['phi']  # true value
            phi_i, Sig = self._sample_posterior_sigma_inverse(w_i, beta_i, **phi_params)
            phi_params['C'] = phi_i.copy()
            phi_params['Sig'] = Sig.copy() if Sig is not None else None

            # phi_i = np.eye(T) #/dim
            # w_i = data['w']  # true value
            w_i, llk = self._sample_posterior_w(w_i, x, y, tau_i, phi_i, beta_i)

            if self.debug_mode:
                self.norm_params['theta'].append(np.mean(theta_i))
                self.norm_params['beta'].append(np.mean(beta_i))
                self.norm_params['tau'].append(1 / tau_i[0])
                self.norm_params['phi'].append(np.mean(phi_i))
                self.norm_params['w'].append(np.mean(w_i))
                self.loglik['w'].append(llk)

            if i > self.gibbs_burnin:
                # start collecting samples
                posterior_samples.append({'w': w_i.copy(), 'tau': tau_i.copy(),
                                          'phi': phi_i.copy(), 'beta': beta_i.copy(),
                                          'theta': theta_i.copy()})

            # p = 0 if p >= post_samp_size-1 else p+1
        return posterior_samples

    def _sample_posterior_w(self, w_p, x, y, tau, phi, beta):
        """Draw from p(w|X,Y,\Tau,\beta)"""
        dim, T = w_p.shape
        w = w_p.copy()

        cov = np.linalg.inv(phi + 1e-10 * np.eye(T))  # covariance matrix
        # print(cov)

        # pre-compute some values to save time
        sig2 = [None] * T
        mu1 = [None] * T
        for k in range(T):
            _k = np.ones((T, ), dtype=bool)
            _k[k] = False  # exclude k-th element
            mu1[k] = np.dot(cov[k, _k], np.linalg.inv(cov[_k, :][:, _k] + 1e-10 * np.eye(T - 1)))
            sig2[k] = cov[k, k] - np.dot(mu1[k], cov[_k, k])

        for j in range(w.shape[0]):  # for each row (dimension)
            if beta[j] == 0:
                w[j, :] = np.random.multivariate_normal(np.zeros((T, )), cov, size=1)
                # print(w[j, :])
            else:
                for k in range(w.shape[1]):  # for each column (task)
                    _k = np.ones((T,), dtype=bool)
                    _k[k] = False  # exclude k-th element
                    # if beta[j] == 0:
                    #     a_jk = np.dot(mu1[k], w[j, _k])
                    #     b_jk = 1.0/sig2[k]
                    # else:
                    mu_jk = np.dot(mu1[k], w[j, _k])
                    _j = np.ones((w.shape[0],), dtype=bool)
                    _j[j] = False  # exclude k-th element

                    wb_jk = w[_j, k] * beta[_j]  # element-wise multiplication
                    alpha_jk = np.dot(x[k][:, _j], wb_jk)

                    # compute once and save it to avoid computing multiple times
                    term0 = beta[j] * tau[k] * np.dot(x[k][:, j], x[k][:, j])  # **2).sum()

                    # mean of the posterior distribution
                    ajk_num = (mu_jk / sig2[k]) + beta[j] * tau[k] * np.dot(y[k] - alpha_jk, x[k][:, j])
                    ajk_den = 1.0 / sig2[k] + term0
                    a_jk = ajk_num / ajk_den

                    # precision of the posterior distribution
                    b_jk = (1.0 / sig2[k]) + term0

                    w[j, k] = np.random.normal(a_jk, np.sqrt(1.0 / b_jk), size=1)[0]
                # print(w[j, :])

        # print('Sampling w took {} secs'.format(time.time()-t0))
        if self.debug_mode:
            # compute loglik of the just sampled w
            wb = np.multiply(w, beta[:, np.newaxis])
            loglik = 0
            for t in range(T):
                loglik += -0.5 * tau[t] * np.sum((y[t] - np.dot(x[t], wb[:, t]))**2)
                # loglik += np.std(y[t]-np.dot(x[t], wb[:, t]))
            return w, loglik
        else:
            return w, loglik

    def _sample_posterior_tau(self, p_tau, w, beta, x, y):
        """Draw from p(tau|y,x,w,beta) """
        T = len(x)  # number of task
        tau = list()
        for t in range(T):
            u = np.dot(x[t], np.multiply(w[:, t], beta))
            shape = p_tau['a'] + 0.5 * x[t].shape[0]
            rate = 1 / (p_tau['b'] + 0.5 * np.sum((y[t] - u)**2))  # rate = 1/scale
            tau.append(np.random.gamma(shape, rate, size=1)[0])

        return tau

    def _bayesian_graphical_lasso(self, w_i, beta, C, Sig, a_lambda, b_lambda):
        """ Bayesian Graphical Lasso """
        def get_dual_index(x, idx):
            p = np.max(idx.shape)
            new_x = np.zeros((p, p))
            for i, j in enumerate(ind_noi):
                new_x[:, i] = x[ind_noi, j]
            return new_x

        w = w_i.copy()
        n, T = w.shape

        if n == 0:  # sample from the prior: no data to update
            return C, Sig
        S = np.dot(w.T, w)

        indmx = np.reshape(np.arange(T**2), (T, T), order='F')

        u = np.triu(indmx, 1).flatten('F')
        upperind = u[u > 0]

        l = np.triu(indmx.T, 1).flatten('F')
        lowerind = l[l > 0]

        tau = np.zeros((T, T))

        ind_noi_all = np.zeros((T - 1, T), dtype=np.int)
        for i in range(T):
            if i == 0:
                ind_noi = np.arange(1, T).T
            elif i == T:
                ind_noi = np.arange(0, T - 1).T
            else:
                ind_noi = np.concatenate((np.arange(0, i),
                                          np.arange(i + 1, T))).T
            ind_noi_all[:, i] = ind_noi

        tau = np.zeros((T, T))

        # Sample lambda
        apost = a_lambda + T * (T + 1) / 2
        bpost = b_lambda + np.abs(C).sum() / 2
        lmbda = np.random.gamma(apost, 1.0 / bpost, size=1)

        # sample tau off-diagonal
        Cadjust = np.maximum(np.abs(C.flat[upperind]), 10e-6)
        lmbda_prime = lmbda**2
        mu_prime = np.minimum(lmbda / Cadjust, 10e12)
        tau_temp = 1. / np.random.wald(mu_prime, lmbda_prime)
        tau.flat[upperind] = tau_temp
        tau.flat[lowerind] = tau_temp

        for i in range(T):
            ind_noi = ind_noi_all[:, i]
            tau_temp = tau[ind_noi, i]

            Sig11 = get_dual_index(Sig, ind_noi)
            Sig12 = Sig[ind_noi, i]
            invC11 = Sig11 - np.dot(Sig12[:, np.newaxis],
                                    Sig12[:, np.newaxis].T) / Sig[i, i]
            Ci = (S[i, i] + lmbda) * invC11 + np.diag(1. / tau_temp)
            Ci_chol = np.linalg.cholesky(Ci).T
            mu_i = np.linalg.solve(-Ci, S[ind_noi, i])

            beta_i = mu_i + np.linalg.solve(Ci_chol, np.random.randn(T - 1))

            C[ind_noi, i] = beta_i
            C[i, ind_noi] = beta_i

            gam = np.random.gamma(n / 2 + 1, 2 / (S[i, i] + lmbda))[0]

            beta_i = beta_i[:, np.newaxis]
            C[i, i] = gam + np.dot(np.dot(beta_i.T, invC11), beta_i)

            invC11beta = np.dot(invC11, beta_i)
            new_sig = invC11 + np.dot(invC11beta, invC11beta.T) / gam
            for ik, jk in enumerate(ind_noi):
                Sig[ind_noi, jk] = new_sig[:, ik]
            Sig12 = -invC11beta / gam
            Sig12 = Sig12[:, 0]
            Sig[ind_noi, i] = Sig12
            Sig[i, ind_noi] = Sig12.T
            Sig[i, i] = 1. / gam

        return C, Sig

    def _sample_posterior_wishart(self, p_phi, w_i, beta):
        w = w_i.copy()
        dim, _ = w.shape

        T = w.shape[1]  # number of tasks

        if dim == 0:
            # sample from the prior: no data to update
            phi = spst.wishart.rvs(T + p_phi['eta'], p_phi['V'], size=1)
        else:
            # there is data to update
            if dim > 1:
                w = w - w.mean(axis=0)
            # assuming prior is diagonal, meaning that tasks are independent
            V0_inv = np.diag(1.0 / np.diag(p_phi['V']))
            newV = np.linalg.inv(V0_inv + np.dot(w.T, w) + 1e-8 * np.eye(T))

            # for numerical stability
            # u, s, v = np.linalg.svd(newV)
            # s_upd = np.maximum(s, 10e-5)
            # newV = np.dot(np.dot(u, np.diag(s_upd)), v)
            phi = spst.wishart.rvs(p_phi['eta'] + T + dim, newV, size=1)

        return phi

    def _sample_posterior_sigma_inverse(self, w, beta, **kwargs):
        """Draw from p(phi|w) """

        if self.sigma_step_method == 'wishart':
            phi = self._sample_posterior_wishart(kwargs['p_phi'], w, beta)
            Sig = None
        elif self.sigma_step_method == 'bgl':  # Bayesian Graphical Lasso
            phi, Sig = self._bayesian_graphical_lasso(w, beta,
                                                      kwargs['C'],
                                                      kwargs['Sig'],
                                                      kwargs['a_lambda'],
                                                      kwargs['b_lambda'])
        else:
            raise NotImplementedError('Unknown inv sigma method {}'.format(method))
        return phi, Sig

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

    def _sample_posterior_theta(self, p_theta, beta):
        """Sampling from p(theta|beta,c,d) """
        theta = np.random.beta(p_theta['c'] + beta.sum(),
                               p_theta['d'] + (1 - beta).sum(), size=1)
        return theta
