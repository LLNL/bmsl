#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 12:40:23 2018

@author: goncalves1
"""
import numpy as np
import sys
sys.path.append('..')
from design import ModelTraining
import datasets.RegressionArtificialDatasetMTL as dba
from methods.regressor.stl.LASSORegressor import LASSORegressor
from methods.regressor.pooled.LinearRegressorPooled import LinearRegressorPooled
from methods.regressor.pooled.LASSORegressorPooled import LASSORegressorPooled
from methods.regressor.mtl.BMSLRegressor import BMSLRegressor
np.random.seed(1234)


if __name__ == '__main__':

    nb_samples = 200
    dimension = 25
    nb_nonrelevant_dims = 5
    nb_tasks = 5
    dataset = dba.RegressionArtificialDatasetMTL(nb_tasks,
                                                 nb_samples,
                                                 dimension,
                                                 nb_nonrelevant_dims)
    dataset.prepare_data()

    # list of methods to compare against
    # note that the same method can be called many times just using
    # different hyper-parameter values
    methods = [
        LASSORegressor(alpha=0.05,
                       normalize=True,
                       fit_intercept=True,
                       name='LASSO'),
        LinearRegressorPooled(normalize=True,
                              fit_intercept=True,
                              name='Pooled'),
        LASSORegressorPooled(alpha=0.05,
                             normalize=True,
                             fit_intercept=True,
                             name='Pooled-LASSO')
    ]
    # BMSLRegressor(tau_a=0.01, tau_b=100, phi_eta=10,
    #               theta_a=1, theta_b=1,
    #               gibbs_iters=3000, gibbs_burnin=1000,
    #               debug_mode=True, fit_intercept=False,
    #               normalize=True, name='BMSL-W'),
    # BMSLRegressor(tau_a=0.01, tau_b=100, phi_eta=10,
    #               theta_a=1, theta_b=1,
    #               gibbs_iters=3000, gibbs_burnin=1000,
    #               sigma_step_method='bgl',
    #               debug_mode=True, fit_intercept=False,
    #               normalize=True, name='BMSL-GL'), ]

    # list of metrics to measure method's performance
    # see list of available metrics in utils/performance_metrics.py
    metrics = ['rmse']

    exp_folder = __file__.strip('.py')
    exp = ModelTraining(exp_folder)
    exp.execute(dataset, methods, metrics, nb_runs=3)
    exp.generate_report()
