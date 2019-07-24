#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:58:03 2018

@author: goncalves1
"""
import numpy as np
from sklearn import linear_model
from ..base import BaseMTLEstimator


class LinearRegressorPooled(BaseMTLEstimator):
    """
    Implement an Ordinary Linear Regression model (no-regularized) using
    tensflow framework.

    Attributes:
        batch_size (int): size of the mini-batch.
        nb_epochs (int): number of epochs for training
    """

    def __init__(self, fit_intercept=True,
                 normalize=False,
                 name='Pooled-Lin-Reg'):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            batch_size (int): mini-batch size.
            nb_epochs (int): number of training epochs.
        """
        # set method's name and paradigm
        super().__init__(name, fit_intercept, normalize)

        self.name = name
        self.model = linear_model.LinearRegression()

    def _fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        # self.logger.info('Traning process is about to start.')

        # get number of tasks
        self.ntasks = len(x)
        self.dimension = x[0].shape[1]

        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.float64).ravel()

        xpooled = np.row_stack(x)
        ypooled = np.concatenate(y)

        # Train the model using the training sets
        self.model.fit(xpooled, ypooled)

        # self.logger.info('Training process finalized.')

    def _predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        yhat = [None] * len(x)
        for t in range(len(x)):
            x[t] = x[t].astype(np.float64)
            yhat[t] = self.model.predict(x[t])
        return yhat

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        pass

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        yield {}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
