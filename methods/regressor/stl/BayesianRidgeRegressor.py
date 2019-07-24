# -*- coding: utf-8 -*-
# @Author: Andre Goncalves
# @Date:   2018-11-28 11:48:54
# @Last Modified by:   Andre Goncalves
# @Last Modified time: 2018-11-28 11:58:06
import sys
import numpy as np
from sklearn import linear_model

sys.path.append('..')
from design import Method


class BayesianRidgeRegressor(Method):
    """
    Implement an Ordinary Linear Regression model (non-regularized) using
    scikit-learn.
    """
    def __init__(self, alpha_1=1e-6, alpha_2=1e-6,
    			 lambda_1=1e-6, lambda_2=1e-6, 
    			 name='BayesRidge'):

        # set method's name and paradigm
        super().__init__(name, 'BayesRidge')

        self.name = name
        self.alpha_1 = alpha_1  # shape parameter for the Gamma distribution prior over the alpha parameter
		self.alpha_2 = alpha_2  # inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter
		self.lambda_1 = lambda_1  # shape parameter for the Gamma distribution prior over the lambda parameter
		self.lambda_2 = lambda_2  # inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter

        self.model = linear_model.BayesianRidge(alpha_1=self.alpha_1,
         									    alpha_2=self.alpha_2,
         									    lambda_1=self.lambda_1,
         									    lambda_2=self.lambda_2)
        self.output_directory = ''

    def fit(self, x, y, **kwargs):
        """
        Train model on given data x and y.
        Args:
            x (pandas.DataFrame): input data matrix (covariates).
            y (pandas.DataFrame): label vector (outcome).
        Returns:
            None.
        """
        self.logger.info('Traning process is about to start.')

        # self.model.alpha = self.alpha_l1

        # Train the model using the training sets
        self.model.fit(x, y)

        self.logger.info('Training process finalized.')

    def predict(self, x, **kwargs):
        """ Predict regression value for the input x.
        Args:
            x (numpy.array): input data matrix.
        Returns:
            numpy.array with the predicted values.
        """
        # convert to numpy array
#        x = x.as_matrix().astype(np.float64)

        y_hat = self.model.predict(x)

        # make sure prediction are consist with problem specifications
        # there are no "negative survival time", for example; and
        # there is a cap limiting the maximum survival time
        return y_hat

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
		self.alpha_1 = params['alpha_1']
		self.alpha_2 = params['alpha_2']
		self.lambda_1 = params['lambda_1']
		self.lambda_2 = params['lambda_2']


    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'alpha_1': self.alpha_1,
		       'alpha_2': self.alpha_2,
			   'lambda_1': self.lambda_1,
			   'lambda_2': self.lambda_2}

        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        alphas_l1 = np.logspace(-4, 2, 30)
        # alphas_l1 = np.linspace(1e-4, 1e3, 20)
        for alpha_l1 in alphas_l1:
            yield {'alpha_l1': alpha_l1}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (string): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
