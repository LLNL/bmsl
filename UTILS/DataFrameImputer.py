#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:21:20 2018

@author: goncalves1
"""

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    # I really don't want to check 
    # X[c].dtype != 'float64'. I would prefer to check
    # X[c].dtype == 'category' but this isn't working.
#    def fit(self, X, y=None):
#        self.fill = pd.Series([X[c].value_counts().index[0]
#            if (X[c].dtype == np.dtype('O') \
#            or X[c].dtype != 'float64') else X[c].mean() for c in X],
#            index=X.columns)
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if (X[c].dtype == np.dtype('O') or X[c].dtype != 'float64')
            else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)
