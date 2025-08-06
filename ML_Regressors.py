"""
Author: Bahram Lavi, João Roberto Bertini Junior
Description: This script is the implimentation of shallow machine learning models considered in the paper:
"Comparing Shallow and Deep Learning Regression Methods to Forecast Short-Term Oil, Water, and Gas Rates in a Pre-Salt Petroleum Field"
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor

def call_ML_regressors():
    return {
            'GTB': {
            'function': GradientBoostingRegressor,
            'params': {},
            'search_params': {
              'min_samples_split': [0.05, 0.1, 0.2],
              'n_estimators': [50, 100, 150],
              'learning_rate': [0.001, 0.01, 0.1],
              'loss': ['squared_error', 'lad']
            }},
            'KNN': {
            'function': KNeighborsRegressor,
            'params': {},
            'search_params': {
              'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              'weights': ('uniform', 'distance')
            }},
            'ENET': {
            'function': ElasticNet,
            'params': { 'max_iter': 100000 },
            'search_params': {
              'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
              'l1_ratio': [0, 0.25, 0.5, 0.75, 1.0]
            }},
            'KRR': {
             'function': KernelRidge,
             'params': {},
             'search_params': {
               'kernel': ['poly'], 'degree': [2,3], 'alpha': [0.001],
               'kernel': ['rbf'], 'gamma':  np.logspace(-3, 3, 7), 'alpha': [1e0, 0.1, 1e-2, 1e-3]  #[1e-3, 1e-1, 1e1]
             }},
            'MLP': {
            'function': MLPRegressor,
            'params': { 'max_iter': 500, 'verbose': 0 },
            'search_params': {
              'learning_rate': ["invscaling"],
              'learning_rate_init': [0.001, 0.01, 0.1],
              'hidden_layer_sizes': [(25,), (50), (100,), (150,), (50,25), (50,50), (100,50), (100, 100)],
              'activation': ["logistic", "relu", "tanh"]
            }},
            'XGBOOST': {
            'function': XGBRegressor,
            'params': { 'objective': 'reg:squarederror', 'n_jobs': -1 },
            'search_params': {
              'n_estimators': [50, 100, 150],
              'learning_rate': [0.01, 0.1, 0.5],
              'max_depth': [1,3,5,7],
              'booster': ['gbtree', 'gblinear']
            }}}