import numpy as np
import sys

class PolynomialRegression:
    
    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias
        
    def fit_transform(self, X):

        if X.ndim == 1:
            n_samples = X.shape[0]
            n_features = 1
            X = X.reshape(-1, 1)
        else:
            n_samples, n_features = X.shape
            
        features_list = []
        
        for d in range(1, self.degree + 1):
            for i in range(n_features):
                power_col = X[:, i] ** d
                features_list.append(power_col.reshape(-1, 1))
                
        X_poly = np.hstack(features_list)
        
        if self.include_bias:
            ones = np.ones((n_samples, 1))
            X_poly = np.hstack((ones, X_poly))
            
        return X_poly