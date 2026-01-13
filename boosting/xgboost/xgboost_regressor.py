import numpy as np
from xgboost_tree import XGBoostTree

class XGBoostRegressor:
    """
    XGBoost Regressor from scratch.
    Objective: Mean Squared Error (Linear Regression).
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        # Initial prediction: Mean of target
        self.base_pred = np.mean(y)
        preds = np.full(y.shape, self.base_pred)

        for _ in range(self.n_estimators):
            # 1. Calculate Gradients and Hessians for MSE
            # Loss = 0.5 * (preds - y)^2
            
            # Gradient (1st deriv): preds - y
            g = preds - y
            
            # Hessian (2nd deriv): 1
            # For MSE, curvature is constant!
            h = np.ones_like(y)

            # 2. Fit XGBoost Tree
            tree = XGBoostTree(
                max_depth=self.max_depth, 
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # 3. Update Predictions
            preds += self.learning_rate * tree.predict(X)

    def predict(self, X):
        preds = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds