import numpy as np
from xgboost_tree import XGBoostTree

class XGBoostClassifier:
    """
    XGBoost Classifier from scratch.
    Objective: Binary Logistic Regression (Log Loss).
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.base_pred = 0.5 # Initial probability guess (log(odds) = 0)

    def fit(self, X, y):
        # Initial predictions (Log Odds)
        # Log(0.5 / (1-0.5)) = 0.0
        preds = np.zeros(y.shape) 

        for _ in range(self.n_estimators):
            # 1. Calculate Gradients and Hessians for Log Loss
            # Sigmoid transform to get probabilities
            p = 1 / (1 + np.exp(-preds))
            
            # Gradient: p - y
            g = p - y
            
            # Hessian: p * (1 - p)
            # This is the "Confidence" weight. Sure predictions have small hessian.
            h = p * (1 - p)

            # 2. Fit XGBoost Tree
            tree = XGBoostTree(
                max_depth=self.max_depth, 
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X, g, h)
            self.trees.append(tree)

            # 3. Update Log-Odds Predictions
            preds += self.learning_rate * tree.predict(X)

    def predict_proba(self, X):
        # Start with 0.0 log odds
        preds = np.zeros(X.shape[0])
        
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        
        # Sigmoid to convert back to probability
        return 1 / (1 + np.exp(-preds))

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)