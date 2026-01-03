import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trees.decisiontree.decision_tree_classifier import MyDecisionTreeClassifier

class GradientBoostingClassifier:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_log_odds = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # 1. Initialize with log-odds of the positive class
        pos_prob = np.mean(y)
        # Avoid div by zero
        pos_prob = np.clip(pos_prob, 1e-10, 1 - 1e-10)
        self.initial_log_odds = np.log(pos_prob / (1 - pos_prob))
        
        # Current raw predictions (log-odds)
        raw_predictions = np.full(y.shape, self.initial_log_odds)

        for _ in range(self.n_estimators):
            # 2. Calculate Probabilities
            probabilities = self._sigmoid(raw_predictions)
            
            # 3. Calculate Residuals (Negative Gradient)
            residuals = y - probabilities
            
            # 4. Fit a Regressor to the Residuals
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 5. Update Predictions
            gradient_step = tree.predict(X)
            raw_predictions += self.learning_rate * gradient_step
            
            self.trees.append(tree)

    def predict_proba(self, X):
        # Start with initial guess
        raw_predictions = np.full(X.shape[0], self.initial_log_odds)
        
        # Add contribution from every tree
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X)
            
        return self._sigmoid(raw_predictions)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)