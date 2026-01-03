import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trees.decisiontree.decision_tree_regressor import MyDecisionTreeRegressor

class GradientBoostingRegressor:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # 1. Initialize with the mean of the target
        self.initial_prediction = np.mean(y)
        
        # Current predictions
        predictions = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            # 2. Calculate Residuals (True - Pred)
            residuals = y - predictions
            
            # 3. Fit a tree to predict the RESIDUALS
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # 4. Update predictions
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            self.trees.append(tree)

    def predict(self, X):
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add up all the residual fixes
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions