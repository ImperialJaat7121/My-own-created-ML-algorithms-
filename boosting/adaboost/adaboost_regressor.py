import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trees.decisiontree.decision_tree_regressor import MyDecisionTreeRegressor

class AdaBoostRegressor:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.clfs = []
        self.betas = [] 

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialize weights uniformly
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        self.betas = []

        for _ in range(self.n_estimators):
            # 1. Resampling: Pick samples based on weights
            indices = np.random.choice(n_samples, n_samples, replace=True, p=w)
            X_sample = X[indices]
            y_sample = y[indices]

            # 2. Train Regressor Stump
            clf = MyDecisionTreeRegressor(max_depth=3) 
            clf.fit(X_sample, y_sample)
            
            # 3. Predict on ORIGINAL data
            predictions = clf.predict(X)
            
            # 4. Calculate Loss
            abs_errors = np.abs(y - predictions)
            max_error = np.max(abs_errors)
            
            if max_error == 0:
                # Perfect fit, stop early
                self.clfs.append(clf)
                self.betas.append(0)
                break 
                
            norm_errors = abs_errors / max_error # Normalize errors to [0,1]
            
            # Average Loss
            avg_loss = np.sum(w * norm_errors)
            
            if avg_loss >= 0.5:
                # Model is too weak, stop
                break 
            
            # 5. Calculate Beta (Confidence)
            beta = avg_loss / (1.0 - avg_loss)
            self.betas.append(beta)
            self.clfs.append(clf)

            # 6. Update Weights
            # Weight increases for higher errors. 
            # Formula: w_new = w_old * beta^(1 - error)
            w *= np.power(beta, (1.0 - norm_errors) * self.learning_rate)
            w /= np.sum(w)

    def predict(self, X):
        
        n_samples = X.shape[0]
        
        # Get all predictions: shape (n_samples, n_estimators)
        preds = np.array([clf.predict(X) for clf in self.clfs]).T 
        
        # Weights are ln(1/beta)
        # Handle beta=0 case safely
        weights = np.array([np.log(1.0 / (b + 1e-10)) for b in self.betas])
        
        final_preds = []
        for i in range(n_samples):
            sample_preds = preds[i]
            
            # Sort predictions and weights based on prediction value
            sorted_idx = np.argsort(sample_preds)
            sorted_preds = sample_preds[sorted_idx]
            sorted_weights = weights[sorted_idx]
            
            # Find Weighted Median
            cum_weights = np.cumsum(sorted_weights)
            cutoff = 0.5 * cum_weights[-1] # 50% of total weight
            median_idx = np.searchsorted(cum_weights, cutoff)
            
            final_preds.append(sorted_preds[median_idx])
            
        return np.array(final_preds)