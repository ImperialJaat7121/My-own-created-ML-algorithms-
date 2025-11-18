import numpy as np 
import sys

class MultipleLinearRegression:
    def __init__(self, fit_intercept=True, positive=False):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.coef_ = None       
        self.intercept_ = None  
        self.n_features_in_ = None

    def fit(self, X, y, Learning_rate=0.01, n_iterations=1000, tol=1e-4):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.coef_ = np.zeros(n_features) 
        self.intercept_ = 0.0 if self.fit_intercept else 0.0

        old_cost = float('inf')

        for i in range(n_iterations):
            y_predicted = np.dot(X, self.coef_)

            if self.fit_intercept:
                y_predicted += self.intercept_
        
            error = y_predicted - y
            cost = (1 / (2 * n_samples)) * np.sum(error**2)
        
            if abs(old_cost - cost) < tol:
                print(f"[LOG] Converged at iteration {i}. Cost: {cost:.6f}")
                break
            old_cost = cost

            dw = (1 / n_samples) * np.dot(X.T, error)

            db = 0.0
            if self.fit_intercept:
                db = (1 / n_samples) * np.sum(error)

            self.coef_ -= Learning_rate * dw
            
            if self.fit_intercept:
                self.intercept_ -= Learning_rate * db
            
            if self.positive:
                self.coef_ = np.maximum(0, self.coef_)

            if i % (n_iterations // 10) == 0:
                print(f"[LOG] Iter {i:04d}: Cost={cost:.5f}")

        print(f"[LOG] Training complete. Final Cost: {cost:.5f}")

    def predict(self, X):
        if self.coef_ is None:
            print("[ERROR] Model not fitted yet.", file=sys.stderr)
            return None
            
        y_predicted = np.dot(X, self.coef_)
        
        if self.fit_intercept:
            y_predicted += self.intercept_
            
        return y_predicted