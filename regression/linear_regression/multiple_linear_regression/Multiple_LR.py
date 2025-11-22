import numpy as np 
import sys

class MultipleLinearRegression:
    def __init__(self, fit_intercept=True, positive=False):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None

    def fit(self, X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0 if self.fit_intercept else 0.0

        old_cost = float('inf')

        for i in range(1, n_iterations + 1):
            y_pred = X.dot(self.coef_) + self.intercept_
            error = y_pred - y

            cost = (1.0 / (2 * n_samples)) * np.sum(error ** 2)
            if not np.isfinite(cost):
                if verbose:
                    print(f"[fit] Stopping: non-finite cost at iter {i}: {cost}")
                break

            dw = (1.0 / n_samples) * X.T.dot(error)
            db = (1.0 / n_samples) * np.sum(error)

            self.coef_ = self.coef_ - learning_rate * dw
            self.intercept_ = self.intercept_ - learning_rate * db

            if self.positive:
                self.coef_ = np.maximum(self.coef_, 0.0)

            if verbose and (i % 500 == 0 or i == 1):
                print(f"Iteration {i:04d}: Cost = {cost:.6f}")

            if abs(old_cost - cost) < tol:
                if verbose:
                    print(f"[fit] Converged at iteration {i}. Cost improvement < tol.")
                break
            old_cost = cost

        if verbose:
            print(f"[LOG] Training complete. Final Cost: {cost:.6f}")

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model is not trained. Call fit() first.")

        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X.dot(self.coef_) + self.intercept_