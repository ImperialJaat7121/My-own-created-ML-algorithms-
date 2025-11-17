import numpy as np 
import sys
import time 

class SimpleLinearRegression:
    def __init__(self, fit_intercept=True, positive=False):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.coef_ = None 
        self.intercept_ = None
        self.n_features_in_ = None
        self.rank_ = None
        self.singular= None
        self._w = None
        self._b = None

    def fit(self, X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6, clip_grad=1e6, scale=True):

        X = np.array(X, dtype=np.float64).reshape(-1)
        y = np.array(y, dtype=np.float64).reshape(-1)
        n_samples = X.shape[0]

        if scale:
            self._X_mean = X.mean()
            self._X_std = X.std() if X.std() > 0 else 1.0
            X_scaled = (X - self._X_mean) / self._X_std
        else:
            X_scaled = X

        w = 0.0
        b = 0.0
        old_cost = np.inf

        for i in range(1, n_iterations + 1):
            y_pred = w * X_scaled + b
            error = y - y_pred

            cost = (1.0 / (2 * n_samples)) * np.sum(error ** 2)
            if not np.isfinite(cost):
                print(f"[fit] Stopping early: non-finite cost at iteration {i}: {cost}")
                break

            # gradients
            dw = -(1.0 / n_samples) * np.sum(error * X_scaled)
            db = -(1.0 / n_samples) * np.sum(error)

            dw = np.clip(dw, -clip_grad, clip_grad)
            db = np.clip(db, -clip_grad, clip_grad)

            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0 or i == 1:
                print(f"Iteration {i:04d}: Cost = {cost:.6f}, w = {w:.6f}, b = {b:.6f}")

            if abs(old_cost - cost) < tol:
                break
            old_cost = cost

        if scale:
            self.coef_ = w / self._X_std
            self.intercept_ = b - (w * self._X_mean / self._X_std)
        else:
            self.coef_ = w
            self.intercept_ = b

    def predict(self, X):
        X = np.array(X, dtype=np.float64)

        if X.ndim == 0:
            X = X.reshape(1)

        coef = getattr(self, "coef_", None)
        intercept = getattr(self, "intercept_", 0.0)
        if coef is None:
            raise ValueError("Model is not trained. 'coef_' is missing.")

        if np.isscalar(coef):
            return coef * X + intercept
        coef_arr = np.array(coef, dtype=np.float64)
        if X.ndim == 1:
            return X.dot(coef_arr) + intercept
        else:
            return X @ coef_arr + intercept