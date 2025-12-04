import numpy as np
import sys

class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.coef_ = None      
        self.intercept_ = None 
        self.classes_ = None

    def _sigmoid(self, z):
        # Using np.clip to avoid overflow in exp for very small negative numbers
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):

        # 1. Setup
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        # 2. Gradient Descent Loop
        for i in range(self.n_iterations):

            linear_model = np.dot(X, self.coef_)
            if self.fit_intercept:
                linear_model += self.intercept_
            y_predicted = self._sigmoid(linear_model)

            error = y_predicted - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update Parameters
            self.coef_ -= self.learning_rate * dw
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.coef_)
        if self.fit_intercept:
            linear_model += self.intercept_
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
