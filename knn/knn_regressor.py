import numpy as np

class MyKNNRegressor:
    def __init__(self, k=3):
        """
        k: Number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict values for multiple samples in X."""
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Helper to predict for a single sample x."""
        # 1. Compute distances
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # 2. Get K nearest indices
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Get values of neighbors
        k_nearest_values = [self.y_train[i] for i in k_indices]
        
        # 4. Return Mean (Average)
        return np.mean(k_nearest_values)

    def score(self, X_test, y_test):
        """Returns R-squared score."""
        preds = self.predict(X_test)
        u = ((y_test - preds) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        return 1 - u/v