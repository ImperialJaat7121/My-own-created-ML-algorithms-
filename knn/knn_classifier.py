import numpy as np
from collections import Counter

class MyKNNClassifier:
    def __init__(self, k=3):
        """
        k: Number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        KNN is a lazy learner. 'Fitting' just means storing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict labels for multiple samples in X."""
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Helper to predict for a single sample x."""
        # 1. Compute distances between x and all examples in the training set
        # Using numpy's linalg.norm is efficient for Euclidean distance
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Majority vote (most common class)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X_test, y_test):
        """Returns accuracy score."""
        preds = self.predict(X_test)
        return np.sum(preds == y_test) / len(y_test)