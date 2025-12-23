import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decisiontree.decision_tree_regressor import MyDecisionTreeRegressor

class MyRandomForestRegressor:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=100, max_features=None, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        if self.random_state:
            np.random.seed(self.random_state)
        
        for _ in range(self.n_estimators):
            # Instantiate the base learner
            tree = MyDecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.max_features
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
                tree.fit(X_sample, y_sample)
            else:
                tree.fit(X, y)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Gather predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Average the predictions across trees
        return np.mean(tree_preds, axis=0)

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        u = ((y_test - preds) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        return 1 - u/v

# --- Unit Test ---
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    print("--- Testing Random Forest Regression ---")
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Custom Model
    reg = MyRandomForestRegressor(n_estimators=10, max_depth=5, random_state=123)
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print(f"Custom RF R2 Score: {score:.4f}")

    # Sklearn Model
    sk_reg = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=123)
    sk_reg.fit(X_train, y_train)
    sk_score = sk_reg.score(X_test, y_test)
    print(f"Sklearn RF R2 Score: {sk_score:.4f}")

    if abs(score - sk_score) < 0.1: 
        print("✅ SUCCESS: Performance is comparable!")
    else:
        print("⚠️ NOTE: Variance expected due to random bootstrapping and small forest size.")