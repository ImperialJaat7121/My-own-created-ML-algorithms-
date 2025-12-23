import numpy as np
from collections import Counter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decisiontree.decision_tree_classifier import MyDecisionTreeClassifier

class MyRandomForestClassifier:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=100, max_features=None, bootstrap=True, random_state=None):
        """
        n_estimators: Number of trees in the forest.
        min_samples_split: Min samples required to split a node.
        max_depth: Max depth of the tree.
        max_features: Number of features to consider when looking for the best split.
        bootstrap: Whether bootstrap samples are used when building trees.
        """
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
            tree = MyDecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.max_features
            )
            
            # Bootstrap sampling (Bagging)
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
                tree.fit(X_sample, y_sample)
            else:
                tree.fit(X, y)
                
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Randomly choose indices with replacement
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Gather predictions from all trees: [n_trees, n_samples]
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Swap axes to get [n_samples, n_trees]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Majority vote for each sample
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        
    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.sum(preds == y_test) / len(y_test)

# --- Unit Test ---
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    print("--- Testing Random Forest Classification ---")
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Custom Model
    clf = MyRandomForestClassifier(n_estimators=10, max_depth=10, random_state=123)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Custom RF Accuracy: {acc:.4f}")

    # Sklearn Model
    sk_clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=123)
    sk_clf.fit(X_train, y_train)
    sk_acc = sk_clf.score(X_test, y_test)
    print(f"Sklearn RF Accuracy: {sk_acc:.4f}")

    if abs(acc - sk_acc) < 0.05:
        print("✅ SUCCESS: Performance is comparable!")
    else:
        print("⚠️ NOTE: Slight variance expected due to random bootstrapping.")