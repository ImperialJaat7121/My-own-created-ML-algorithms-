import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class MyDecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # If no split reduces variance (e.g. all y are same), make leaf
        if best_feat is None:
             leaf_value = np.mean(y)
             return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_reduction = -1
        split_idx, split_thresh = None, None
        
        parent_var = np.var(y) * len(y)

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate Variance Reduction
                left_idxs, right_idxs = self._split(X_column, thr)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                var_l = np.var(y[left_idxs]) * n_l
                var_r = np.var(y[right_idxs]) * n_r
                
                child_var = var_l + var_r
                reduction = parent_var - child_var

                if reduction > best_reduction:
                    best_reduction = reduction
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        u = ((y_test - preds) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        return 1 - u/v