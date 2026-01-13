import numpy as np

class XGBoostNode:
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, value=None):
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value

    def is_leaf(self):
        return self.value is not None

class XGBoostTree:
    """
    A single tree that learns from Gradients (g) and Hessians (h).
    It maximizes the Structure Score (Gain).
    """
    def __init__(self, max_depth=3, min_child_weight=1, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight # Min sum of hessians needed in a leaf
        self.reg_lambda = reg_lambda # L2 regularization
        self.gamma = gamma # Minimum gain required to split
        self.root = None

    def fit(self, X, g, h):
        self.root = self._build_tree(X, g, h, depth=0)

    def _build_tree(self, X, g, h, depth):
        # 1. Calculate Leaf Weight (Prediction)
        # Formula: w = - Sum(g) / (Sum(h) + lambda)
        G = np.sum(g)
        H = np.sum(h)
        leaf_value = -G / (H + self.reg_lambda)
        
        # Stopping Criteria
        if depth >= self.max_depth or H < self.min_child_weight:
            return XGBoostNode(value=leaf_value)

        # 2. Find Best Split
        best_gain = -float('inf')
        best_split = None 
        
        n_features = X.shape[1]
        
        # Current score (Structure Score before splitting)
        current_score = (G**2) / (H + self.reg_lambda)

        for feat_idx in range(n_features):
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            
            for thresh in thresholds:
                # Create masks
                left_mask = X_col <= thresh
                right_mask = X_col > thresh
                
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Calculate G and H for children
                g_l, h_l = g[left_mask], h[left_mask]
                g_r, h_r = g[right_mask], h[right_mask]
                
                G_l, H_l = np.sum(g_l), np.sum(h_l)
                G_r, H_r = np.sum(g_r), np.sum(h_r)

                # XGBoost Gain Formula
                # Gain = 0.5 * [ (GL^2 / (HL+lam)) + (GR^2 / (HR+lam)) - (G_total^2 / (H_total+lam)) ] - gamma
                score_l = (G_l**2) / (H_l + self.reg_lambda)
                score_r = (G_r**2) / (H_r + self.reg_lambda)
                
                gain = 0.5 * (score_l + score_r - current_score) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat_idx, thresh)

        # 3. Create Node or Leaf
        if best_gain > 0:
            feat_idx, thresh = best_split
            left_mask = X[:, feat_idx] <= thresh
            right_mask = X[:, feat_idx] > thresh
            
            left_child = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
            right_child = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)
            
            return XGBoostNode(left=left_child, right=right_child, feature_idx=feat_idx, threshold=thresh)
        else:
            return XGBoostNode(value=leaf_value)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)