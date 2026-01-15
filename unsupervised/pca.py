import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) from scratch.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None

    def fit(self, X):
        # 1. Mean Centering
        # PCA is sensitive to scale, so centering is crucial.
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # 2. Covariance Matrix
        # Needs samples as columns for np.cov, so we transpose
        # cov shape: (n_features, n_features)
        cov = np.cov(X.T)

        # 3. Eigen Decomposition
        # vectors shape: (n_features, n_features)
        # values shape: (n_features,)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 4. Sort Eigenvectors
        # We want the ones with highest eigenvalues (most variance)
        # argsort gives ascending order, so we flip it [::-1]
        idxs = np.argsort(eigenvalues)[::-1]
        
        self.eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors.T # Transpose for easier indexing
        sorted_components = eigenvectors[idxs]

        # 5. Store first n_components
        self.components = sorted_components[0:self.n_components]

    def transform(self, X):
        # Project data onto our new axes
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self):
        # Returns the percentage of variance explained by each selected component
        total_var = np.sum(self.eigenvalues)
        return self.eigenvalues[:self.n_components] / total_var