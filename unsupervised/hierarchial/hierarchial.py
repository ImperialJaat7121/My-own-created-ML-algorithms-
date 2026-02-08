import numpy as np

class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering from scratch.
    """
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage # 'single' or 'complete'
        self.labels = None

    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # 1. Initialize each sample as a distinct cluster
        # Dictionary format: {cluster_id: [list of sample_indices]}
        self.clusters = {i: [i] for i in range(n_samples)}
        
        # 2. Compute Initial Distance Matrix (N x N)
        # We use a vectorized numpy approach for speed
        # Shape: (N, N)
        dist_matrix = self._compute_initial_distances(X)
        
        # Fill diagonal with infinity so clusters don't merge with themselves
        np.fill_diagonal(dist_matrix, np.inf)

        # 3. Merge Loop
        current_n_clusters = n_samples
        
        while current_n_clusters > self.n_clusters:
            # Find the indices (i, j) of the minimum distance in the matrix
            # unravel_index converts the flat index back to (row, col)
            min_dist_idx = np.argmin(dist_matrix)
            i, j = np.unravel_index(min_dist_idx, dist_matrix.shape)
            
            # Ensure i < j so we always merge into the lower index and remove the higher one
            if i > j:
                i, j = j, i
            
            # Merge cluster j into cluster i
            self.clusters[i].extend(self.clusters[j])
            del self.clusters[j]
            
            # 4. Update Distance Matrix (The Linkage Step)
            # We need to recalculate the distance from the new cluster 'i' to all other clusters 'k'
            # Rows/Cols for 'j' will be effectively removed (set to Infinity)
            
            for k in range(dist_matrix.shape[0]):
                if k != i and k != j and dist_matrix[i, k] != np.inf:
                    # Single Linkage: min(dist(i,k), dist(j,k))
                    if self.linkage == 'single':
                        new_dist = min(dist_matrix[i, k], dist_matrix[j, k])
                    
                    # Complete Linkage: max(dist(i,k), dist(j,k))
                    elif self.linkage == 'complete':
                        new_dist = max(dist_matrix[i, k], dist_matrix[j, k])
                    
                    # Update matrix symmetrically
                    dist_matrix[i, k] = new_dist
                    dist_matrix[k, i] = new_dist
            
            # "Remove" index j from matrix by setting it to Infinity
            dist_matrix[j, :] = np.inf
            dist_matrix[:, j] = np.inf
            
            current_n_clusters -= 1

        # 5. Convert Cluster Dict to Final Labels Array
        self.labels = np.zeros(n_samples, dtype=int)
        
        # Remap arbitrary cluster IDs (e.g., 0, 5, 12) to clean 0, 1, 2...
        for new_id, (old_id, sample_idxs) in enumerate(self.clusters.items()):
            self.labels[sample_idxs] = new_id
            
        return self.labels

    def _compute_initial_distances(self, X):
        # Vectorized Euclidean distance calculation
        # (x-y)^2 = x^2 + y^2 - 2xy
        # Note: For strict "from scratch", standard double loops work too, 
        # but this is much faster for the user.
        
        # Simple Loop version for readability/purity:
        n = X.shape[0]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((X[i] - X[j])**2))
                dists[i, j] = d
                dists[j, i] = d
        return dists