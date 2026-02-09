import numpy as np

class DBSCAN:
    """
    DBSCAN Clustering from scratch.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit_predict(self, X):
        self.n_samples = X.shape[0]
        # Initialize labels to -1 (Noise)
        self.labels = np.full(self.n_samples, -1)
        visited = np.full(self.n_samples, False)
        
        cluster_id = 0

        for i in range(self.n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            
            # Find neighbors
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                # Mark as Noise (already -1), but it might be picked up later as a Border point
                continue
            else:
                # It's a Core Point -> Start a new cluster
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1
        
        return self.labels

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        # Assign the core point to the cluster
        self.labels[point_idx] = cluster_id
        
        # Create a queue to check all density-reachable points
        # We convert to list to allow dynamic appending
        queue = list(neighbors)
        
        i = 0
        while i < len(queue):
            neighbor_idx = queue[i]
            i += 1
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)
                
                # If this neighbor is ALSO a core point, add its neighbors to the queue
                if len(new_neighbors) >= self.min_samples:
                    queue.extend(new_neighbors)
            
            # If point was Noise (-1) or unassigned, assign it to this cluster
            # Note: A point marked as Noise previously can become a Border Point here.
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id

    def _region_query(self, X, point_idx):
        # Calculate distance from point_idx to all other points
        # Using Euclidean Distance
        distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
        
        # Return indices where distance <= eps
        return np.where(distances <= self.eps)[0]