import numpy as np

class KMeans:
    """
    K-Means Clustering from scratch.
    """
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps # Useful if we want to visualize animation later

        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # The centers (mean vector) for each cluster
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # 1. Initialize Centroids
        # Pick K random samples from the dataset to be initial centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization Loop
        for _ in range(self.max_iters):
            # 2. Assign samples to closest centroids (Create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()

            # 3. Update centroids based on new clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # 4. Check for Convergence
            # If centroids didn't change, we are done
            if self._is_converged(centroids_old, self.centroids):
                break
        
        # Return cluster labels
        return self._get_cluster_labels(self.clusters)

    def predict(self, X):
        # Assign new samples to existing centroids
        # Note: This implementation assumes fit() has been called
        # For pure prediction, we just find the nearest centroid
        labels = []
        for sample in X:
            distances = [self._euclidean_distance(sample, point) for point in self.centroids]
            closest_idx = np.argmin(distances)
            labels.append(closest_idx)
        return np.array(labels)

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [self._euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            # Calculate mean of all samples in this cluster
            # Note: handle empty clusters to avoid NaN
            if len(cluster) > 0:
                cluster_mean = np.mean(self.X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
            else:
                # If a cluster is empty, re-initialize it randomly or keep old pos
                centroids[cluster_idx] = self.centroids[cluster_idx]
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        # Check distances between old and new centroids
        distances = [self._euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels