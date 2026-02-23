import numpy as np

def silhouette_score(X, labels):
    """
    Calculates the mean Silhouette Coefficient of all samples from scratch.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    
    # If there's only 1 cluster, or every point is its own cluster, score is 0
    if len(unique_labels) <= 1 or len(unique_labels) == n_samples:
        return 0.0
        
    s_i = np.zeros(n_samples)
    
    for i in range(n_samples):
        own_cluster = labels[i]
        
        # 1. Calculate a(i) - Intra-cluster distance
        own_cluster_points = X[labels == own_cluster]
        if len(own_cluster_points) > 1:
            # Distance to all other points in the same cluster
            distances = np.linalg.norm(own_cluster_points - X[i], axis=1)
            # Subtract 1 because distance to itself is 0, but we shouldn't count it in the mean
            a_i = np.sum(distances) / (len(own_cluster_points) - 1)
        else:
            # If it's a cluster of 1 point, s(i) is 0 by definition
            a_i = 0.0
            
        # 2. Calculate b(i) - Nearest-cluster distance
        b_i = float('inf')
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
                
            other_cluster_points = X[labels == other_cluster]
            # Mean distance to all points in this other cluster
            mean_dist_to_other = np.mean(np.linalg.norm(other_cluster_points - X[i], axis=1))
            
            # Keep the minimum distance (the nearest neighboring cluster)
            if mean_dist_to_other < b_i:
                b_i = mean_dist_to_other
                
        # 3. Calculate s(i) - Silhouette Coefficient for this point
        if len(own_cluster_points) == 1:
            s_i[i] = 0.0
        else:
            s_i[i] = (b_i - a_i) / max(a_i, b_i)
            
    # Return the global mean
    return np.mean(s_i)