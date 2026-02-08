import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from hierarchical import HierarchicalClustering

def test():
    print("\n--- 🧪 Testing Hierarchical Clustering ---")
    
    # Generate Data
    X, y_true = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    # 1. Own Implementation (Single Linkage)
    print("Running Own Hierarchical (Single Linkage)...")
    hc = HierarchicalClustering(n_clusters=3, linkage='single')
    y_pred = hc.fit_predict(X)
    
    # 2. Scikit-Learn Comparison
    print("Running Sklearn Agglomerative...")
    sk_hc = AgglomerativeClustering(n_clusters=3, linkage='single')
    sk_y_pred = sk_hc.fit_predict(X)

    # Compare using Adjusted Rand Index (ARI)
    # ARI measures similarity between two clusterings (1.0 is perfect match, 0.0 is random)
    score = adjusted_rand_score(sk_y_pred, y_pred)
    print(f"✅ Similarity to Sklearn (ARI Score): {score:.4f}")

    if score > 0.95:
        print("\n🎉 SUCCESS: Implementation logic matches Scikit-Learn.")
    else:
        print("\n⚠️ NOTE: Results differ.")

if __name__ == "__main__":
    test()