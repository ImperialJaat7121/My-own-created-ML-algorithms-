import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score
from kmeans import KMeans

def test():
    print("\n--- 🧪 Testing K-Means ---")
    
    # Generate Synthetic Data (Blobs)
    # 3 distinct centers, easy to cluster
    X, y_true = make_blobs(n_samples=500, centers=3, n_features=2, random_state=42)

    # 1. Own Implementation
    print("Running Own K-Means...")
    km = KMeans(K=3, max_iters=100)
    y_pred = km.fit(X)
    
    # We can't compare labels directly (0 could be 2), so we use Silhouette Score
    # Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
    # Range: -1 to 1 (Higher is better)
    score_own = silhouette_score(X, y_pred)
    print(f"✅ Own Silhouette Score: {score_own:.4f}")

    # 2. Scikit-Learn Comparison
    print("Running Sklearn K-Means...")
    sk_km = SklearnKMeans(n_clusters=3, random_state=42, n_init=10)
    sk_y_pred = sk_km.fit_predict(X)
    score_sk = silhouette_score(X, sk_y_pred)
    print(f"✅ Sklearn Silhouette Score: {score_sk:.4f}")

    if abs(score_own - score_sk) < 0.02:
        print("\n🎉 SUCCESS: Clustering quality matches Scikit-Learn.")
    else:
        print("\n⚠️ NOTE: Results differ (Check initialization randomness).")

if __name__ == "__main__":
    test()