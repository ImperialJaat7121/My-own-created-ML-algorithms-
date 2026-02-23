import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette
from silhouette import silhouette_score as own_silhouette

def test():
    print("\n--- 🧪 Testing Silhouette Score (Grand Finale) ---")
    
    # Generate random data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

    # Use Sklearn's KMeans just to get some quick labels to test our metric
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # 1. Own Implementation
    print("Calculating Own Silhouette Score...")
    score_own = own_silhouette(X, labels)
    print(f"✅ Own Score:     {score_own:.5f}")

    # 2. Scikit-Learn Comparison
    print("Calculating Sklearn Silhouette Score...")
    score_sk = sk_silhouette(X, labels)
    print(f"✅ Sklearn Score: {score_sk:.5f}")

    if abs(score_own - score_sk) < 1e-5:
        print("\n🎉 GRAND FINALE SUCCESS: Your math is flawless.")
    else:
        print("\n⚠️ NOTE: Math discrepancy detected.")

if __name__ == "__main__":
    test()