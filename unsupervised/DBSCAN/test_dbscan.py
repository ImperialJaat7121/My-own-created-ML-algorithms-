import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from dbscan import DBSCAN

def test():
    print("\n--- 🧪 Testing DBSCAN ---")
    
    # Generate Moons (Standard test for DBSCAN)
    X, y_true = make_moons(n_samples=200, noise=0.1, random_state=42)

    # Parameters
    eps = 0.25
    min_samples = 5

    # 1. Own Implementation
    print("Running Own DBSCAN...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = db.fit_predict(X)
    
    # Count clusters (excluding noise -1)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    print(f"Own: Found {n_clusters} clusters and {n_noise} noise points.")

    # 2. Scikit-Learn Comparison
    print("Running Sklearn DBSCAN...")
    sk_db = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    sk_y_pred = sk_db.fit_predict(X)
    
    sk_n_clusters = len(set(sk_y_pred)) - (1 if -1 in sk_y_pred else 0)
    sk_n_noise = list(sk_y_pred).count(-1)
    print(f"Sklearn: Found {sk_n_clusters} clusters and {sk_n_noise} noise points.")

    # Compare Similarity
    score = adjusted_rand_score(sk_y_pred, y_pred)
    print(f"✅ Similarity (ARI Score): {score:.4f}")

    if score > 0.95:
        print("\n🎉 SUCCESS: Results match Scikit-Learn.")
    else:
        print("\n⚠️ NOTE: Results differ (Check distance calculation or edge cases).")

if __name__ == "__main__":
    test()