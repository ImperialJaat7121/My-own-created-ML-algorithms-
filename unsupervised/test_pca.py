import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA as SklearnPCA
from pca import PCA

def test():
    print("\n--- 🧪 Testing PCA ---")
    
    # Load Iris Data (4 dimensions)
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # 1. Own Implementation
    print("Running Own PCA...")
    pca = PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)
    
    print(f"Original Shape: {X.shape}")
    print(f"Projected Shape: {X_projected.shape}")
    print(f"Explained Variance Ratio (Own): {pca.explained_variance_ratio()}")

    # 2. Scikit-Learn Comparison
    print("\nRunning Sklearn PCA...")
    sk_pca = SklearnPCA(n_components=2)
    sk_pca.fit(X)
    sk_X_projected = sk_pca.transform(X)
    print(f"Explained Variance Ratio (Sklearn): {sk_pca.explained_variance_ratio_}")

    # Check Validity
    # Note: Eigenvectors can have arbitrary signs (+/-). 
    # If our values match sklearn's values (even if negative), the math is correct.
    diff = np.abs(pca.explained_variance_ratio() - sk_pca.explained_variance_ratio_)
    if np.all(diff < 1e-5):
        print("\n🎉 SUCCESS: Explained Variance matches Scikit-Learn exactly.")
    else:
        print("\n⚠️ NOTE: Variance calculation differs slightly.")

if __name__ == "__main__":
    test()