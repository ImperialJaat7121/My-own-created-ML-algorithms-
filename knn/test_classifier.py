import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn_classifier import MyKNNClassifier

def run_test():
    print("--- Testing KNN Classification (Iris Dataset) ---")
    
    # 1. Load Data
    data = datasets.load_iris()
    X, y = data.data, data.target
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # 3. Train Custom Model
    k_val = 3
    clf = MyKNNClassifier(k=k_val)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    acc = clf.score(X_test, y_test)
    print(f"Custom KNN (k={k_val}) Accuracy: {acc:.4f}")
    
    # 5. Sanity Check against Sklearn
    from sklearn.neighbors import KNeighborsClassifier
    sk_clf = KNeighborsClassifier(n_neighbors=k_val)
    sk_clf.fit(X_train, y_train)
    sk_acc = sk_clf.score(X_test, y_test)
    print(f"Sklearn KNN (k={k_val}) Accuracy: {sk_acc:.4f}")
    
    if acc == sk_acc:
        print("✅ SUCCESS: Matches Scikit-Learn implementation!")
    else:
        print("⚠️ WARNING: Results differ.")

if __name__ == "__main__":
    run_test()