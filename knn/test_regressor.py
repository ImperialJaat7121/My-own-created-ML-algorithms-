import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn_regressor import MyKNNRegressor

def run_test():
    print("--- Testing KNN Regression (Diabetes Dataset) ---")
    
    # 1. Load Data
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # 3. Train Custom Model
    k_val = 5
    reg = MyKNNRegressor(k=k_val)
    reg.fit(X_train, y_train)
    
    # 4. Evaluate
    r2 = reg.score(X_test, y_test)
    print(f"Custom KNN Regressor (k={k_val}) R2 Score: {r2:.4f}")
    
    # 5. Sanity Check against Sklearn
    from sklearn.neighbors import KNeighborsRegressor
    sk_reg = KNeighborsRegressor(n_neighbors=k_val)
    sk_reg.fit(X_train, y_train)
    sk_r2 = sk_reg.score(X_test, y_test)
    print(f"Sklearn KNN Regressor (k={k_val}) R2 Score: {sk_r2:.4f}")

    if abs(r2 - sk_r2) < 1e-9:
        print("✅ SUCCESS: Matches Scikit-Learn implementation!")
    else:
        print("⚠️ WARNING: Results differ.")

if __name__ == "__main__":
    run_test()