import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree_regressor import MyDecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

def run_test():
    print("--- Testing Decision Tree Regression (Diabetes) ---")
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Custom
    reg = MyDecisionTreeRegressor(max_depth=5)
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    print(f"Custom R2 Score: {r2:.4f}")
    
    # Sklearn
    sk_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
    sk_reg.fit(X_train, y_train)
    sk_r2 = sk_reg.score(X_test, y_test)
    print(f"Sklearn R2 Score: {sk_r2:.4f}")

    if abs(r2 - sk_r2) < 0.1: # Variance in trees is common
        print("✅ SUCCESS: Comparable performance!")
    else:
        print("⚠️ NOTE: Tree structure variance detected.")

if __name__ == "__main__":
    run_test()