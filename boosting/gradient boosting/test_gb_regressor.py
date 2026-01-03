import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor as SklearnGB
from gradient_boosting_regressor import GradientBoostingRegressor

def test():
    print("\n--- 🧪 Testing Gradient Boosting Regressor ---")
    
    # Load Diabetes Data (Regression task)
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Own Implementation
    print("Training 'Own' Gradient Boosting...")
    reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ Own MSE:     {mse:.4f}")

    # 2. Scikit-Learn Comparison
    print("Training Scikit-Learn Gradient Boosting...")
    sk_reg = SklearnGB(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    sk_reg.fit(X_train, y_train)
    sk_mse = mean_squared_error(y_test, sk_reg.predict(X_test))
    print(f"✅ Sklearn MSE: {sk_mse:.4f}")

    if abs(mse - sk_mse) < 500: # MSE scale for diabetes is around 3000
        print("\n🎉 SUCCESS: Comparable Error Rates.")
    else:
        print("\n⚠️ NOTE: Large deviation detected.")

if __name__ == "__main__":
    test()