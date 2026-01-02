import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor as SklearnAdaBoost
from adaboost_regressor import AdaBoostRegressor

def test():
    print("\n--- 🧪 Testing AdaBoost Regressor ---")
    
    # Load Diabetes Data
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Own Implementation
    print("Training 'Own' Regressor...")
    reg = AdaBoostRegressor(n_estimators=50, learning_rate=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ Own MSE:     {mse:.4f}")

    # 2. Scikit-Learn Comparison
    print("Training Scikit-Learn Regressor...")
    sk_reg = SklearnAdaBoost(n_estimators=50, learning_rate=1.0, random_state=42)
    sk_reg.fit(X_train, y_train)
    sk_mse = mean_squared_error(y_test, sk_reg.predict(X_test))
    print(f"✅ Sklearn MSE: {sk_mse:.4f}")

if __name__ == "__main__":
    test()