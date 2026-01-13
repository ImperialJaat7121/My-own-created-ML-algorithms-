import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost_regressor import XGBoostRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def test():
    print("\n--- 🧪 Testing XGBoost Regressor ---")
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Own Implementation
    print("Training 'Own' XGBoost Regressor...")
    reg = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ Own MSE: {mse:.4f}")

    if HAS_XGB:
        print("Training Official XGBoost Regressor...")
        xgb_reg = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=1.0, random_state=42)
        xgb_reg.fit(X_train, y_train)
        real_mse = mean_squared_error(y_test, xgb_reg.predict(X_test))
        print(f"✅ Real MSE: {real_mse:.4f}")

if __name__ == "__main__":
    test()