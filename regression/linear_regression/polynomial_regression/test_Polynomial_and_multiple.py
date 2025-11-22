import numpy as np
import os, sys, traceback
# ensure repo root is on sys.path so package imports work when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from Own_ml_algorithms.regression.linear_regression.multiple_linear_regression.Multiple_LR import MultipleLinearRegression
    from Own_ml_algorithms.regression.linear_regression.polynomial_regression.Polynomial_Regression import PolynomialRegression as PolynomialFeatures
    from Own_ml_algorithms.preprocessing import train_test_split
    from Own_ml_algorithms.performance.regression import calculate_rmse, calculate_r_squared
except Exception:
    print("="*60)
    print("Failed to import project modules. Make sure you run this from the repository root or add the repo root to PYTHONPATH.")
    print("Import error details:")
    traceback.print_exc()
    print("="*60)
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import root_mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    print("Scikit-Learn not found. Comparison will be skipped.")

def test_polynomial_regression():
    print("\n" + "="*60)
    print("TEST B: POLYNOMIAL REGRESSION (Non-Linear Curve)")
    print("="*60)
    np.random.seed(42)
    m = 100
    X = 6 * np.random.rand(m, 1) - 3 
    y = 0.5 * X**2 + 2 + np.random.randn(m, 1) * 0.2
    y = y.flatten()

    print("Training Custom PolynomialRegression...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = MultipleLinearRegression(fit_intercept=True)
    model.fit(X_train, y_train, learning_rate=0.1, n_iterations=2000)
    
    preds = model.predict(X_test)
    r2 = calculate_r_squared(y_test, preds)
    rmse = calculate_rmse(y_test, preds)

    print(f"\n[Results - Custom Polynomial]")
    print(f"R-Squared: {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    
    if SKLEARN_AVAILABLE:
        print(f"\n[Comparison - Scikit-Learn]")
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        sk_r2 = r2_score(y_test, pred)
        sk_rmse = root_mean_squared_error(y_test, pred)

        print(f"Sklearn R2:   {sk_r2:.4f}")
        print(f"Sklearn RMSE: {sk_rmse:.4f}")

def test_multiple_linear_regression():
    print("\n" + "="*60)
    print("TEST A: MULTIPLE LINEAR REGRESSION (2 Features)")
    print("="*60)
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 2) * 10 
    
    true_w = np.array([4.0, -3.0])
    true_b = 10.0
    
    y = np.dot(X, true_w) + true_b + np.random.randn(m) * 0.5
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Custom MultipleLinearRegression...")
    model = MultipleLinearRegression(fit_intercept=True)
    model.fit(X_train, y_train, learning_rate=0.01, n_iterations=5000, tol=1e-6)
    
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n[Results - Custom Model]")
    print(f"Learned Weights: {model.coef_}")
    print(f"Learned Bias:   {model.intercept_:.4f}")
    print(f"R-Squared:    {r2:.4f}")
    print(f"RMSE:          {rmse:.4f}")

    if SKLEARN_AVAILABLE:
        print(f"\n[Comparison - Scikit-Learn]")
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        sk_r2 = r2_score(y_test, pred)
        sk_rmse = root_mean_squared_error(y_test, pred)

        print(f"Sklearn Weights: {model.coef_}")
        print(f"Sklearn Bias:    {model.intercept_:.4f}")
        print(f"Sklearn R2:      {sk_r2:.4f}")
        print(f"Sklearn RMSE:    {sk_rmse:.4f}")


if __name__ == "__main__":
    test_multiple_linear_regression()
    test_polynomial_regression()
    print("\n" + "="*60)