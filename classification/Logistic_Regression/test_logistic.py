import numpy as np
import os, sys, traceback
# ensure repo root is on sys.path so package imports work when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from Own_ml_algorithms.classification.Logistic_Regression.Logistic_regression import LogisticRegression
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
    from sklearn.linear_model import LogisticRegression as SklearnLogistic
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] Scikit-Learn not found. Comparison will be skipped.")


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def test_logistic_regression():
    print("\n" + "="*60)
    print("TEST: LOGISTIC REGRESSION (Classification)")
    print("="*60)
    
    if SKLEARN_AVAILABLE:
        X, y = make_classification(n_samples=500, n_features=4, n_informative=2,n_redundant=0, n_clusters_per_class=1, random_state=42)
    else:
        np.random.seed(42)
        X = np.random.randn(500, 4)
        scores = X[:, 0] + X[:, 1]
        y = (scores > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(learning_rate=0.1, n_iterations=3000, fit_intercept=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test, threshold=0.5)
    acc = calculate_accuracy(y_test, preds)
    print(f"\n[Results - Custom Model]")
    print(f"Learned Weights: {model.coef_}")
    print(f"Learned Bias:    {model.intercept_:.4f}")
    print(f"Accuracy:        {acc * 100:.2f}%")

    if SKLEARN_AVAILABLE:
        print(f"\n[Comparison - Scikit-Learn]")
        sk_model = SklearnLogistic(penalty=None, solver='lbfgs', max_iter=3000)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        print(f"Sklearn Weights: {sk_model.coef_.flatten()}")
        print(f"Sklearn Bias:    {sk_model.intercept_[0]:.4f}")
        print(f"Sklearn Accuracy:{sk_acc * 100:.2f}%")

if __name__ == "__main__":
    test_logistic_regression()