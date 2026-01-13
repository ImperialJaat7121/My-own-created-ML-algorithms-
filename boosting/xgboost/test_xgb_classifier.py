import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost_classifier import XGBoostClassifier

# Optional: Try to import real xgboost for comparison
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def test():
    print("\n--- 🧪 Testing XGBoost Classifier ---")
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Own Implementation
    print("Training 'Own' XGBoost Classifier...")
    clf = XGBoostClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Own Accuracy: {acc:.4f}")

    if HAS_XGB:
        print("Training Official XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, reg_lambda=1.0, random_state=42)
        xgb_clf.fit(X_train, y_train)
        real_acc = accuracy_score(y_test, xgb_clf.predict(X_test))
        print(f"✅ Real Accuracy: {real_acc:.4f}")

if __name__ == "__main__":
    test()