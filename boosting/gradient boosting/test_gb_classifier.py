import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as SklearnGB
from gradient_boosting_classifier import GradientBoostingClassifier

def test():
    print("\n--- 🧪 Testing Gradient Boosting Classifier ---")
    
    # Load Breast Cancer Data
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Own Implementation
    print("Training 'Own' Gradient Boosting...")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Own Accuracy:     {acc:.4f}")

    # 2. Scikit-Learn Comparison
    print("Training Scikit-Learn Gradient Boosting...")
    sk_clf = SklearnGB(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    sk_clf.fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, sk_clf.predict(X_test))
    print(f"✅ Sklearn Accuracy: {sk_acc:.4f}")

    if abs(acc - sk_acc) < 0.05:
        print("\n🎉 SUCCESS: Your implementation is solid.")
    else:
        print("\n⚠️ NOTE: Check hyperparams. (GBM is sensitive to initialization).")

if __name__ == "__main__":
    test()