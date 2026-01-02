import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost

# Import your scratch implementation
from adaboost_classifier import AdaBoostClassifier as CustomAdaBoost

def test():
    print("Loading Breast Cancer dataset...")
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    y_scaled = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

    # --- Custom Implementation ---
    print("\n--- Training Custom AdaBoost Classifier ---")
    clf = CustomAdaBoost(n_estimators=50, learning_rate=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Custom AdaBoost Classifier Accuracy: {acc:.4f}")

    # --- Scikit-Learn Implementation ---
    print("\n--- Training Scikit-Learn AdaBoost Classifier ---")
    sk_clf = SklearnAdaBoost(n_estimators=50, learning_rate=1.0, random_state=42)
    sk_clf.fit(X_train, y_train)
    sk_y_pred = sk_clf.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_y_pred)
    print(f"Sklearn AdaBoost Classifier Accuracy: {sk_acc:.4f}")

    print("-" * 30)
    if abs(acc - sk_acc) < 0.05:
        print("SUCCESS: Your implementation is comparable to Scikit-Learn!")
    else:
        print("NOTE: Performance gap detected (check hyperparameters or randomness).")

if __name__ == "__main__":
    test()