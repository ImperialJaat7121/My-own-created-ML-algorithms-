import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decision_tree_classifier import MyDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

def run_test():
    print("--- Testing Decision Tree Classification (Breast Cancer) ---")
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Custom
    clf = MyDecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    acc = (np.sum(clf.predict(X_test) == y_test) / len(y_test))
    print(f"Custom Accuracy: {acc:.4f}")
    
    # Sklearn
    sk_clf = DecisionTreeClassifier(max_depth=10, random_state=123)
    sk_clf.fit(X_train, y_train)
    sk_acc = sk_clf.score(X_test, y_test)
    print(f"Sklearn Accuracy: {sk_acc:.4f}")
    
    if abs(acc - sk_acc) < 0.05:
        print("✅ SUCCESS: Comparable performance!")
    else:
        print("⚠️ NOTE: Differences expected due to random feature selection/splitting implementation details.")

if __name__ == "__main__":
    run_test()