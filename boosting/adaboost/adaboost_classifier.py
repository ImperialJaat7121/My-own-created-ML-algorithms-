import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from trees.decisiontree.decision_tree_classifier import MyDecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.clfs = []
        self.alphas = []
        self.classes_ = None
        self.label_map_ = None
        self.inverse_label_map_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Convert labels to 0, 1, 2, ... format for decision tree compatibility
        self.classes_ = np.unique(y)
        self.label_map_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.inverse_label_map_ = {idx: label for label, idx in self.label_map_.items()}
        y_encoded = np.array([self.label_map_[label] for label in y])
        
        # Initialize weights uniformly (1/N)
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []
        self.alphas = []

        for _ in range(self.n_estimators):
            # 1. Resampling: Create a new dataset based on weights 'w'
            indices = np.random.choice(n_samples, n_samples, replace=True, p=w)
            X_sample = X[indices]
            y_sample = y_encoded[indices]

            # 2. Train Stump (Weak Learner)
            clf = MyDecisionTreeClassifier(max_depth=1)
            clf.fit(X_sample, y_sample)
            
            # 3. Predict on ORIGINAL data to calculate the true error
            predictions = clf.predict(X)
            
            # 4. Calculate Error (sum of weights of misclassified points)
            error = np.sum(w[y_encoded != predictions])

            # Safety caps for error
            if error > 0.5:
                error = 0.5 # In a real scenario, we might invert the learner, but here we cap.
            if error == 0:
                error = 1e-10 # Prevent division by zero

            # 5. Calculate Alpha (Amount of Say)
            # Formula: 0.5 * ln((1-error)/error)
            EPS = 1e-10
            alpha = 0.5 * np.log((1.0 - error + EPS) / (error + EPS))
            alpha *= self.learning_rate
            
            # 6. Update Weights
            # Matches: 1 if correct, -1 if wrong
            matches = np.where(y_encoded == predictions, 1, -1)
            
            # w_new = w_old * exp(-alpha * match)
            # If match (1): exp(-alpha) -> weight decreases
            # If mismatch (-1): exp(alpha) -> weight increases
            w *= np.exp(-alpha * matches)
            w /= np.sum(w) # Normalize
            
            self.clfs.append(clf)
            self.alphas.append(alpha)

    def predict(self, X):
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples)
        
        # Get predictions from all stumps (n_estimators, n_samples)
        all_preds = np.array([clf.predict(X) for clf in self.clfs])
        
        # Weighted Voting
        for i in range(n_samples):
            votes = {}
            for j in range(len(self.clfs)):
                pred = all_preds[j, i]
                weight = self.alphas[j]
                
                if pred in votes:
                    votes[pred] += weight
                else:
                    votes[pred] = weight
            
            # Return the label with the highest total weight
            final_predictions[i] = max(votes, key=votes.get)
        
        # Convert predictions back to original label format
        final_predictions = np.array([self.inverse_label_map_[int(pred)] for pred in final_predictions])
        return final_predictions