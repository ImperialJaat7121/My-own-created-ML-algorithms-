import numpy as np

class MySVMClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param 
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train SVM Classifier using Hinge Loss and Gradient Descent.
        y must be converted to {-1, 1} for the math to work.
        """
        # Convert labels: 0 becomes -1, 1 stays 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Check geometric margin constraint: y_i * (w.x + b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Point is correctly classified and safely outside margin
                    # Gradient comes only from regularization term (2 * lambda * w)
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Point is inside margin or misclassified (Hinge Loss active)
                    # Gradient includes data term to push point back
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx] # Bias update (gradient is -y_i, so we subtract -y_i)

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        # Return -1 or 1 based on sign
        return np.sign(approx)