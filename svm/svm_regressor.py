import numpy as np

class MySVMRegressor:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, epsilon=0.1):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.epsilon = epsilon # The width of the "tube" where error is ignored
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            # Batch Gradient Descent for Regression
            
            # 1. Forward pass
            y_pred = np.dot(X, self.w) + self.b
            
            # 2. Calculate raw errors
            errors = y - y_pred
            
            dw = np.zeros(n_features)
            db = 0
            
            # 3. Calculate Gradients based on Epsilon-Insensitive Loss
            for i in range(n_samples):
                if errors[i] > self.epsilon:
                    # Point is above the tube (y > y_pred + eps)
                    # Pull model UP towards point
                    dw += -X[i]
                    db += -1
                elif errors[i] < -self.epsilon:
                    # Point is below the tube (y < y_pred - eps)
                    # Pull model DOWN towards point
                    dw += X[i]
                    db += 1
                # If |error| <= epsilon, gradient is 0 (ignore point)
            
            # Add Regularization gradient
            dw += 2 * self.lambda_param * self.w
            
            # 4. Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# TESTING THE SVM REGRESSOR

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    print("--- SVM Regression Test ---")
    X_reg, y_reg = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    
    # Scaling is crucial for SVM!
    scaler = StandardScaler()
    X_reg = scaler.fit_transform(X_reg)
    
    reg = MySVMRegressor(learning_rate=0.01, n_iterations=1000, epsilon=5)
    reg.fit(X_reg, y_reg)
    
    mse = mean_squared_error(y_reg, reg.predict(X_reg))
    print(f"MSE: {mse:.4f}")