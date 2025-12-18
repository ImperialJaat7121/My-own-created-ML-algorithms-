import numpy as np

class MyGaussianNB:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        Train the model by calculating statistics (Mean, Variance, Prior) for each class.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            # Filter samples belonging to class c
            X_c = X[y == c]
            
            # 1. Calculate Mean per feature
            self.mean[c] = np.mean(X_c, axis=0)
            
            # 2. Calculate Variance per feature
            self.var[c] = np.var(X_c, axis=0)
            
            # 3. Calculate Prior (Frequency of class c)
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def _gaussian_pdf(self, class_idx, x):
        """
        Calculate the Gaussian Probability Density Function (PDF).
        P(x|c) = (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_sample(self, x):
        """
        Predict the class label for a single sample.
        Returns the class with the highest posterior probability.
        """
        posteriors = []

        for c in self.classes:
            # P(y)
            prior = np.log(self.priors[c])
            
            # P(X|y)
            # We use Log Probability to prevent underflow (multiplying small numbers -> 0)
            # Sum of logs is equivalent to Log of product
            posterior = np.sum(np.log(self._gaussian_pdf(c, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """Predict class labels for samples in X."""
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
