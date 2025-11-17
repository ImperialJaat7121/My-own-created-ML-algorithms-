# Machine Learning Algorithms from Scratch

"Why just import when you can build?"

## About
A personal repository to implement core machine learning algorithms and utilities using only Python and NumPy. The goal is to learn the inner workings of algorithms (not to replace libraries like scikit-learn).

## Implemented Modules

### Supervised Learning
- Regression
  - Simple Linear Regression (OLS) — LR_OLS.py
  - Simple Linear Regression (Gradient Descent) — LR.py (from-scratch API compatible with scikit-learn)

### Utilities
- Preprocessing
  - train_test_split (custom implementation)
- Performance / Metrics
  - calculate_mae, calculate_mse, calculate_rmse, calculate_r_squared

## Installation
Clone the repository and use your preferred environment (recommended: conda/venv).

```bash
git clone <repo-url>
cd <Your directory>
# create or activate environment, then:
pip install -r requirements.txt
```

## Quick Usage Example

```python
import numpy as np
from My-own-created-ML-algorithms-.regression.linear_regression.simple_linear_regression.LR import SimpleLinearRegression
from My-own-created-ML-algorithms-.preprocessing import train_test_split
from My-own-created-ML-algorithms-.performance.regression import (
    calculate_mae, calculate_mse, calculate_rmse, calculate_r_squared
)

# create synthetic data
X = np.linspace(1, 20, 100)
y = 3 * X + 5 + np.random.randn(100) * 2.0

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = SimpleLinearRegression(fit_intercept=True)
model.fit(X_train, y_train, learning_rate=0.01, n_iterations=1000)

# predict & evaluate
preds = model.predict(X_test)
print("Coef:", model.coef_, "Intercept:", model.intercept_)
print("RMSE:", calculate_rmse(y_test, preds), "R2:", calculate_r_squared(y_test, preds))
```

## Notes & Recommendations
- Use feature scaling when training gradient-descent models to avoid overflow/exploding gradients.
- For multivariate extensions, ensure input X has shape (n_samples, n_features).
- This repo is educational — production use should prefer well-tested libraries.

## Project Roadmap
My goal is to continue adding all Machine Learning algorithms. The next major steps are:
- Multiple Linear Regression: Vectorize the Gradient Descent model to accept (n_samples, n_features) input.
- Logistic Regression: Adapt the Gradient Descent engine for classification using the Sigmoid function and Log Loss (Binary Cross-Entropy).
- K-Nearest Neighbors (k-NN): Implement a non-parametric "lazy learning" algorithm.
- Decision Tree: Build a rule-based model using recursive partitioning and Gini/Entropy impurity.
- ...and many more!

## Contributing
This is a personal learning project, but if you see a bug or have a suggestion for improvement, feel free to open an issue or submit a pull request!