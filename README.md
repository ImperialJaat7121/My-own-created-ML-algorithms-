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

Windows (venv)
```powershell
git clone <repo-url>
cd <Your path>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Windows (conda)
```bash
git clone <repo-url>
cd <Your path>
conda create -n ml python=3.10
conda activate ml
pip install -r requirements.txt
```

## Quick Usage Example

```python
import numpy as np
from Own_ml_algorithms.regression.linear_regression.simple_linear_regression.LR import SimpleLinearRegression
from Own_ml_algorithms.preprocessing import train_test_split
from Own_ml_algorithms.performance.regression import (
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

## Test scripts (sklearn comparison)
The test scripts are provided to validate implementations and compare against scikit‑learn.

### test_LR.py (module-level comparison)
- Location: `Own_ml_algorithms\regression\linear_regression\simple_linear_regression\test_LR.py`
- Purpose: compares three implementations on the included dataset `weight_height.csv`:
  - Own gradient‑descent SimpleLinearRegression (LR.py)
  - Own OLS implementation (LR_OLS.py)
  - scikit‑learn LinearRegression
- Behavior: loads `weight_height.csv`, scales features, trains all three models, prints MSE and R² for side‑by‑side comparison.
- Run options:
  - Option A — run from that folder (recommended for direct imports):
    ```powershell
    cd Own_ml_algorithms\regression\linear_regression\simple_linear_regression
    python test_LR.py
    ```
  - Option B — run with pytest (if pytest installed):
    ```bash
    pytest Own_ml_algorithms/regression/linear_regression/simple_linear_regression/test_LR.py
    ```

Notes:
- Ensure `scikit-learn` and `pandas` are installed to run the comparison test.
- `test_LR.py` expects `weight_height.csv` to be in the same directory as the test file (it is included in the repo).

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