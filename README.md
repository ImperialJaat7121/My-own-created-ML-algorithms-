# Machine Learning Algorithms from Scratch

"Why just import when you can build?"

## About
A personal repository to implement core machine learning algorithms and utilities using only Python and NumPy. The goal is to learn the inner workings of algorithms (not to replace libraries like scikit-learn).

## Implemented Modules

### Supervised Learning
- Regression
  - Simple Linear Regression (OLS) — LR_OLS.py
  - Simple Linear Regression (Gradient Descent) — LR.py (from-scratch API compatible with scikit-learn)
  - Multiple Linear Regression (Gradient Descent) — Multiple_LR.py
  - Polynomial feature transformer — Polynomial_Regression.py (used with MultipleLinearRegression for polynomial regression)


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

## Test scripts (sklearn comparison)
The test scripts are provided to validate implementations and compare against scikit‑learn.

1. test.py (repo root)
   - Basic end‑to‑end check for SimpleLinearRegression using synthetic data.
   - Run from repo root:
     - venv: `venv\Scripts\activate` then `python test.py`
     - conda: `conda activate ml` then `python test.py`

2. test_LR.py (module-level)
   - Location: `Own_ml_algorithms/regression/linear_regression/simple_linear_regression/test_LR.py`
   - Compares LR.py (gradient descent), LR_OLS.py (closed-form OLS) and scikit‑learn LinearRegression on the included `weight_height.csv`.
   - Run:
     ```powershell
     cd Own_ml_algorithms\regression\linear_regression\simple_linear_regression
     python test_LR.py
     ```

3. test_Polynomial_and_multiple.py (module-level)
   - Location: `Own_ml_algorithms/regression/linear_regression/polynomial_regression/test_Polynomial_and_multiple.py`
   - Two tests:
     - TEST A: Multiple Linear Regression (2 features) — trains MultipleLinearRegression and compares with scikit‑learn.
     - TEST B: Polynomial Regression — uses Polynomial_Regression to generate polynomial features, then fits MultipleLinearRegression and compares with scikit‑learn LinearRegression.
   - Run (from repo root for package imports to resolve):
     ```powershell
     venv\Scripts\activate
     python -m Own_ml_algorithms.regression.linear_regression.polynomial_regression.test_Polynomial_and_multiple
     ```
   - Or run directly from the module folder (the test script adds repo root to sys.path automatically), e.g.:
     ```powershell
     cd Own_ml_algorithms\regression\linear_regression\polynomial_regression
     python test_Polynomial_and_multiple.py
     ```

Notes:
- Ensure `scikit-learn` and `pandas` are installed to run the comparison test.
- `test_LR.py` expects `weight_height.csv` to be in the same directory as the test file (it is included in the repo).

## Module notes / API highlights
- MultipleLinearRegression.fit accepts common gradient descent parameters:
  - learning_rate (float), n_iterations (int), tol (float), verbose (bool).
- Polynomial_Regression.PolynomialRegression provides fit_transform(X) to generate polynomial features of specified degree; combine with MultipleLinearRegression for polynomial regression.
- All models store learned parameters in the scikit‑learn style attributes (coef_, intercept_).

## Notes & Recommendations

- Use feature scaling when training gradient-descent models to avoid overflow/exploding gradients.
- For multivariate extensions, ensure input X has shape (n_samples, n_features).
- This repo is educational — production use should prefer well-tested libraries.

## Project Roadmap
- Expand Multiple Linear Regression and vectorized GD
- Logistic Regression (binary classification)
- k‑NN, Naive Bayes, Decision Trees, K‑Means
- Expand unit tests and documentation
- ...and many more!

## Contributing
This is a personal learning project, but if you see a bug or have a suggestion for improvement, feel free to open an issue or submit a pull request!