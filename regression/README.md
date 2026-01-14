# Regression (Linear)

Summary
- Simple and multiple linear regression implementations plus polynomial feature transformer.

Files
- `linear_regression/simple_linear_regression/LR.py` — `SimpleLinearRegression` (GD) with `fit` and `predict`; stores `coef_` and `intercept_`.
- `linear_regression/simple_linear_regression/LR_OLS.py` — `SimpleLinearRegressionOLS` (closed-form OLS).
- `linear_regression/multiple_linear_regression/Multiple_LR.py` — `MultipleLinearRegression` (GD) with `fit` and `predict`.
- `linear_regression/polynomial_regression/Polynomial_Regression.py` — `PolynomialRegression` providing `fit_transform(X)`.
- Tests: `test_LR.py`, `test_Polynomial_and_multiple.py` (see files for run instructions).

Quick run
```powershell
# Example: run polynomial & multiple test from repo root
python -m Own_ml_algorithms.regression.linear_regression.polynomial_regression.test_Polynomial_and_multiple
```

Notes
- Gradient-descent methods accept `learning_rate`, `n_iterations`, `tol`, and `verbose` where applicable.
- Scale features for stable training when using GD.
