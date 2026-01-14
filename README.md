10. Boosting (AdaBoost, Gradient Boosting, XGBoost)
   - AdaBoost Classifier
     - Location: `Own_ml_algorithms/boosting/adaboost/adaboost_classifier.py` (class `AdaBoostClassifier(n_estimators=50, learning_rate=1.0)`) — methods: `fit(X, y)`, `predict(X)`. Uses `MyDecisionTreeClassifier` (stumps) as weak learners. Test: `boosting/adaboost/test_adaboost_classifier.py`, demos: `boosting/adaboost/visualization_classifier.ipynb`.
   - AdaBoost Regressor
     - Location: `Own_ml_algorithms/boosting/adaboost/adaboost_regressor.py` (class `AdaBoostRegressor(n_estimators=50, learning_rate=1.0)`) — methods: `fit(X, y)`, `predict(X)`. Uses `MyDecisionTreeRegressor` as weak regressors. Test: `boosting/adaboost/test_adaboost_regressor.py`, demos: `boosting/adaboost/visualization_regressor.ipynb`.
   - Gradient Boosting Classifier
     - Location: `Own_ml_algorithms/boosting/gradient boosting/gradient_boosting_classifier.py` (class `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)`) — methods: `fit(X, y)`, `predict_proba(X)`, `predict(X)`. Uses `MyDecisionTreeRegressor` to fit negative gradients (residuals). Test: `boosting/gradient boosting/test_gb_classifier.py`, demo: `boosting/gradient boosting/visual_gb_classifier.ipynb`.
   - Gradient Boosting Regressor
     - Location: `Own_ml_algorithms/boosting/gradient boosting/gradient_boosting_regressor.py` (class `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)`) — methods: `fit(X, y)`, `predict(X)`. Uses residual-fitting with `MyDecisionTreeRegressor`. Test: `boosting/gradient boosting/test_gb_regressor.py`, demo: `boosting/gradient boosting/visual_gb_regressor.ipynb`.
   - XGBoost
     - Location: `Own_ml_algorithms/boosting/xgboost/xgboost_classifier.py` and `xgboost_regressor.py`; tree core: `xgboost_tree.py` (class `XGBoostTree`).
     - API: `XGBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0)` — `fit(X, y)`, `predict_proba(X)`, `predict(X)`; `XGBoostRegressor(...)` — `fit(X, y)`, `predict(X)`. Tests: `boosting/xgboost/test_xgb_classifier.py`, `boosting/xgboost/test_xgb_regressor.py`; demos: `boosting/xgboost/visual_xgb_classifier.ipynb`, `visual_xgb_regressor.ipynb`.

# Machine Learning Algorithms from Scratch


"Why just import when you can build?"


## About
A personal repository to implement core machine learning algorithms and utilities using only Python and NumPy. The goal is to learn the inner workings of algorithms (not to replace libraries like scikit-learn).



## Implemented Modules


### Supervised Learning
- Regression
 - Regression
  - Simple Linear Regression (OLS) — `regression/linear_regression/simple_linear_regression/LR_OLS.py` (class `SimpleLinearRegressionOLS`) — methods: `fit(x, y)`, `predict(X)`.
  - Simple Linear Regression (Gradient Descent) — `regression/linear_regression/simple_linear_regression/LR.py` (class `SimpleLinearRegression`) — `fit(X, y, learning_rate=0.001, n_iterations=1000, tol=1e-6, clip_grad=1e6, scale=True)` and `predict(X)`; stores `coef_` and `intercept_`.
  - Multiple Linear Regression (Gradient Descent) — `regression/linear_regression/multiple_linear_regression/Multiple_LR.py` (class `MultipleLinearRegression`) — `fit(X, y, learning_rate=0.01, n_iterations=1000, tol=1e-6, verbose=False)` and `predict(X)`; stores `coef_` and `intercept_`.
  - Polynomial feature transformer — `regression/linear_regression/polynomial_regression/Polynomial_Regression.py` (class `PolynomialRegression`) — `fit_transform(X)` returns polynomial feature matrix (use with `MultipleLinearRegression` for polynomial regression).
  - Support Vector Machine (SVM) — An algorithm that predicts a continuous numerical value, answering "how much" (e.g., predicting the exact price of a house) / (from-scratch regressor; see `svm/comparision.ipynb` and `svm/demo.ipynb` for usage/tests)
  - K-Nearest Neighbors (KNN) Regressor — A non‑parametric method that predicts continuous values by averaging the targets of the k closest training samples / (from‑scratch regressor; see knn folder for usage/tests)


- Classification
  - Logistic Regression — statistical algorithm used to predict the probability that a specific instance belongs to a particular category (like "Yes" or "No"). Implementation: `classification/Logistic_Regression/Logistic_regression.py` (class `LogisticRegression`) — methods: `fit(X, y)`, `predict_proba(X)`, `predict(X, threshold=0.5)`. Test: `classification/Logistic_Regression/test_logistic.py`.
  - Support Vector Machine (SVM) — An algorithm that categorizes data into distinct groups, answering "which one" (e.g., predicting if a fruit is an apple or an orange) / (from-scratch classifier; see `svm/comparision.ipynb` and `svm/demo.ipynb` for usage/tests).
  - Naive Bayes — Gaussian Naive Bayes implementation. Implementation: `classification/Naive_Bayes/naive_bayes.py` (class `MyGaussianNB`) — methods: `fit(X, y)`, `predict(X)`, and `accuracy(y_true, y_pred)`. Demo: `classification/Naive_Bayes/test.ipynb`.
  - K-Nearest Neighbors (KNN) — instance‑based classifier that assigns a class by majority vote among the k nearest training points / (from‑scratch classifier; see knn folder for usage/tests)


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
conda create -n ml python=3.13
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

4. test_logistic.py (module-level)
   - Location: `Own_ml_algorithms/classification/Logistic_Regression/test_logistic.py`
   - Implementation: `Own_ml_algorithms/classification/Logistic_Regression/Logistic_regression.py` — class `LogisticRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)` with `fit(X, y)`, `predict_proba(X)`, `predict(X, threshold=0.5)`; learned parameters available as `coef_` and `intercept_`.
   - Purpose: validates the from-scratch `LogisticRegression` implementation and compares results with scikit-learn's `LogisticRegression` when available.
   - Run (from repo root):
     ```powershell
     venv\Scripts\activate
     python -m Own_ml_algorithms.classification.Logistic_Regression.test_logistic
     ```

5. SVM comparison / demo (notebook)
  - Location: `Own_ml_algorithms/svm/comparision.ipynb` and `Own_ml_algorithms/svm/demo.ipynb`
  - Implementation: `Own_ml_algorithms/svm/svm_classifier.py` (`MySVMClassifier`) and `Own_ml_algorithms/svm/svm_regressor.py` (`MySVMRegressor`).
  - API summary:
    - `MySVMClassifier(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)` — methods: `fit(X, y)` (expects labels in {0,1} or converted internally to {-1,1}), `predict(X)` (returns -1 or 1 predictions).
    - `MySVMRegressor(learning_rate=0.001, lambda_param=0.01, n_iterations=1000, epsilon=0.1)` — methods: `fit(X, y)`, `predict(X)` (returns continuous predictions); uses epsilon-insensitive loss for training.
  - Purpose: benchmark the project's linear SVM classifier/regressor against scikit‑learn's `SVC`/`SVR` on standard datasets (Breast Cancer for classification, California Housing for regression); includes visual comparisons (confusion matrices, accuracy, prediction plots) and a demonstration of margins and epsilon-tube for SVR.
  - Run: open the notebook in Jupyter and execute all the cells.

6. Naive Bayes demo/notebook (module-level)
  - Location: `Own_ml_algorithms/classification/Naive_Bayes/test.ipynb`
  - Purpose: interactive notebook demonstrating the project's Gaussian Naive Bayes implementation (`naive_bayes.py`), decision-boundary visualizations on synthetic data (Moons), and comparison with scikit‑learn's `GaussianNB`.
  - Run: open the notebook in Jupyter / VS Code

7. KNN classifier test (module-level)
   - Location: `Own_ml_algorithms/knn/test_classifier.py`
   - Implementation: `Own_ml_algorithms/knn/knn_classifier.py` — provides `MyKNNClassifier(k=3)` with `fit(X,y)`, `predict(X)`, and `score(X,y)` (returns accuracy).
   - Purpose: evaluates `MyKNNClassifier` on the Iris dataset and compares with scikit‑learn's `KNeighborsClassifier`.
   - Run:
     ```powershell
     # From repo root (module run)
     venv\Scripts\activate
     python -m Own_ml_algorithms.knn.test_classifier

     # Or run directly from folder
     cd Own_ml_algorithms\knn
     python test_classifier.py
     ```

8. KNN regressor test (module-level)
   - Location: `Own_ml_algorithms/knn/test_regressor.py`
   - Implementation: `Own_ml_algorithms/knn/knn_regressor.py` — provides `MyKNNRegressor(k=3)` with `fit(X,y)`, `predict(X)`, and `score(X,y)` (returns R²).
   - Purpose: evaluates `MyKNNRegressor` on the Diabetes dataset and compares with scikit‑learn's `KNeighborsRegressor`.
   - Run:
     ```powershell
     # From repo root (module run)
     venv\Scripts\activate
     python -m Own_ml_algorithms.knn.test_regressor

     # Or run directly from folder
     cd Own_ml_algorithms\knn
     python test_regressor.py
     ```

9. Trees (Decision Trees & Random Forests)
   - Decision Tree Classifier
     - Location: `Own_ml_algorithms/trees/decisiontree/decision_tree_classifier.py`
     - Implementation: class `MyDecisionTreeClassifier(min_samples_split=2, max_depth=100, n_features=None)` — methods: `fit(X, y)`, `predict(X)`. Uses information gain (entropy) for splits and returns class labels.
   - Decision Tree Regressor
     - Location: `Own_ml_algorithms/trees/decisiontree/decision_tree_regressor.py`
     - Implementation: class `MyDecisionTreeRegressor(min_samples_split=2, max_depth=100, n_features=None)` — methods: `fit(X, y)`, `predict(X)`, `score(X_test, y_test)` (returns R²). Uses variance reduction for splits.
   - Random Forest Classifier
     - Location: `Own_ml_algorithms/trees/randomforest/random_forest_classifier.py`
     - Implementation: class `MyRandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=100, max_features=None, bootstrap=True, random_state=None)` — methods: `fit(X, y)`, `predict(X)`, `score(X_test, y_test)`. Uses bagging + `MyDecisionTreeClassifier` as base learner.
   - Random Forest Regressor
     - Location: `Own_ml_algorithms/trees/randomforest/random_forest_regressor.py`
     - Implementation: class `MyRandomForestRegressor(n_estimators=100, min_samples_split=2, max_depth=100, max_features=None, bootstrap=True, random_state=None)` — methods: `fit(X, y)`, `predict(X)`, `score(X_test, y_test)` (R²). Uses bagging + `MyDecisionTreeRegressor` as base learner.
   - Demo / Tests:
     - Random Forest demo: `trees/randomforest/demo_random_forest.ipynb` (classification and regression examples).

Notes:
- Ensure `scikit-learn` and `pandas` are installed to run the comparison test.
- `test_LR.py` expects `weight_height.csv` to be in the same directory as the test file (it is included in the repo).

## Notebooks (comparison and visualization)
- SVM comparison: [svm/comparision.ipynb](svm/comparision.ipynb)
- SVM demo/visualization: [svm/demo.ipynb](svm/demo.ipynb)
  - Open these notebooks to see side-by-side results and visualizations (e.g., decision boundaries) for the custom SVM vs scikit‑learn.
- Naive Bayes demo/visualization: [classification/Naive_Bayes/test.ipynb](classification/Naive_Bayes/test.ipynb)
  - Visualizes Gaussian Naive Bayes behavior on the Moons dataset, decision boundaries, and comparison with scikit‑learn.
- KNN classifier demo: [knn/demo_knn_classifier.ipynb](knn/demo_knn_classifier.ipynb)
  - Visualizes decision boundaries on Iris using `MyKNNClassifier` with different k values and neighbor connections.
- KNN regressor demo: [knn/demo_knn_regressor.ipynb](knn/demo_knn_regressor.ipynb)
  - Fits `MyKNNRegressor` to a noisy sine wave and compares different k values for under/overfitting behavior.

## Module notes / API highlights
- MultipleLinearRegression.fit accepts common gradient descent parameters:
  - learning_rate (float), n_iterations (int), tol (float), verbose (bool).
- Polynomial_Regression.PolynomialRegression provides fit_transform(X) to generate polynomial features of specified degree; combine with MultipleLinearRegression for polynomial regression.
- All models store learned parameters in the scikit‑learn style attributes (coef_, intercept_).
- Logistic Regression (from scratch): provides `fit` (supports `learning_rate`, `n_iterations`, `tol`, `verbose`), `predict_proba`, and `predict` (default threshold 0.5); learned parameters are available as `coef_` and `intercept_`. Implementation uses vectorized gradient descent for binary classification — recommended to scale features before training. See `classification/Logistic_Regression/test_logistic.py` for usage and scikit‑learn comparisons.
- SVM (from scratch) mirrors scikit‑learn’s interface where possible (fit, predict); includes both classifier and regressor variants. See svm/test_svm.py for usage and comparisons.
- Naive Bayes implementations included (Gaussian). See `naive_bayes/` for API examples and comparison tests with scikit‑learn.
- KNN (Classifier & Regressor): lazy learners — `fit(X, y)` stores training data; `predict(X)` finds k nearest neighbors via Euclidean distance and aggregates by majority vote (classifier) or mean (regressor); `score()` returns accuracy (classifier) or R² (regressor). Primary hyperparameter: `k` (int). Consider feature scaling for meaningful distance measures.

## Notes & Recommendations

- Use feature scaling when training gradient-descent models to avoid overflow/exploding gradients.
- For multivariate extensions, ensure input X has shape (n_samples, n_features).
- This repo is educational — production use should prefer well-tested libraries.

## Project Roadmap
- k‑Means and other clustering
- Broader unit tests, benchmarks, and documentation

## Contributing
This is a personal learning project, but if you see a bug or have a suggestion for improvement, feel free to open an issue or submit a pull request!