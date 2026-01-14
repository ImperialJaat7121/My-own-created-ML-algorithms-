# Classification

Summary
- Collection of from-scratch classifiers: Logistic Regression, Naive Bayes, SVM, and KNN classifier.

Files
- `Logistic_Regression/Logistic_regression.py` — `LogisticRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)` with `fit`, `predict_proba`, `predict`.
- `Naive_Bayes/naive_bayes.py` — `MyGaussianNB` with `fit`, `predict`, `accuracy` and an interactive demo notebook.
- `Logistic_Regression/test_logistic.py` — unit/test script comparing with scikit-learn when available.

Quick run
```powershell
# Run logistic regression test
python -m Own_ml_algorithms.classification.Logistic_Regression.test_logistic
```

Notes
- Logistic regression returns probabilities; default decision threshold is 0.5.
- See root `README.md` for high-level links.
