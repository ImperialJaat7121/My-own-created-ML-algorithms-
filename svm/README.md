# SVM (Support Vector Machines)

Summary
- Linear SVM classifier and regressor implemented from scratch (hinge loss for classifier, epsilon-insensitive loss for regressor).

Files
- `svm_classifier.py` — `MySVMClassifier(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)` with `fit(X, y)`, `predict(X)`.
- `svm_regressor.py` — `MySVMRegressor(..., epsilon=0.1)` with `fit(X, y)`, `predict(X)`.
- `comparision.ipynb`, `demo.ipynb` — notebooks demonstrating behavior and benchmarking vs scikit-learn.

Quick run
- Open the notebooks in Jupyter / VS Code and execute all cells.

Notes
- Classifier outputs -1/1; labels may be mapped internally.
- See root `README.md` for API summaries and examples.
