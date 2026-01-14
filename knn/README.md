# KNN (K-Nearest Neighbors)

Summary
- Instance-based learners: classifier and regressor implemented from scratch using Euclidean distance.

Files
- `knn_classifier.py` — `MyKNNClassifier(k=3)` with `fit(X, y)`, `predict(X)`, `score(X, y)` (accuracy).
- `knn_regressor.py` — `MyKNNRegressor(k=3)` with `fit(X, y)`, `predict(X)`, `score(X, y)` (R²).
- `test_classifier.py`, `test_regressor.py` — test scripts comparing with scikit-learn.
- `demo_knn_classifier.ipynb`, `demo_knn_regressor.ipynb` — visualization notebooks.

Quick run
```powershell
# Run classifier test from repo root
python -m Own_ml_algorithms.knn.test_classifier

# Run regressor test from repo root
python -m Own_ml_algorithms.knn.test_regressor
```

Notes
- Feature scaling is recommended for distance-based methods.
- See root `README.md` for cross-references and usage examples.
