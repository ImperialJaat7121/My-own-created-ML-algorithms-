# Trees (Decision Trees & Random Forests)

Summary
- Decision tree classifiers/regressors and Random Forest implementations.

Files
- `decisiontree/decision_tree_classifier.py` — `MyDecisionTreeClassifier(min_samples_split=2, max_depth=100, n_features=None)` with `fit` and `predict`.
- `decisiontree/decision_tree_regressor.py` — `MyDecisionTreeRegressor(...)` with `fit`, `predict`, `score`.
- `randomforest/random_forest_classifier.py` — `MyRandomForestClassifier(...)` with `fit`, `predict`, `score`.
- `randomforest/random_forest_regressor.py` — `MyRandomForestRegressor(...)` with `fit`, `predict`, `score`.
- `randomforest/demo_random_forest.ipynb` — demo notebook.

Quick run
- Open `randomforest/demo_random_forest.ipynb` to see examples and comparisons with scikit-learn.

Notes
- Random forest uses bagging over `MyDecisionTree*` base learners; `max_features` and `bootstrap` are supported.
