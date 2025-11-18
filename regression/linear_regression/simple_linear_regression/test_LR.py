import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from LR import SimpleLinearRegression
from LR_OLS import SimpleLinearRegressionOLS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

df=pd.read_csv("weight_height.csv")

X = df[['Weight']]
y = df['Height']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

## Using Own SimpleLinearRegression
model_own = SimpleLinearRegression(fit_intercept=True)
model_own.fit(X_train.flatten(), y_train.values, learning_rate=0.01, n_iterations=1000)
y_pred_own = model_own.predict(X_test.flatten())
mse_own = mean_squared_error(y_test, y_pred_own)
r2_own = r2_score(y_test, y_pred_own)
print("\n[OWN SimpleLinearRegression] --- Performance Metrics ---")
print(f"Mean Squared Error (MSE): {mse_own}")
print(f"R-squared (R2 ): {r2_own}")

## Using Own SimpleLinearRegressionOLS
model_ols = SimpleLinearRegressionOLS()
model_ols.fit(X_train.flatten(), y_train.values)
y_pred_ols = model_ols.predict(X_test.flatten())
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)
print("\n[OWN SimpleLinearRegressionOLS] --- Performance Metrics ---")
print(f"Mean Squared Error (MSE): {mse_ols}")
print(f"R-squared (R2 ): {r2_ols}")

## Using Sklearn LinearRegression
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
print("\n[Sklearn LinearRegression] --- Performance Metrics ---")
print(f"Mean Squared Error (MSE): {mse_sklearn}")
print(f"R-squared (R2 ): {r2_sklearn}")