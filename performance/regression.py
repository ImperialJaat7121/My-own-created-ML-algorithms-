import numpy as np
import sys

def calculate_mae(y_true, y_pred):
    try:
        error = np.mean(np.abs(y_true - y_pred))
        return error
    except Exception as e:
        print(f"[ERROR in MAE]: {e}", file=sys.stderr)
        return None

def calculate_mse(y_true, y_pred):
    try:
        error = np.mean((y_true - y_pred)**2)
        return error
    except Exception as e:
        print(f"[ERROR in MSE]: {e}", file=sys.stderr)
        return None
    
def calculate_rmse(y_true, y_pred):
    try:
        mse = calculate_mse(y_true, y_pred)
        if mse is not None:
            return np.sqrt(mse)
        else:
            return None
    except Exception as e:
        print(f"[ERROR in RMSE]: {e}", file=sys.stderr)
        return None
    
def calculate_r_squared(y_true, y_pred):
    try:
        sum_squared_residuals = np.sum((y_true - y_pred)**2)
        
        y_mean = np.mean(y_true)
        total_sum_of_squares = np.sum((y_true - y_mean)**2)
        
        if total_sum_of_squares == 0:
            print("Cannot calculate R-Squared. All true y values are identical.", file=sys.stderr)
            return None

        r2 = 1 - (sum_squared_residuals / total_sum_of_squares)
        return r2
    
    except Exception as e:
        print(f"[ERROR in R²]: {e}", file=sys.stderr)
        return None