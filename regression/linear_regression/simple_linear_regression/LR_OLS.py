import numpy as np
import sys

"""
METHOD: Ordinary Least Squares
"""

class SimpleLinearRegressionOLS:

    def __init__(self):
        self.m_slope = None
        self.b_intercept = None

    def fit(self, x, y):

        print("______ Simple Linear Regression OLS ______")

        try:
            x_mean = np.mean(x)
            y_mean = np.mean(y)
        except Exception as e:
            print(f"[ERROR] Could not calculate means. Data may be invalid. Details: {e}", file=sys.stderr)
            return

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            print("[WARNING] Cannot fit a line. All X values are identical.", file=sys.stderr)
            self.m_slope = 0
            self.b_intercept = y_mean
            print(f"[LOG] Model fit with m=0 and b={y_mean:.3f} due to constant X.")
            return
        
        self.m_slope = numerator / denominator

        self.b_intercept = y_mean - (self.m_slope * x_mean)

        print(f"Final model: y = {self.m_slope:.3f} * x + {self.b_intercept:.3f}")
        print(" EXECUTION COMPLETED SUCCESSFULLY! \n")

    def predict(self, X):
        if self.m_slope is None or self.b_intercept is None:
            print("[WARNING] predict() called before .fit(). Model has not been trained.", file=sys.stderr)
            return np.full(X.shape, np.nan)
        return self.m_slope * X + self.b_intercept

