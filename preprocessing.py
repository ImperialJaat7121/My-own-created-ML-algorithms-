import numpy as np
import sys

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    
    try:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples (rows).")
    except Exception as e:
        print(f"[ERROR in train_test_split]: {e}", file=sys.stderr)
        return None
        
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        
        np.random.shuffle(indices)

    n_test = int(n_samples * test_size)
    if n_test == 0:
        print("[WARNING in train_test_split]: Test size is 0. Adjusting to 1.", file=sys.stderr)
        n_test = 1
        
    n_train = n_samples - n_test
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Data split: {n_train} training samples, {n_test} test samples.")
    return X_train, X_test, y_train, y_test