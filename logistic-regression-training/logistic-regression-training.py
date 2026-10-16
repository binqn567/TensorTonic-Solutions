import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n = X.shape[0] 
    m = X.shape[1] 

    W = np.zeros(m)
    b = 0.0 

    for i in range( steps ) : 
        p = _sigmoid( X @ W  + b )
        grad_w = X.T @ ( p - y ) / n 
        grad_b = np.mean( p - y )
        W -= lr * grad_w 
        b -= lr * grad_b 

    return W , float(b)
    pass