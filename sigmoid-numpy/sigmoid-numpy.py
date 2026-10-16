import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    arr = np.asarray( x , dtype = float )
    
    t = 1.0 / ( 1.0 + np.exp(-arr) ) 

    return float(t) if arr.ndim == 0 else t 
    pass