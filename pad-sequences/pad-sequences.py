import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) == 0 :
        return np.empty( (0,0) , dtype = int )
        
    if max_len == None :
        L = max( len(s) for s in seqs )
    else :
        L = max_len

    res = np.full( ( len(seqs) , L ) , pad_value , dtype = int )

    for i, s in enumerate( seqs ):
        arr = s[:L] 
        res[i , :len(arr) ] = arr 

    return res 
    
    pass