import numpy as np

def dyadup(x, n=1, phase=1):
    """
    Python implementation of MATLAB's dyadup function for dyadic upsampling.
    
    Parameters:
    -----------
    x : array_like
        Input signal/array to be upsampled
    n : int, optional
        Number of times to perform dyadic upsampling (default=1)
    phase : int, optional
        Phase of the upsampled signal (0 or 1, default=0)
        0 = even phase (zeros inserted after samples)
        1 = odd phase (zeros inserted before samples)
    
    Returns:
    --------
    y : ndarray
        Upsampled signal
    """
    x = np.asarray(x)
    
    # Handle multiple upsampling iterations
    for _ in range(n):
        new_shape = list(x.shape)
        new_shape[0] *= 2
        y = np.zeros(new_shape, dtype=x.dtype)
        # Insert values based on phase
        if phase == 0:
            y[::2] = x  # Even phase: insert zeros after samples
        else:
            y[1::2] = x  # Odd phase: insert zeros before samples
        x = y  
    return y