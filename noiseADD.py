import numpy as np

def withnoise(x,r=0.05):
    m = len(x)
    n = len(x[0])
    return x + np.random.normal(np.zeros((m,n)), r)