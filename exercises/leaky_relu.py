import numpy as np
def leaky_relu(x, alpha=0.01):
    activated_x = np.copy(x)
    negative_indices = x <= 0
    activated_x[negative_indices] *= alpha
    return activated_x
