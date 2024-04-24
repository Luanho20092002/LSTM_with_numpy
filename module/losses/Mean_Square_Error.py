import numpy as np

class Mean_Square_Error:

    def loss(self, y_pre, y_true):
        return np.mean((y_true - y_pre) ** 2)
    
    def dloss(self, y_pre, y_true):
        return  2 * (y_pre - y_true) / len(y_true)