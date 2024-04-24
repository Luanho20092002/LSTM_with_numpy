import numpy as np

class RMSE:
    
    def metric(self, y_pre, y_true):
        return np.sqrt(np.mean((y_true - y_pre) ** 2))