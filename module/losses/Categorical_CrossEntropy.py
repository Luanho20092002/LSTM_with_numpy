import numpy as np

class Categorical_CrossEntropy:
    
    def loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-8))
    
    def dloss(self, y_pre, y_true):
        return y_pre - y_true