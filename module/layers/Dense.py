import numpy as np

class Dense:

    def __init__(self, units, active="relu") -> None:
        self.units = units
        self.active = active
        self.has_params = True
        self.has_optimizer_params = True

    def forward(self, X):
        n, d = X.shape
        self.X = X
        if self.has_params:
            self.has_params = False
            self.params = {"w": 0.01 * np.random.randn(d, self.units),
                           "b": 0.01 * np.random.randn(1, self.units)}
        
        Z = X @ self.params["w"] + self.params["b"]
        if self.active == "relu":
            self.mask = (Z > 0)
            return Z * self.mask
        elif self.active == "softmax":
            return self.softmax(Z)
        return Z
    
    def backward(self, dZ, optimizer):
        if self.active == "relu":
            dZ = dZ * self.mask
        dX = dZ @ self.params["w"].T
        dw = self.X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        optimizer.update(self, self.params.keys(), dw, db)
        return dX

    def softmax(self, Z): 
        eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return eZ/eZ.sum(axis=1).reshape(-1, 1)

    def get_weight(self):
        return self.w
    
    def get_bias(self):
        return self.b
    
    def set_weight(self, w):
        self.w = w

    def set_bias(self, b):
        self.b = b