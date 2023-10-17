import numpy as np

class RMSProp:

    def __init__(self, lr=0.001, gamma=0.9, eps=1e-8) -> None:
        self.lr = lr
        self.gamma = gamma
        self.eps = eps

    def update(self, layer, key_params, *arg):
        D = arg # (dw,..., db)
        if layer.has_optimizer_params:
            layer.has_optimizer_params = False
            layer.optimizer_params = {}
            for k in key_params:
                layer.optimizer_params[k] = np.zeros_like(layer.params[k])
        
        for i, k in enumerate(key_params):
            layer.optimizer_params[k] = self.gamma * layer.optimizer_params[k] + (1 - self.gamma) * (np.array(D[i])**2)
            layer.params[k] -= self.lr * D[i] / (np.sqrt(layer.optimizer_params[k]) + self.eps)