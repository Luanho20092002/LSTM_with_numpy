import numpy as np

        
class Adam:

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def update(self, layer, key_params, *arg):
        D = arg # (dw, .., db)
        if layer.has_optimizer_params:
            layer.has_optimizer_params = False
            layer.optimizer_params = {}
            layer.iter = 0
            for k in key_params:
                layer.optimizer_params[k] = {"m": np.zeros_like(layer.params[k]), 
                                             "v": np.zeros_like(layer.params[k])} 
        layer.iter += 1
        """ if layer.iter % 100 == 0:
            self.lr += 0.01 """
        for i, k in enumerate(key_params):
            layer.optimizer_params[k]["m"] = self.beta1 * layer.optimizer_params[k]["m"] + (1 - self.beta1) * (D[i]) 
            layer.optimizer_params[k]["v"] = self.beta2 * layer.optimizer_params[k]["v"] + (1 - self.beta2) * (D[i]**2)
            w_m_hat = layer.optimizer_params[k]["m"] / (1 - self.beta1**layer.iter)
            w_v_hat = layer.optimizer_params[k]["v"] / (1 - self.beta2**layer.iter)
            layer.params[k] -= self.lr / (np.sqrt(w_v_hat) + self.eps) * w_m_hat     

