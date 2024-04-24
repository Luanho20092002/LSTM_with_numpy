import numpy as np

class SGD:

    def __init__(self, lr=0.001, momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        
    def update(self, this, dw, db):
        if this.is_init_mv:
            this.is_backward = False
            this.v_w = np.zeros_like(this.w)
            this.v_b = np.zeros_like(this.b)
        this.v_w = self.momentum * this.v_w + self.lr * dw
        this.v_b = self.momentum * this.v_b + self.lr * db
        this.w -= this.v_w
        this.b -= this.v_b