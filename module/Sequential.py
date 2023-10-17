import numpy as np
from module.optimizer.Adam import Adam
from module.optimizer.RMSProp import RMSProp
from module.optimizer.SGD import SGD

class Sequential:

    def __init__(self, *args) -> None:
        self.layers = args

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X) 
        return X
    
    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ, optimizer=self.optimizer)
            
    def fit(self, X, y, batch_size=10, epochs=10, validation=None) -> None:
        N = len(X)
        mark = np.linspace(N/10, N, 10)
        for e in range(epochs):
            curr_mark = 0
            if e < 10:
                print(f"Epoch {e}  [", end="")
            else:
                print(f"Epoch {e} [", end="")
            for b in range(0, N, batch_size): # batch
                X_batch = X[b:b+batch_size]
                y_batch = y[b:b+batch_size]
                #Forward
                A = self.forward(X_batch)
                #Backpropagation & update weight
                dZ = (A - y_batch) / batch_size
                self.backward(dZ)
                if (b+batch_size) >= mark[curr_mark]:
                    curr_mark += 1
                    print("=", end="")
            print("]", end="")
            y_pred, score = self.evalute(X, y)
            print(f"  loss: {self.loss(y_pred, y):.4f}, accuracy {score*100:.2f}%") #, self.optimizer.lr)
            if (score > 0.95):
                break
            
    def add(self, l):
        self.layers = self.layers + (l,)
        
    def predict(self, X):
        for md in self.layers:
            X = md.forward(X)
        return X
    
    def evalute(self, Xtest, ytest):
        y_pred = self.predict(Xtest)
        score = np.mean(np.argmax(y_pred, axis=1) == np.argmax(ytest, axis=1))
        return y_pred, score
    
    def compile(self, loss="categorical_crossEntropy", optimizer="adam", metric="accuracy"):
        if isinstance(optimizer, str):
            if optimizer == "adam":
                self.optimizer = Adam()
            if optimizer == "sgd":
                self.optimizer = SGD()
            if optimizer == "rmsprop":
                self.optimizer = RMSProp()
        else: self.optimizer = optimizer
        
    def loss(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred + 1e-8))
