import numpy as np
from module.optimizer.Adam import Adam
from module.optimizer.RMSProp import RMSProp
from module.optimizer.SGD import SGD
from module.losses.Categorical_CrossEntropy import Categorical_CrossEntropy
from module.losses.Mean_Square_Error import Mean_Square_Error
from module.metrics.RMSE import RMSE

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
                dZ = self.loss.dloss(A, y_batch) / batch_size
                self.backward(dZ)

                if (b+batch_size) >= mark[curr_mark]:
                    curr_mark += 1
                    print("=", end="")
            print("]", end="")
            y_pred, score = self.evalute(X, y)
            print(f"  loss: {self.loss.loss(y_pred, y):.4f}, metric: {score:.4f}") #, self.optimizer.lr)

            
    def add(self, l):
        self.layers = self.layers + (l,)
        
    def predict(self, X):
        for md in self.layers:
            X = md.forward(X)
        return X
    
    def evalute(self, Xtest, ytest):
        y_pred = self.predict(Xtest)
        # score = np.mean(np.argmax(y_pred, axis=1) == np.argmax(ytest, axis=1))
        score = self.metric.metric(y_pred, ytest)
        return self.loss.loss(y_pred, ytest), score
    
    def compile(self, loss="categorical_crossEntropy", optimizer="adam", metric="accuracy"): 
        if optimizer == "sgd":
            self.optimizer = SGD()
        elif optimizer == "rmsprop":
            self.optimizer = RMSProp()
        else: self.optimizer = Adam()

        if loss == "mse":
            self.loss = Mean_Square_Error()
        else: self.loss = Categorical_CrossEntropy()

        if metric == "rmse":
            self.metric = RMSE()
        else: pass
        


        
