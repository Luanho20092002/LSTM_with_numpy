import numpy as np


class Relu:
    def forward(self, X) -> None:
        return np.maximum(0, X)
    
    def backward(self, X) -> None:
        return np.where(X > 0, 1, 0)
    
class Tanh:
    def forward(self, X) -> None:
        return np.tanh(X)
    
    def backward(self, X) -> None:
        return (1 - np.tanh(X)**2)

class LSTM:

    def __init__(self, units, active="tanh") -> None:
        self.units = units
        if active == "relu":
            self.active = Relu()
        elif active == "tanh":
            self.active = Tanh()
        self.has_params = True
        self.has_optimizer_params = True
    
    def forward(self, X):
        n, timestep, d = X.shape
        self.X = X

        if self.has_params:
            self.has_params = False
            self.params = {"Wf":  0.01 * np.random.randn(d, self.units),
                           "Wi":  0.01 * np.random.randn(d, self.units),
                           "Wu":  0.01 * np.random.randn(d, self.units),
                           "Wo":  0.01 * np.random.randn(d, self.units),
                           "Whf": 0.01 * np.random.randn(self.units, self.units),
                           "Whi": 0.01 * np.random.randn(self.units, self.units),
                           "Whu": 0.01 * np.random.randn(self.units, self.units),
                           "Who": 0.01 * np.random.randn(self.units, self.units),
                           "bf":  0.01 * np.random.randn(1, self.units),
                           "bi":  0.01 * np.random.randn(1, self.units),
                           "bu":  0.01 * np.random.randn(1, self.units),
                           "bo":  0.01 * np.random.randn(1, self.units)}
        h = np.zeros((n, self.units))
        c = np.zeros((n, self.units))
        self.hs = {0: h}
        self.cs = {0: c}

        self.zf = np.zeros((timestep, n, self.units))
        self.zi = np.zeros((timestep, n, self.units))
        self.zu = np.zeros((timestep, n, self.units))
        self.zo = np.zeros((timestep, n, self.units))

        for t in range(timestep):
            x = X[:, t]
            self.zf[t] = x @ self.params["Wf"] + h @ self.params["Whf"] + self.params["bf"]
            f = self.sigmoid(self.zf[t])

            self.zi[t] = x @ self.params["Wi"] + h @ self.params["Whi"] + self.params["bi"]
            i = self.sigmoid(self.zi[t])

            self.zu[t] = (x @ self.params["Wu"]) + (h @ self.params["Whu"]) + self.params["bu"]
            u = self.active.forward(self.zu[t])

            c = (c * f) + (i * u)
            self.cs[t + 1] = c

            self.zo[t] = x @ self.params["Wo"] + h @ self.params["Who"] + self.params["bo"]
            o = self.sigmoid(self.zo[t])

            h = self.active.forward(c) * o
            self.hs[t + 1] = h
        return h

    def backward(self, dh, optimizer):
        _, timestep, d = self.X.shape
        dWf = np.zeros((d, self.units))
        dWi = np.zeros((d, self.units))
        dWu = np.zeros((d, self.units))
        dWo = np.zeros((d, self.units))

        dWhf = np.zeros((self.units, self.units))
        dWhi = np.zeros((self.units, self.units))
        dWhu = np.zeros((self.units, self.units))
        dWho = np.zeros((self.units, self.units))

        dbf  = np.zeros((1, self.units))
        dbi  = np.zeros((1, self.units))
        dbu  = np.zeros((1, self.units))
        dbo  = np.zeros((1, self.units))

        dzf = np.zeros_like(dh)
        dzi = np.zeros_like(dh)
        dzu = np.zeros_like(dh)
        dzo = np.zeros_like(dh)
        dc = np.zeros_like(dh)
        dc_prev = np.zeros_like(dh)

        for t in reversed(range(timestep)):
            x = self.X[:, t]   

            # dc = dL/dh * dh/dacitve * dactive/dc
            dc = dh * self.sigmoid(self.zo[t]) * self.active.backward(self.cs[t + 1]) + dc_prev

            # dzo = (dL/dh * dh/do * do/dzo)
            dzo  = dh * self.active.forward(self.cs[t + 1]) * self.dsigmoid(self.zo[t])
            dWo  += x.T @ dzo
            dWho += self.hs[t].T @ dzo
            dbo  += np.sum(dzo, axis=0)

            # dzu = dL/dc * dc/du * du/dzu
            dzu  = dc * self.sigmoid(self.zi[t]) * self.active.backward(self.zu[t])
            dWu  += x.T @ dzu
            dWhu += self.hs[t].T @ dzu
            dbu  += np.sum(dzu, axis=0)

            # dzi = dL/dc * dc/di * di/dzi
            dzi  = dc * self.active.forward(self.zu[t]) * self.dsigmoid(self.zi[t])
            dWi  += x.T @ dzi
            dWhi += self.hs[t].T @ dzi
            dbi  += np.sum(dzi, axis=0)

            # dzf = dL/dc * dc/df * df/dzf
            dzf  = dc * self.cs[t] * self.dsigmoid(self.zf[t])
            dWf  += x.T @ dzf
            dWhf += self.hs[t].T @ dzf
            dbf  += np.sum(dzf, axis=0)

            # dc_prev = dL/dc * dc/dc_prev
            dc_prev = dc * self.sigmoid(self.zf[t])
            dh = (dzo @ self.params["Who"].T) + (dzu @ self.params["Whu"].T) + (dzi @ self.params["Whi"].T) + (dzf @ self.params["Whf"].T)

        for d in [dWf, dWi, dWu, dWo, dWhf, dWhi, dWhu, dWho, dbf, dbi, dbu, dbo]:
            np.clip(d, -1, 1, out=d)
        optimizer.update(self, self.params.keys(), dWf, dWi, dWu, dWo, dWhf, dWhi, dWhu, dWho, dbf, dbi, dbu, dbo)
        return dh


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    

    
    