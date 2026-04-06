import numpy as np
from .base import BaseProblem

class L1LogReg(BaseProblem):
    def __init__(self, dims:int, condnum:float, lmbd:float=1e-3, randseed:int=1):
        super().__init__(randseed)
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)
        self.lmbd = lmbd

    def _eval_deterministic(self, x: np.ndarray) -> float:
        clp = 200
        z = clp*np.tanh(self.b@self.A@x / clp)
        # z = self.b@self.A@x
        q = np.log(1 + np.exp(-z)) + self.lmbd * abs(x).sum()
        if q == np.nan:
            print(z)
        return q

    def gradient(self, x: np.ndarray) -> np.ndarray:
        clp = 200
        u = self.b @ self.A @ x          # scalar
        z = clp * np.tanh(u / clp)        # clipped scalar
        # dz/du = sech^2(u/clp) = 1 - tanh^2(u/clp)
        sech2 = 1.0 - np.tanh(u / clp) ** 2
        # df/dz = -1/(1+exp(z)):  stable via -sigmoid(-z)
        sig = 1.0 / (1.0 + np.exp(-z))   # sigmoid(z)
        dfdz = -(1.0 - sig)              # = -1/(1+exp(z))
        return dfdz * sech2 * (self.A.T @ self.b) + self.lmbd * np.sign(x)

class L2LogReg(BaseProblem):
    def __init__(self, dims:int, condnum:float, lmbd:float=1e-3, randseed:int=1):
        super().__init__(randseed)
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)
        self.lmbd = lmbd

    def _eval_deterministic(self, x: np.ndarray) -> float:
        clp = 200
        z = clp*np.tanh(self.b@self.A@x / clp)
        # z = self.b@self.A@x
        q = np.log(1 + np.exp(-z)) + self.lmbd/2 * np.dot(x,x)
        return q

    def gradient(self, x: np.ndarray) -> np.ndarray:
        clp = 200
        u = self.b @ self.A @ x          # scalar
        z = clp * np.tanh(u / clp)        # clipped scalar
        sech2 = 1.0 - np.tanh(u / clp) ** 2
        sig = 1.0 / (1.0 + np.exp(-z))
        dfdz = -(1.0 - sig)
        return dfdz * sech2 * (self.A.T @ self.b) + self.lmbd * x
