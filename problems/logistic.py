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
