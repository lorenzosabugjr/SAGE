import numpy as np
from .base import BaseProblem

class LogSumExp(BaseProblem):
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
        clp = 150
        z = clp*np.tanh((self.A@x - self.b)/clp)
        q = np.log(np.exp(z).sum()) + self.lmbd/2 * np.dot(x,x)
        return q
