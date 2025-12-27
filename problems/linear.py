import numpy as np
from .base import BaseProblem

class LeastSquares(BaseProblem):
    def __init__(self, dims:int, condnum:float, randseed:int=1):
        super().__init__(randseed)
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)

        self.x_opt = np.linalg.pinv(self.A)@self.b
        self.z_opt = 0.5*(self.b - self.A@self.x_opt).T.dot(self.b - self.A@self.x_opt)

    def _eval_deterministic(self, x: np.ndarray) -> float:
        q = self.b - self.A@x
        return 0.5*q.T.dot(q) - self.z_opt

class Lasso(BaseProblem):
    def __init__(self, dims:int, condnum:float, randseed:int=1):
        super().__init__(randseed)
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)

    def _eval_deterministic(self, x: np.ndarray) -> float:
        q = self.b - self.A@x
        return 0.5*q.T.dot(q) + abs(x).sum()
