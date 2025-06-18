import numpy as np
from sklearn.datasets import load_diabetes

class BaseProblem:
    def __init__(self, randseed:int=1):
        pass

    def eval(self, x:np.array, noisebnd:float=0.0):
        pass

# ==========================================
# LEAST SQUARES COST FUNCTION
# ==========================================
class LeastSquares(BaseProblem):
    def __init__(self, dims:int, condnum:float, randseed:int=1):
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

    def eval(self, x, noisebnd:float=0.0):
        q = self.b - self.A@x
        return 0.5*q.T.dot(q) - self.z_opt + noisebnd*(np.random.rand(1)-0.5)[0]

# ==========================================
# LASSO (L1-REGULARIZED LEAST SQUARES) COST FUNCTION
# ==========================================
class Lasso(BaseProblem):
    def __init__(self, dims:int, condnum:float, randseed:int=1):
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)

    def eval(self, x, noisebnd:float=0.0):
        q = self.b - self.A@x
        return 0.5*q.T.dot(q) + abs(x).sum() + noisebnd*(np.random.rand(1)-0.5)[0]

# ==========================================
# DIABETES (LINEAR REGRESSION) COST FUNCTION
# ==========================================
class DiabetesRegression(BaseProblem):
    def __init__(self):
        # Load dataset (diabetes)
        self.X, self.y = load_diabetes(return_X_y=True, as_frame=False)

        self.x_opt = np.linalg.pinv(self.X)@self.y
        q = self.y - self.X@self.x_opt
        self.z_opt = 1//self.y.shape[0] * q.T.dot(q)
    
    def eval(self, x, noisebnd:float=0.0):
        q = self.y - self.X@x
        return 1/self.y.shape[0] * q.T.dot(q) - self.z_opt + noisebnd*(np.random.rand(1)-0.5)[0]


# ==========================
# SPARSE LOGISTIC REGRESSION
# ==========================
class L1LogReg(BaseProblem):
    def __init__(self, dims:int, condnum:float, lmbd:float=1e-3, randseed:int=1):
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)
        self.lmbd = lmbd

    def eval(self, x, noisebnd:float=0.0):
        clp = 200
        z = clp*np.tanh(self.b@self.A@x / clp)
        # z = self.b@self.A@x
        q = np.log(1 + np.exp(-z)) + self.lmbd * abs(x).sum()
        if q == np.nan:
            print(z)
        return q + noisebnd*(np.random.rand(1)-0.5)[0]

# ==================================
# L2-REGULARIZED LOGISTIC REGRESSION
# ==================================
class L2LogReg(BaseProblem):
    def __init__(self, dims:int, condnum:float, lmbd:float=1e-3, randseed:int=1):
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)
        self.lmbd = lmbd

    def eval(self, x, noisebnd:float=0.0):
        clp = 200
        z = clp*np.tanh(self.b@self.A@x / clp)
        # z = self.b@self.A@x
        q = np.log(1 + np.exp(-z)) + self.lmbd/2 * np.dot(x,x)
        return q + noisebnd*(np.random.rand(1)-0.5)[0]

# ===========
# LOG-SUM-EXP
# ===========
class LogSumExp(BaseProblem):
    def __init__(self, dims:int, condnum:float, lmbd:float=1e-3, randseed:int=1):
        np.random.seed(randseed)
        A = 2*np.random.rand(dims,dims) - 1
        A = A + A.T
        u,s,v = np.linalg.svd(A)

        s = s[0]*(1-((condnum-1)/condnum)*(s[0]-s)/(s[0]-s[-1]))
        s = np.diag(s)

        self.A = u@s@v
        self.b = np.random.rand(dims)
        self.lmbd = lmbd

    def eval(self, x, noisebnd:float=0.0):
        clp = 150
        z = clp*np.tanh((self.A@x - self.b)/clp)
        # z = self.A@x - self.b
        q = np.log(np.exp(z).sum()) + self.lmbd/2 * np.dot(x,x)
        return q + noisebnd*(np.random.rand(1)-0.5)[0]
