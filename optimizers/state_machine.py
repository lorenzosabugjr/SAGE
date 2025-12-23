import numpy as np
import scipy as cp
from numpy.linalg import norm
from enum import Enum, unique


@unique
class State(Enum):
    REFINE_GDT = 0
    LINE_SCH = 1


@unique
class StepSizeMode(Enum):
    FIXED = 0
    ADAPTIVE = 1


class BaseOptim:
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.FIXED,
        bfgs: bool = False,
    ):
        # Store the data set
        self.Xn = Xn
        self.Zn = np.array(Zn)

        self.k = 0
        if Zn.shape == ():
            self.D = Xn.shape[0]
            self.n = 1
        else:
            self.D = Xn.shape[1]
            self.n = Zn.shape[0]

        self.state = State.REFINE_GDT  # self.state is by default REFINE_GDT
        self.gdt_est = np.zeros(self.D)
        self.gdt_est_rdy = False
        self.ns_est = 0.0  # Estimated noise bounds (if available from the specific solver)

        # Step size-related parameters
        self.eta0 = stepsize  # Default step size
        self.eta = self.eta0  # Current step size
        self.etaM = 0.5  # Step size multiplier (backtracking)
        self.etaT = 1e-6  # Factor for Armijo's condition
        self.eta_mode = stepsizemode

        # Generate initial iterate from the data set
        if self.Zn.shape == ():
            self.x_k = self.Xn
            self.z_k = self.Zn
        else:
            x_kk = np.argmin(self.Zn)
            self.x_k = self.Xn[x_kk, :]
            self.z_k = self.Zn[x_kk]

        self.x_kp = self.x_k

        # BFGS-related parameters
        self.bfgs = bfgs
        self.bfgs_hinv = np.identity(self.D)  # Hessian estimate used in BFGS
        self.bfgs_gdtp = np.zeros(
            self.D,
        )  # Previous gradient estimate used in BFGS

        self.state_machine()

    # Method add_samples when supplying a complete data set
    # i.e., with sampled points, and corresponding values
    def add_samples(self, Xadd: np.ndarray, Zadd: np.ndarray):
        assert np.atleast_2d(Xadd).shape[1] == self.D
        assert np.atleast_1d(Zadd).shape[0] == np.atleast_2d(Xadd).shape[0]

        self.Xn = np.vstack((self.Xn, Xadd))
        self.Zn = np.hstack((self.Zn, Zadd))
        self.n = self.Xn.shape[0]
        if (Xadd == self.x_k).all():
            self.z_k = Zadd
        self.z_n = Zadd

        self.add_samples_gdtest(Xadd, Zadd)

        self.state_machine()

    def add_samples_gdtest(self, Xadd, Zadd):
        pass

    # Method add_samples when supplying a sampled value
    def add_value(self, zadd: float):
        return self.add_samples(self.x_n, zadd)

    def state_machine(self):
        if self.state == State.REFINE_GDT:
            if self.gdt_est_rdy:

                if np.isnan(self.bfgs_hinv).any():
                    self.bfgs_hinv = np.identity(self.D)

                if self.bfgs:
                    B = self.bfgs_hinv
                    Id = np.identity(self.D)
                    sk = np.atleast_2d(self.x_k - self.x_kp).T
                    yk = np.atleast_2d(self.gdt_est - self.bfgs_gdtp).T
                    if not (yk.T @ sk == 0.0).all():
                        rho = 1 / (yk.T @ sk)
                        self.bfgs_hinv = (Id - rho * sk @ yk.T) @ B @ (
                            Id - rho * yk @ sk.T
                        ) + (rho * sk @ sk.T)
                    self.bfgs_gdtp = self.gdt_est

                # Compute state machine
                if self.eta_mode == StepSizeMode.ADAPTIVE:
                    self.state = State.LINE_SCH
                    if self.bfgs:
                        self.x_n = -self.eta * self.bfgs_hinv @ self.gdt_est + self.x_k
                    else:
                        self.x_n = -self.eta * self.gdt_est + self.x_k
                else:
                    self.state = State.REFINE_GDT
                    self.k += 1
                    self.x_k = -self.eta0 * self.gdt_est + self.x_k
                    self.x_n = self.x_k
        else:
            # Test Armijo's condition
            if (self.z_n <= self.z_k - self.etaT * self.eta * norm(self.gdt_est) ** 2):
                # If passed, update iterate and switch to refine gradient mode
                self.state = State.REFINE_GDT
                self.k += 1
                self.x_kp = self.x_k
                self.x_k = self.x_n
                self.z_k = self.z_n

                self.eta = self.eta / self.etaM
            else:
                # If fail, adjust step size and try another line search
                self.state = State.LINE_SCH
                if self.eta > 1e-6:
                    self.eta = self.eta * self.etaM
                else:
                    self.add_samples_gdtest(None, None)
                    self.eta = self.eta0
                self.x_n = self.x_k - self.eta * self.gdt_est


class FFDOpt(BaseOptim):
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        bfgs: bool = False,
        ffdstep: float = 1e-6,
    ):
        super().__init__(Xn, Zn, stepsize, stepsizemode, bfgs)

        self.ffd_step = ffdstep
        self.ffd_samples = 0
        self.ffd_Xn = np.empty((0, self.D))
        self.ffd_Zn = np.empty(0)
        self.add_samples_gdtest(None, None)

        self.state_machine()

    def add_samples_gdtest(self, Xadd, Zadd):
        if self.state == State.REFINE_GDT and self.ffd_samples < self.D:
            if (
                Xadd is not None
                and Zadd is not None
                and (Xadd != self.x_k).any()
                and (Zadd != self.z_k).any()
            ):
                self.ffd_Xn = np.vstack((self.ffd_Xn, Xadd))
                self.ffd_Zn = np.hstack((self.ffd_Zn, Zadd))
                self.ffd_samples += 1

            if self.ffd_samples < self.D:
                self.state = State.REFINE_GDT

                # I generate the next sampling point by iterating
                #   through the FFD steps from the current iterate
                stp = np.zeros(self.D)
                stp[self.ffd_samples] = self.ffd_step

                self.x_n = self.x_k + stp
                self.gdt_est_rdy = False
            else:
                # Start to actually calculate the gradient estimate
                self.gdt_est = np.zeros(self.D)
                for idx in range(self.D):
                    self.gdt_est[idx] = (self.ffd_Zn[idx] - self.z_k) / self.ffd_step

                # Reset data used for FFD calculations
                self.ffd_samples = 0
                self.ffd_Xn = np.empty((0, self.D))
                self.ffd_Zn = np.empty(0)

                self.gdt_est_rdy = True


class CFDOpt(BaseOptim):
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        cfdstep: float = 1e-6,
        bfgs: bool = False,
    ):
        super().__init__(Xn, Zn, stepsize, stepsizemode, bfgs)

        self.cfd_step = cfdstep
        self.cfd_samples = 0
        self.cfd_Xn = np.empty((0, self.D))
        self.cfd_Zn = np.empty(0)
        self.add_samples_gdtest(None, None)

        self.state_machine()

    def add_samples_gdtest(self, Xadd, Zadd):
        if self.state == State.REFINE_GDT and self.cfd_samples < 2 * self.D:
            if (
                Xadd is not None
                and Zadd is not None
                and (Xadd != self.x_k).any()
                and (Zadd != self.z_k).any()
            ):
                self.cfd_Xn = np.vstack((self.cfd_Xn, Xadd))
                self.cfd_Zn = np.hstack((self.cfd_Zn, Zadd))
                self.cfd_samples += 1

            if self.cfd_samples < 2 * self.D:
                self.state = State.REFINE_GDT

                # I generate the next sampling point by iterating
                #   through the CFD steps from the current iterate
                stp = np.zeros(self.D)
                sgn = (-1) ** self.cfd_samples
                stp[int(self.cfd_samples / 2)] = sgn * self.cfd_step

                self.x_n = self.x_k + stp
                self.gdt_est_rdy = False
            else:
                # Start to actually calculate the gradient estimate
                self.gdt_est = np.zeros(self.D)
                for idx in range(self.D):
                    self.gdt_est[idx] = (
                        self.cfd_Zn[2 * idx] - self.cfd_Zn[2 * idx + 1]
                    ) / (2 * self.cfd_step)

                # Reset data used for CFD calculations
                self.cfd_samples = 0
                self.cfd_Xn = np.empty((0, self.D))
                self.cfd_Zn = np.empty(0)

                self.gdt_est_rdy = True


class GSGOpt(BaseOptim):
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        m: int,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        u: float = 1e-6,
        bfgs: bool = False,
        seed: int | None = None,
    ):
        super().__init__(Xn, Zn, stepsize, stepsizemode, bfgs)

        self.u = u
        self.m = m

        self.e_gen = np.random.default_rng(seed)
        self.e = self.e_gen.normal(size=(self.m, self.D))

        self.gsg_samples = 0
        self.gsg_Xn = np.empty((0, self.D))
        self.gsg_Zn = np.empty(0)
        self.add_samples_gdtest(None, None)

        self.state_machine()

    def add_samples_gdtest(self, Xadd, Zadd):
        if self.state == State.REFINE_GDT and self.gsg_samples < self.m:
            if (
                Xadd is not None
                and Zadd is not None
                and (Xadd != self.x_k).any()
                and (Zadd != self.z_k).any()
            ):
                self.gsg_Xn = np.vstack((self.gsg_Xn, Xadd))
                self.gsg_Zn = np.hstack((self.gsg_Zn, Zadd))
                self.gsg_samples += 1

            if self.gsg_samples < self.m:
                self.state = State.REFINE_GDT

                # I generate the next sampling point by iterating
                #   through the CFD steps from the current iterate
                e = self.e[self.gsg_samples, :]

                self.x_n = self.x_k + self.u * e
                self.gdt_est_rdy = False
            else:
                # Start to actually calculate the gradient estimate
                self.gdt_est = np.zeros(self.D)
                for idx in range(self.m):
                    self.gdt_est += (self.gsg_Zn[idx] - self.z_k) / self.u * self.e[
                        idx, :
                    ]
                self.gdt_est /= self.m

                if (
                    np.max(
                        abs((self.gsg_Xn[-1, :] - self.x_k) / self.u - self.e[-1, :])
                    )
                    > 1e-4
                ):
                    self.gdt_est = 0

                # Reset data used for CFD calculations
                self.gsg_samples = 0
                self.gsg_Xn = np.empty((0, self.D))
                self.gsg_Zn = np.empty(0)
                self.e = self.e_gen.normal(size=(self.m, self.D))

                self.gdt_est_rdy = True


class cGSGOpt(BaseOptim):
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        m: int,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        u: float = 1e-6,
        bfgs: bool = False,
        seed: int | None = None,
    ):
        super().__init__(Xn, Zn, stepsize, stepsizemode, bfgs)

        self.u = u
        self.m = m

        self.e_gen = np.random.default_rng(seed)
        self.e = self.e_gen.normal(size=(self.m, self.D))

        self.cgsg_samples = 0
        self.cgsg_Xn = np.empty((0, self.D))
        self.cgsg_Zn = np.empty(0)
        self.add_samples_gdtest(None, None)

        self.state_machine()

    def add_samples_gdtest(self, Xadd, Zadd):
        if self.state == State.REFINE_GDT and self.cgsg_samples < 2 * self.m:
            if (
                Xadd is not None
                and Zadd is not None
                and (Xadd != self.x_k).any()
                and (Zadd != self.z_k).any()
            ):
                self.cgsg_Xn = np.vstack((self.cgsg_Xn, Xadd))
                self.cgsg_Zn = np.hstack((self.cgsg_Zn, Zadd))
                self.cgsg_samples += 1

            if self.cgsg_samples < 2 * self.m:
                self.state = State.REFINE_GDT

                # I generate the next sampling point by iterating
                #   through the CFD steps from the current iterate
                sgn = (-1) ** self.cgsg_samples
                e = self.e[int(self.cgsg_samples / 2), :]

                self.x_n = self.x_k + sgn * self.u * e
                self.gdt_est_rdy = False
            else:
                # Start to actually calculate the gradient estimate
                self.gdt_est = np.zeros(self.D)
                for idx in range(self.m):
                    self.gdt_est += (self.cgsg_Zn[2 * idx] - self.cgsg_Zn[2 * idx + 1]) / (
                        2 * self.u
                    ) * self.e[idx, :]
                self.gdt_est /= self.m

                if (
                    np.max(
                        abs((self.cgsg_Xn[-2, :] - self.x_k) / self.u - self.e[-1, :])
                    )
                    > 1e-4
                ):
                    self.gdt_est = 0

                # Reset data used for CFD calculations
                self.cgsg_samples = 0
                self.cgsg_Xn = np.empty((0, self.D))
                self.cgsg_Zn = np.empty(0)
                self.e = self.e_gen.normal(size=(self.m, self.D))

                self.gdt_est_rdy = True


class NMXFDOpt(BaseOptim):
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        m: int,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        rangeintegral: tuple = (-2, 2),
        numpoints: int = 4,
        sigma: float = 1e-2,
        lmbd: float = 1e-3,
        bfgs: bool = False,
    ):
        super().__init__(Xn, Zn, stepsize, stepsizemode, bfgs)

        self.n_u = numpoints
        self.u = np.linspace(rangeintegral[0], rangeintegral[1], numpoints, dtype="d")
        self.delta = self.u[1] - self.u[0]
        self.sigma = sigma
        self.lmbd = lmbd
        self.dphi = -self.gaussian_derivative(self.u, 0, 1).reshape(-1, 1)
        self.m = m

        self.nmxfd_samples = 0
        self.nmxfd_Xn = np.empty((0, self.D))
        self.nmxfd_Zn = np.empty(0)
        self.add_samples_gdtest(None, None)

        self.state_machine()

    # Taken from https://github.com/marcoboresta/NMXFD-gradient-approximation/blob/master/gradient_approximation_methods.py
    def gaussian_derivative(self, x, mu, sigma):
        # Compute the denominator (normalization factor for Gaussian derivative)
        denominator = np.sqrt(2 * np.pi) * (sigma**3)
        # Compute the numerator (scaled Gaussian function)
        numerator = (x - mu) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        # Return the derivative
        return -numerator / denominator

    def add_samples_gdtest(self, Xadd, Zadd):
        if self.state == State.REFINE_GDT and self.nmxfd_samples < self.D * len(self.u):
            if (
                Xadd is not None
                and Zadd is not None
                and (Xadd != self.x_k).any()
                and (Zadd != self.z_k).any()
            ):
                self.nmxfd_Xn = np.vstack((self.nmxfd_Xn, Xadd))
                self.nmxfd_Zn = np.hstack((self.nmxfd_Zn, Zadd))
                self.nmxfd_samples += 1

            if self.nmxfd_samples < self.D * self.n_u:
                self.state = State.REFINE_GDT

                # I generate the next sampling point by iterating
                #   through the NMXFD steps from the current iterate
                dim = int(self.nmxfd_samples / self.n_u)
                stp = self.nmxfd_samples % self.n_u
                dx = np.zeros(self.D)
                dx[dim] = self.u[stp]
                self.x_n = self.x_k + dx

                self.gdt_est_rdy = False
            else:
                self.gdt_est = np.zeros(self.D)

                # Start to actually calculate the gradient estimate
                for j in range(self.D):
                    res = self.nmxfd_Zn[(j * self.n_u) : (j * self.n_u + self.n_u)]
                    differences = np.array(
                        [res[::-1][k] - res[k] for k in range(self.n_u // 2)]
                    ).reshape(
                        -1, 1
                    )  # Compute finite differences by calculating difference from opposite ends of the array
                    phi_coeff = np.abs(self.dphi[: self.n_u // 2])

                    # Calculate integration coefficients
                    h = (self.u[-1] - self.u[0]) / (self.n_u - 1)
                    mult = self.u[::-1][: self.n_u // 2].reshape(-1, 1).astype(float)
                    coeff = phi_coeff * mult * h
                    coeff[0] /= 2  # Adjust for boundary conditions

                    # Normalize coefficients if needed
                    normd_coeff = coeff / np.sum(coeff)

                    # Compute gradient contribution for the current dimension
                    output = normd_coeff * differences / (mult * self.sigma) / 2
                    self.gdt_est[j] = np.sum(output)

                # Reset data used for NMXFD calculations
                self.nmxfd_samples = 0
                self.nmxfd_Xn = np.empty((0, self.D))
                self.nmxfd_Zn = np.empty(0)

                self.gdt_est_rdy = True


class SAGEOpt(BaseOptim):
    def __init__(
        self,
        Xn: np.ndarray,
        Zn: np.ndarray,
        stepsize: float = 1.0,
        stepsizemode: StepSizeMode = StepSizeMode.ADAPTIVE,
        bfgs: bool = False,
        autonoise: bool = True,
        noisebnd: float = 0.0,
        quickmode: bool = True,
    ):
        super().__init__(Xn, Zn, stepsize, stepsizemode, bfgs)

        self.gdtset_diaid = 0.05  # Ideal gradient set diameter
        self.gdtset_diath = 0.05  # Gradient set maximum diameter. If the estimated gradient
        #   set diameter is higher than this, we try additional
        #   samples to shrink the gradient set.
        self.gdt_est_frc = False  # A flag to force the acceptance of the current gradient estimate
        #   e.g., when are too many auxiliary samples already done

        # Gradient estimator-relevant quantities
        self.autonoise = autonoise  # Do we need to estimate the noise?
        self.ns_est = 0.0 if autonoise else noisebnd  # Estimated noise bounds
        self.hess_norm = 0.0
        self.hess_lipsc = 0.0
        self.A2 = None
        self.b2 = None
        self.gd_v = np.nan  # Chord vector defining the gradient set diameter
        self.gd_vm = np.inf  # Gradient set diameter

        self.quickmode = quickmode  # Compute gradient estimates only from limited samples around current iterate

        # For logging purposes: history of count of auxiliary samples per iteration
        self.aux_samples = 0
        self.hist_aux_samples = np.empty((0,))

        self.add_samples_gdtest(Xn, Zn)
        self.state_machine()

    def update_gradient(self, x: np.ndarray = None):
        if x is None:
            x = self.x_k

        # Update gradient estimate g
        self.grad_est(x)

        # Update gradient set uncertainty radius
        self.grad_set_diam(x)

    def grad_est(self, x: np.ndarray = None):
        # If no x is supplied, simply return the gradient estimate
        # at the current iterate self.x_k
        if x is None:
            x = self.x_k

        A = np.empty([0, self.D + 2 + 1 * self.autonoise])
        b = np.empty([0, 1])

        x_in = np.all(np.equal(self.Xn, x), axis=1)
        assert np.any(x_in), "Is your supplied x included in the data set?"
        x_idx = np.nonzero(x_in)[0]

        # Do a for-loop for the rest of the samples, denote as xj

        # If quickcalc is enabled, collect only 5*D number of samples
        # according to distance from the current iterate
        if self.Zn.size > 5 * self.D + 1 and self.quickmode:
            coll_x = [self.Xn[j] for j in range(self.Zn.size) if j not in x_idx]
            coll_idx = [j for j in range(self.Zn.size) if j not in x_idx]

            # Computing the best radius from x_k for optimal gradient set uncertainty
            # We now get the roots of the DERIVATIVE of the uncertainty bounds w.r.t. mu_ij
            aa = 1 / 3 * self.hess_lipsc
            bb = 1 / 2 * self.hess_norm
            dd = -2 * self.ns_est
            rt = np.roots([aa, bb, 0, dd])
            # rt always contains 3 roots: 1 real-positive and 2 complex
            # roots. The real root is the one we are interested in.
            alpha = rt[np.isreal(rt) & (rt.real >= 0)]
            if alpha.size == 0:
                alpha = 0
            else:
                alpha = alpha.real[0]

            cost_fn = np.abs(np.sum((coll_x - x) ** 2, axis=1) - alpha**2)
            sort_idx = np.argsort(cost_fn)[: 5 * self.D]
            coll_idx = [coll_idx[j] for j in sort_idx]
        else:
            coll_idx = [j for j in range(self.Zn.size) if j not in x_idx]

        for j in coll_idx:
            z = self.Zn[x_idx[0]]
            # - Compute the distance (dij)
            dij = np.linalg.norm(self.Xn[j] - x)
            if dij == 0.0:
                dij = 1.0
            # - Compute the unit direction vector (uij)
            uij = (self.Xn[j] - x) / dij
            # - Compute the estimated gradient (gij)
            gij = (self.Zn[j] - z) / dij
            # - Append two rows to matrix A
            if not self.autonoise:
                A = np.vstack((A, np.hstack((-uij, -0.5 * dij, -1 / 6 * dij**2))))
                A = np.vstack((A, np.hstack((uij, -0.5 * dij, -1 / 6 * dij**2))))
            else:
                A = np.vstack(
                    (A, np.hstack((-uij, -0.5 * dij, -1 / 6 * dij**2, -2 / dij)))
                )
                A = np.vstack(
                    (A, np.hstack((uij, -0.5 * dij, -1 / 6 * dij**2, -2 / dij)))
                )
            # - Append two entries to column vector b
            # b = np.vstack((b,-gij+2*self.ns_est/dij,gij+2*self.ns_est/dij))
            b = np.vstack((b, -gij, gij))

            if np.isnan(A).any():
                raise ValueError("Matrix A should not have any NaN entries.")

        if not self.autonoise:
            Ae = np.vstack((A, np.hstack((np.zeros(A.shape[1] - 2), -1, 0))))
            Ae = np.vstack((Ae, np.hstack((np.zeros(A.shape[1] - 2), 0, -1))))
            be = np.vstack((b, 0, 0))
        else:
            Ae = np.vstack((A, np.hstack((np.zeros(A.shape[1] - 3), -1, 0, 0))))
            Ae = np.vstack((Ae, np.hstack((np.zeros(A.shape[1] - 3), 0, -1, 0))))
            Ae = np.vstack((Ae, np.hstack((np.zeros(A.shape[1] - 3), 0, 0, -1))))
            be = np.vstack((b, 0, 0, 0))

        be = be.flatten()

        # Do the optimisation, documentation is given at
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
        if not self.autonoise:
            res = cp.optimize.linprog(
                np.hstack((np.zeros(Ae.shape[1] - 2), 1, 1)),
                A_ub=Ae,
                b_ub=be,
                bounds=(None, None),
                method="highs-ipm",
            )
            self.gdt_est = res.x[:-2]
            self.hess_norm = res.x[-2]  # Norm of Hessian at current point
            self.hess_lipsc = np.max(
                [res.x[-1], self.hess_lipsc]
            )  # Lipschitz constant for Hessian Lipschitz continuity
        else:
            res = cp.optimize.linprog(
                np.hstack((np.zeros(Ae.shape[1] - 3), 1, 1, 1)),
                A_ub=Ae,
                b_ub=be,
                bounds=(None, None),
            )
            if res.success:
                self.gdt_est = res.x[:-3]
                self.hess_norm = res.x[-3]  # Norm of Hessian at current point
                self.hess_lipsc = np.max(
                    [res.x[-2], self.hess_lipsc]
                )  # Lipschitz constant for Hessian Lipschitz continuity
                self.ns_est = res.x[-1]  # Estimated noise bounds

        self.Al = A[:, 0 : self.D]
        Ar = A[:, self.D :]
        self.A2 = np.vstack(
            (
                np.hstack((self.Al, np.zeros(self.Al.shape))),
                np.hstack((np.zeros(self.Al.shape), self.Al)),
            )
        )

        if not self.autonoise:
            self.bl = b.flatten() - (Ar @ np.array([self.hess_norm, self.hess_lipsc]))
        else:
            self.bl = b.flatten() - (
                Ar @ np.array([self.hess_norm, self.hess_lipsc, self.ns_est])
            )
        self.b2 = np.vstack((self.bl, self.bl)).flatten()

        return self.gdt_est

    def grad_set_diam(self, xi: np.ndarray = None):
        # If no x is supplied, simply return the gradient estimate
        # at the current iterate self.x_k
        if xi is None:
            xi = self.x_k

        # The gradient set is defined by matrices self.A2 and self.b2
        # Therefore if A2 does not exist, then we need to compute
        # self.A2 and self.b2 by grad_est
        if self.A2 is None:
            self.grad_est(xi)

        # Find the diameter of the found gradients set (non-convex problem)
        # Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        P = np.vstack(
            (
                np.hstack((np.identity(self.D), -np.identity(self.D))),
                np.hstack((-np.identity(self.D), np.identity(self.D))),
            )
        )

        def obj(x):
            return -x.T @ P @ x

        cons = {"type": "ineq", "fun": lambda x: -(self.A2 @ x - self.b2)}
        res = cp.optimize.minimize(
            obj,
            np.hstack((self.gdt_est, self.gdt_est)) + 1e-3 * np.random.rand(2 * self.D),
            method="SLSQP",
            constraints=cons,
            options={"disp": False},
        )

        # The results of this operation is a vector of TWO gradients:
        # these gradients are those in the (polytopic) gradient set, but which are
        # the most different from each other. The difference between these gradients
        # is the "diameter" of the gradient set.

        # gd_v is the vector describing the diameter of the gradient set,
        # computed by taking the difference between the two halves of res.x
        self.gd_v = res.x[: self.D] - res.x[self.D :]
        self.gd_vm = np.linalg.norm(self.gd_v)

        return self.gd_vm

    def add_samples_gdtest(self, Xadd, Zadd):
        self.update_gradient()
        if self.state == State.REFINE_GDT:

            if self.gd_vm < self.gdtset_diath or self.gdt_est_frc:
                self.hist_aux_samples = np.hstack(
                    (self.hist_aux_samples, self.aux_samples)
                )
                self.aux_samples = 0
                self.gdt_est_rdy = True
                self.gdt_est_frc = False
            else:
                # Compute the next auxiliary sample self.x_n
                if self.ns_est <= 1e-9:
                    alpha = 1e-6
                    self.gdtset_diath = self.gdtset_diaid
                else:
                    # In case of finite noise bounds, I have to compute the radius for which
                    # there will be minimum resulting gradient estimate uncertainty

                    aa = 1 / 3 * self.hess_lipsc
                    bb = 1 / 2 * self.hess_norm
                    dd = -2 * self.ns_est
                    rt = np.roots([aa, bb, 0, dd])
                    # rt always contains 3 roots: 1 real-positive and 2 complex
                    # roots. The real root is the one we are interested in.
                    alpha = rt[np.isreal(rt) & (rt.real >= 0)].real[0]
                    self.gdtset_diath = 1.01 * alpha  # Just a little multiplier for safety
                    # (not getting stuck in gradient refinement stage)

                self.x_n = self.x_k + alpha * self.gd_v / norm(self.gd_v)
                self.aux_samples += 1

                # If taking too much iterations on sampling, just force the line search
                if self.aux_samples >= 2.5 * self.D:
                    self.gdt_est_frc = True

                self.gdt_est_rdy = False
