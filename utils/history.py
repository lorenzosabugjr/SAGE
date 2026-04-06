from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple


@dataclass
class HistoryBuffer:
    """
    Simple evaluation history buffer for (x, z) pairs.

    Stores all evaluations, including duplicates, to preserve evaluation order.
    Also tracks the current iterate's values at each evaluation.
    """
    Xn: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Zn: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    
    # Current iterate tracking (one entry per evaluation)
    z_k_eval_hist: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    z_k_true_hist: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    t_hist: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    def add(
        self,
        x: np.ndarray,
        z: float,
        z_k_eval: float = None,
        z_k_true: float = None,
        t: float = None,
    ) -> None:
        x_row = np.atleast_2d(x)
        if self.Xn.size == 0:
            self.Xn = x_row.copy()
            self.Zn = np.atleast_1d(z).copy()
        else:
            self.Xn = np.vstack((self.Xn, x_row))
            self.Zn = np.hstack((self.Zn, z))
        
        # Track iterate state if provided
        if z_k_eval is not None:
            self.z_k_eval_hist = np.hstack((self.z_k_eval_hist, z_k_eval))
        if z_k_true is not None:
            self.z_k_true_hist = np.hstack((self.z_k_true_hist, z_k_true))
        if t is not None:
            self.t_hist = np.hstack((self.t_hist, t))

    def add_batch(self, X: np.ndarray, Z: np.ndarray) -> None:
        X = np.atleast_2d(X)
        Z = np.atleast_1d(Z)
        if X.size == 0:
            return
        if self.Xn.size == 0:
            self.Xn = X.copy()
            self.Zn = Z.copy()
            return
        self.Xn = np.vstack((self.Xn, X))
        self.Zn = np.hstack((self.Zn, Z))

    def find_indices(self, x: np.ndarray) -> np.ndarray:
        if self.Xn.size == 0:
            return np.empty((0,), dtype=int)
        x_in = np.all(np.equal(self.Xn, x), axis=1)
        return np.nonzero(x_in)[0]

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.Xn, self.Zn

