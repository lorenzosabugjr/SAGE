from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple


@dataclass
class HistoryBuffer:
    """
    Simple evaluation history buffer for (x, z) pairs.

    Stores all evaluations, including duplicates, to preserve evaluation
    order -- this raw record (Xn, Zn, Zn_true) is what estimators reuse.

    Separately tracks the *accepted incumbent* at each evaluation index
    (z_k_eval_hist, z_k_true_hist): every raw evaluation forward-fills the
    current incumbent by default (it is an observation only -- an initial
    stencil sample, an auxiliary probe, or a rejected line-search trial),
    and `accept_incumbent` retroactively marks the most recent evaluation as
    the new incumbent in place, so an accepted evaluation's own index (not
    the following one) reflects the new iterate.
    """
    Xn: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Zn: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    # Raw noiseless value at each evaluation, aligned with Xn/Zn (NaN where
    # not supplied).
    Zn_true: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    # Accepted-incumbent tracking (one entry per evaluation, forward-filled).
    z_k_eval_hist: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    z_k_true_hist: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    t_hist: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    # Current incumbent state, forward-filled onto every new raw evaluation
    # until `accept_incumbent` moves it. None until `init_incumbent` is
    # called (e.g. bare HistoryBuffer usage that never tracks iterates).
    _incumbent_z_eval: Optional[float] = field(default=None, repr=False)
    _incumbent_z_true: Optional[float] = field(default=None, repr=False)

    def init_incumbent(self, z_eval: float, z_true: float) -> None:
        """Mark the current values as the initial incumbent (e.g. the
        original sampled center), before any evaluation is recorded."""
        self._incumbent_z_eval = z_eval
        self._incumbent_z_true = z_true

    def accept_incumbent(self, z_eval: float = None, z_true: float = None) -> None:
        """Mark the most recently recorded raw evaluation as the newly
        accepted incumbent.

        Updates the tracked incumbent state and overwrites that same
        evaluation's z_k_eval_hist/z_k_true_hist entry in place, so
        acceptance is attributed to the evaluation that produced it rather
        than the following one. Defaults to the last raw (Zn, Zn_true)
        values when not given explicitly.
        """
        if self.Zn.size == 0:
            return
        if z_eval is None:
            z_eval = float(self.Zn[-1])
        if z_true is None and self.Zn_true.size > 0:
            z_true = float(self.Zn_true[-1])

        self._incumbent_z_eval = z_eval
        self._incumbent_z_true = z_true
        if self.z_k_eval_hist.size > 0:
            self.z_k_eval_hist[-1] = z_eval
        if self.z_k_true_hist.size > 0 and z_true is not None:
            self.z_k_true_hist[-1] = z_true

    def add(
        self,
        x: np.ndarray,
        z: float,
        z_true: float = None,
        t: float = None,
    ) -> None:
        x_row = np.atleast_2d(x)
        if self.Xn.size == 0:
            self.Xn = x_row.copy()
            self.Zn = np.atleast_1d(z).copy()
        else:
            self.Xn = np.vstack((self.Xn, x_row))
            self.Zn = np.hstack((self.Zn, z))

        # Raw noiseless value at this evaluation, aligned with Zn.
        self.Zn_true = np.hstack((self.Zn_true, z_true if z_true is not None else np.nan))

        # Forward-fill the current incumbent onto this evaluation. This is
        # an observation by default; accept_incumbent() retroactively fixes
        # it up when this very evaluation is accepted as the new iterate.
        if self._incumbent_z_eval is not None:
            self.z_k_eval_hist = np.hstack((self.z_k_eval_hist, self._incumbent_z_eval))
            self.z_k_true_hist = np.hstack((self.z_k_true_hist, self._incumbent_z_true))

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

