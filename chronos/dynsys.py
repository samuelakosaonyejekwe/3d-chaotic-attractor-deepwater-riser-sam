"""
chronos.dynsys
==============
A minimal, universal container for an autonomous (or explicitly time-forced)
dynamical system  ``dy/dt = f(t, y)``  of arbitrary dimension.

The class provides:
  * a cached finite-difference Jacobian  (used by the Lyapunov engine and for
    fixed-point / stability analysis),
  * fixed-point location via Newton iteration,
  * linear-stability classification of a fixed point,
so that the *same* code path serves the canonical Lorenz system, the original
conventional low-order attractor and the novel riser model -- i.e. the
solver is universal.
"""

from __future__ import annotations
import numpy as np


class DynamicalSystem:
    def __init__(self, rhs, dim, name="system", state_names=None, args=(),
                 jac_fn=None):
        self.rhs = rhs
        self.dim = int(dim)
        self.name = name
        self.args = args
        self.jac_fn = jac_fn        # optional analytic Jacobian J(t, y)
        self.state_names = state_names or [f"x{i}" for i in range(dim)]

    # ------------------------------------------------------------------ #
    def f(self, t, y):
        return np.asarray(self.rhs(t, y, *self.args), dtype=float)

    def jacobian(self, t, y, eps=1e-7):
        """Analytic Jacobian when available, else central finite-difference."""
        if self.jac_fn is not None:
            return np.asarray(self.jac_fn(t, np.asarray(y, float)), dtype=float)
        return self._fd_jacobian(t, y, eps)

    def _fd_jacobian(self, t, y, eps=1e-7):
        """Central finite-difference Jacobian J_ij = d f_i / d y_j."""
        y = np.asarray(y, dtype=float)
        n = y.size
        J = np.empty((n, n))
        f0 = self.f(t, y)   # noqa (kept for readability / possible reuse)
        for j in range(n):
            h = eps * max(1.0, abs(y[j]))
            yp = y.copy(); yp[j] += h
            ym = y.copy(); ym[j] -= h
            J[:, j] = (self.f(t, yp) - self.f(t, ym)) / (2.0 * h)
        return J

    def divergence(self, t, y):
        """Phase-space volume contraction rate  div f = trace(J)."""
        return float(np.trace(self.jacobian(t, y)))

    # ------------------------------------------------------------------ #
    def fixed_point(self, y_guess, t=0.0, tol=1e-10, itmax=100):
        """Locate a fixed point via damped Newton iteration."""
        y = np.asarray(y_guess, dtype=float).copy()
        for _ in range(itmax):
            fv = self.f(t, y)
            if np.linalg.norm(fv) < tol:
                return y, True
            J = self.jacobian(t, y)
            try:
                dy = np.linalg.solve(J, -fv)
            except np.linalg.LinAlgError:
                return y, False
            y = y + dy
        return y, np.linalg.norm(self.f(t, y)) < 1e-6

    def classify_fixed_point(self, y_fp, t=0.0):
        """Return eigenvalues and a stability label for a fixed point."""
        ev = np.linalg.eigvals(self.jacobian(t, y_fp))
        re = ev.real
        if np.all(re < -1e-9):
            label = "stable (attracting)"
        elif np.all(re > 1e-9):
            label = "unstable (repelling)"
        else:
            label = "saddle / marginal"
        if np.any(np.abs(ev.imag) > 1e-9):
            label += " focus/spiral"
        return ev, label
