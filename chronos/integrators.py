"""
chronos.integrators
====================
Self-contained, universal ODE integrators.

Two embedded adaptive Runge-Kutta pairs are implemented from their Butcher
tableaux (no dependency on ``scipy.integrate``), so the engine is fully
transparent and portable:

* ``rkf45``   -- Runge-Kutta-Fehlberg 4(5)  (Fehlberg, 1969)  -- the method
                as used in conventional low-order solvers, re-implemented with a proper
                embedded error estimate and PI step-size controller.
* ``dopri5``  -- Dormand-Prince 5(4)        (Dormand & Prince, 1980).

Both accept an arbitrary right-hand side ``f(t, y, *args) -> dy`` where ``y``
is a 1-D ``numpy`` array of any dimension, making the solver universal.

A dense, uniformly-sampled solution is returned via 4th-order cubic-Hermite
interpolation (matching state and derivative at every node) so that every
downstream diagnostic (FFT, wavelet, rainflow, Poincare) receives an
evenly-spaced time series that preserves the integrator's accuracy.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
#  Butcher tableaux
# ----------------------------------------------------------------------------
# Runge-Kutta-Fehlberg 4(5)  (Fehlberg 1969)
_RKF_C = np.array([0.0, 1/4, 3/8, 12/13, 1.0, 1/2])
_RKF_A = [
    [],
    [1/4],
    [3/32, 9/32],
    [1932/2197, -7200/2197, 7296/2197],
    [439/216, -8.0, 3680/513, -845/4104],
    [-8/27, 2.0, -3544/2565, 1859/4104, -11/40],
]
_RKF_B5 = np.array([16/135, 0.0, 6656/12825, 28561/56430, -9/50, 2/55])   # 5th order
_RKF_B4 = np.array([25/216, 0.0, 1408/2565, 2197/4104, -1/5, 0.0])        # 4th order

# Dormand-Prince 5(4)
_DP_C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
_DP_A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84],
]
_DP_B5 = np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
_DP_B4 = np.array([5179/57600, 0.0, 7571/16695, 393/640,
                   -92097/339200, 187/2100, 1/40])

_TABLEAUX = {
    "rkf45":  (_RKF_C, _RKF_A, _RKF_B5, _RKF_B4, 4),
    "dopri5": (_DP_C, _DP_A, _DP_B5, _DP_B4, 4),
}


class Solution:
    """Container for an integrated trajectory (uniformly resampled)."""

    def __init__(self, t, y, n_steps, n_rejected, method):
        self.t = t                    # (N,)   uniform time grid
        self.y = y                    # (N, d) states
        self.n_steps = n_steps        # accepted steps
        self.n_rejected = n_rejected  # rejected steps
        self.method = method

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])

    def __repr__(self):
        return (f"<Solution method={self.method} N={len(self.t)} "
                f"dim={self.y.shape[1]} steps={self.n_steps} "
                f"rejected={self.n_rejected}>")


def integrate(f, t_span, y0, *, method="rkf45", h0=1e-3, rtol=1e-8,
              atol=1e-10, dt_out=0.01, args=(), max_steps=20_000_000,
              h_min=1e-10, h_max=None):
    """
    Adaptive embedded Runge-Kutta integration with PI step control.

    Parameters
    ----------
    f        : callable  ``f(t, y, *args) -> ndarray``
    t_span   : (t0, tf)
    y0       : initial state (1-D array-like)
    method   : "rkf45" | "dopri5"
    h0       : initial step
    rtol,atol: relative / absolute tolerances for the embedded error norm
    dt_out   : spacing of the uniformly-resampled output grid
    args     : extra arguments forwarded to ``f``

    Returns
    -------
    Solution
    """
    if method not in _TABLEAUX:
        raise ValueError(f"unknown method {method!r}")
    C, A, B_hi, B_lo, order = _TABLEAUX[method]
    E = B_hi - B_lo                       # error weights
    s = len(C)

    t0, tf = float(t_span[0]), float(t_span[1])
    y = np.asarray(y0, dtype=float).copy()
    d = y.size
    t = t0
    h = float(h0)
    if h_max is None:
        h_max = (tf - t0) / 10.0

    # storage of raw (non-uniform) nodes, densified later
    ts = [t0]
    ys = [y.copy()]

    n_acc = 0
    n_rej = 0
    err_prev = 1.0
    safety, min_fac, max_fac = 0.9, 0.2, 5.0

    while t < tf - 1e-14 and n_acc < max_steps:
        if t + h > tf:
            h = tf - t

        # ---- stages -------------------------------------------------------
        k = np.empty((s, d))
        k[0] = f(t, y, *args)
        for i in range(1, s):
            yi = y.copy()
            ai = A[i]
            for j in range(i):
                if ai[j] != 0.0:
                    yi += h * ai[j] * k[j]
            k[i] = f(t + C[i] * h, yi, *args)

        y_hi = y + h * (B_hi @ k)
        err_vec = h * (E @ k)

        # ---- error norm ---------------------------------------------------
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_hi))
        err = np.sqrt(np.mean((err_vec / scale) ** 2))

        if err <= 1.0 or h <= h_min:
            # accept
            t += h
            y = y_hi
            ts.append(t)
            ys.append(y.copy())
            n_acc += 1
            # PI controller
            if err == 0.0:
                fac = max_fac
            else:
                fac = safety * err ** (-0.7 / (order + 1)) * err_prev ** (0.4 / (order + 1))
                fac = min(max_fac, max(min_fac, fac))
            err_prev = max(err, 1e-4)
            h = min(h * fac, h_max)
        else:
            # reject, shrink
            n_rej += 1
            fac = max(min_fac, safety * err ** (-1.0 / (order + 1)))
            h *= fac
            if h < h_min:
                h = h_min

    ts = np.asarray(ts)
    ys = np.asarray(ys)

    # ---- dense uniform resampling (cubic Hermite, 4th-order accurate) ------
    # Linear interpolation between accepted nodes would inject an O(h^2) error
    # onto the OUTPUT grid even though the nodes themselves are accurate to the
    # requested tolerance. We instead use a piecewise-cubic Hermite that matches
    # both the state and its derivative f(t, y) at every node, so the resampled
    # trajectory keeps the integrator's accuracy (verified against SciPy).
    fs = np.empty_like(ys)
    for i in range(len(ts)):
        fs[i] = f(ts[i], ys[i], *args)

    n_out = int(np.floor((tf - t0) / dt_out)) + 1
    t_uniform = t0 + dt_out * np.arange(n_out)
    y_uniform = _hermite_resample(ts, ys, fs, t_uniform)

    return Solution(t_uniform, y_uniform, n_acc, n_rej, method)


def _hermite_resample(ts, ys, fs, t_query):
    """Vectorised piecewise cubic-Hermite interpolation of ``ys`` (with node
    derivatives ``fs``) onto ``t_query``. Exact at nodes; 4th-order between."""
    if len(ts) == 1:
        return np.repeat(ys, len(t_query), axis=0)
    idx = np.searchsorted(ts, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(ts) - 2)
    t_l = ts[idx]
    h = ts[idx + 1] - t_l
    s = (t_query - t_l) / h
    s2 = s * s
    s3 = s2 * s
    h00 = (2 * s3 - 3 * s2 + 1)[:, None]
    h10 = (s3 - 2 * s2 + s)[:, None]
    h01 = (-2 * s3 + 3 * s2)[:, None]
    h11 = (s3 - s2)[:, None]
    hcol = h[:, None]
    return (h00 * ys[idx] + h10 * hcol * fs[idx]
            + h01 * ys[idx + 1] + h11 * hcol * fs[idx + 1])
