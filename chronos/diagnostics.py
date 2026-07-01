"""
chronos.diagnostics
====================
Nonlinear time-series & dynamical-systems diagnostics used across the CHRONOS
post-processing suite.

Contents
--------
lyapunov_spectrum      full spectrum via Benettin/Gram-Schmidt of the
                       variational flow (validated against Lorenz).
kaplan_yorke_dimension Lyapunov (KY) fractal dimension from a spectrum.
poincare_section       plane-crossing Poincare map (interpolated crossings).
power_spectral_density Welch PSD.
fft_spectrum           single-sided amplitude spectrum.
dominant_frequency     peak-picking on the PSD.
wavelet_transform      continuous wavelet transform (Morlet) scaleogram.
recurrence_matrix      recurrence plot / recurrence-quantification helpers.
return_map             first-return map (x_n vs x_{n+1} at extrema).
bifurcation_diagram    parameter sweep of a scalar observable's extrema.
correlation_dimension  Grassberger-Procaccia D2.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
import pywt


# --------------------------------------------------------------------------- #
#  Lyapunov spectrum  (Benettin et al. 1980 ; Wolf et al. 1985)
# --------------------------------------------------------------------------- #
def lyapunov_spectrum(system, y0, t_max=200.0, dt=0.01, t_transient=20.0,
                      renorm_every=1):
    """
    Full Lyapunov spectrum of ``system`` (a :class:`chronos.dynsys.DynamicalSystem`)
    by simultaneous integration of the state and an orthonormal perturbation
    frame, with periodic Gram-Schmidt (QR) re-orthonormalisation.

    A simple fixed-step RK4 is used for both the nonlinear flow and the
    linearised (variational) flow so the two stay perfectly synchronised.

    Returns
    -------
    exponents : ndarray (dim,)   sorted descending
    history   : ndarray (M, dim) running-average convergence history
    """
    d = system.dim
    y = np.asarray(y0, float).copy()
    Q = np.eye(d)
    n = int(round(t_max / dt))
    n_trans = int(round(t_transient / dt))

    def flow(t, yy):
        return system.f(t, yy)

    def var_flow(t, yy, V):
        return system.jacobian(t, yy) @ V

    lyap_sum = np.zeros(d)
    hist = []
    t = 0.0
    count = 0
    for step in range(n):
        # --- RK4 on state + variational frame (matrix) ---
        k1 = flow(t, y);            K1 = var_flow(t, y, Q)
        k2 = flow(t + dt/2, y + dt/2*k1); K2 = var_flow(t + dt/2, y + dt/2*k1, Q + dt/2*K1)
        k3 = flow(t + dt/2, y + dt/2*k2); K3 = var_flow(t + dt/2, y + dt/2*k2, Q + dt/2*K2)
        k4 = flow(t + dt, y + dt*k3);     K4 = var_flow(t + dt, y + dt*k3, Q + dt*K3)
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        Q = Q + dt/6*(K1 + 2*K2 + 2*K3 + K4)
        t += dt

        if (step % renorm_every) == 0:
            Q, R = np.linalg.qr(Q)
            # fix signs so diagonal of R is positive
            sgn = np.sign(np.diag(R))
            sgn[sgn == 0] = 1.0
            Q *= sgn
            diagR = np.abs(np.diag(R))
            diagR = np.where(diagR < 1e-300, 1e-300, diagR)
            if step >= n_trans:
                lyap_sum += np.log(diagR)
                count += 1
                if count % 50 == 0:
                    hist.append(lyap_sum / (count * dt * renorm_every))

    exponents = lyap_sum / (count * dt * renorm_every)
    exponents = np.sort(exponents)[::-1]
    return exponents, np.array(hist)


def kaplan_yorke_dimension(exponents):
    """Lyapunov (Kaplan-Yorke) dimension from a descending exponent spectrum."""
    lam = np.sort(np.asarray(exponents, float))[::-1]
    cumulative = np.cumsum(lam)
    k = 0
    for i in range(len(lam)):
        if cumulative[i] >= 0:
            k = i + 1
        else:
            break
    if k == 0:
        return 0.0
    if k >= len(lam):
        return float(len(lam))
    return k + cumulative[k-1] / abs(lam[k])


# --------------------------------------------------------------------------- #
#  Poincare section
# --------------------------------------------------------------------------- #
def poincare_section(t, Y, plane_index=2, plane_value=None, direction=1):
    """
    Crossings of the trajectory ``Y`` (N, d) through the hyperplane
    ``Y[:, plane_index] = plane_value`` in the given ``direction`` (+1 upward).
    Crossings are linearly interpolated. Returns the interpolated crossing
    states (M, d) and their times (M,).
    """
    Y = np.asarray(Y)
    s = Y[:, plane_index]
    if plane_value is None:
        plane_value = float(np.mean(s))
    g = s - plane_value
    cross = []
    tc = []
    for i in range(len(g) - 1):
        if g[i] == 0.0:
            continue
        if (g[i] < 0 < g[i+1] and direction >= 0) or \
           (g[i] > 0 > g[i+1] and direction <= 0):
            frac = g[i] / (g[i] - g[i+1])
            cross.append(Y[i] + frac * (Y[i+1] - Y[i]))
            tc.append(t[i] + frac * (t[i+1] - t[i]))
    if not cross:
        return np.empty((0, Y.shape[1])), np.empty(0)
    return np.asarray(cross), np.asarray(tc)


# --------------------------------------------------------------------------- #
#  Spectral analysis
# --------------------------------------------------------------------------- #
def power_spectral_density(x, fs, nperseg=None):
    x = np.asarray(x, float)
    if nperseg is None:
        nperseg = min(len(x), 4096)
    f, pxx = signal.welch(x - np.mean(x), fs=fs, nperseg=nperseg,
                          scaling="density")
    return f, pxx


def fft_spectrum(x, fs):
    """Single-sided amplitude spectrum."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    amp = (2.0 / np.sum(win)) * np.abs(X)
    return f, amp


def dominant_frequency(x, fs, fmin=1e-3):
    f, p = power_spectral_density(x, fs)
    mask = f > fmin
    if not np.any(mask):
        return 0.0
    fpk = f[mask][np.argmax(p[mask])]
    return float(fpk)


# --------------------------------------------------------------------------- #
#  Continuous wavelet transform (scaleogram)
# --------------------------------------------------------------------------- #
def wavelet_transform(x, fs, wavelet="cmor1.5-1.0", n_scales=128,
                      f_min=0.05, f_max=None):
    x = np.asarray(x, float) - np.mean(x)
    if f_max is None:
        f_max = fs / 2.0
    fc = pwt_center = pywt.central_frequency(wavelet)
    freqs = np.linspace(f_min, f_max, n_scales)
    scales = fc * fs / freqs
    coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0/fs)
    power = np.abs(coeffs)
    return freqs, power   # (n_scales,), (n_scales, N)


# --------------------------------------------------------------------------- #
#  Recurrence plot & return map
# --------------------------------------------------------------------------- #
def _embedding_delay(x, max_lag=200):
    """Estimate a time-delay as the first zero-crossing of the autocorrelation."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    max_lag = min(max_lag, n // 4)
    ac = np.correlate(x, x, mode="full")[n-1:n-1+max_lag]
    ac = ac / (ac[0] + 1e-30)
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            return max(1, k)
    return max(1, int(np.argmin(ac)))


def recurrence_matrix(x, target_rate=0.15, max_points=700, emb_dim=3,
                      delay=None):
    """
    Recurrence matrix of a scalar series using Takens time-delay embedding.

    The signal is embedded into ``emb_dim`` dimensions with lag ``delay``
    (auto-estimated from the autocorrelation), distances are computed between
    embedded state vectors, and the threshold ``eps`` is chosen so the overall
    recurrence rate equals ``target_rate`` -- giving a clean, scale-independent
    recurrence plot with the characteristic diagonal-line texture of a
    deterministic (chaotic) system rather than a grainy scalar-coincidence field.
    """
    x = np.asarray(x, float)
    # pre-decimate long signals so autocorrelation / embedding stay cheap
    target_len = max(max_points * 3, 2500)
    if len(x) > target_len:
        x = x[np.linspace(0, len(x)-1, target_len).astype(int)]
    if delay is None:
        delay = _embedding_delay(x, max_lag=target_len // 8)
    # build embedded vectors
    span = (emb_dim - 1) * delay
    N = len(x) - span
    if N < 20:                              # fall back to scalar if too short
        emb = x.reshape(-1, 1)
    else:
        emb = np.column_stack([x[i*delay:i*delay + N] for i in range(emb_dim)])
    if len(emb) > max_points:
        idx = np.linspace(0, len(emb)-1, max_points).astype(int)
        emb = emb[idx]
    diff = emb[:, None, :] - emb[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    eps = np.quantile(D[np.triu_indices(len(D), 1)], target_rate)
    return (D <= eps).astype(float), D


def recurrence_rate(R):
    return float(np.mean(R))


def return_map(x, fs=None):
    """First-return map: local maxima x_n vs next local maximum x_{n+1}."""
    x = np.asarray(x, float)
    peaks, _ = signal.find_peaks(x)
    if len(peaks) < 3:
        # fall back to plain successive samples
        return x[:-1], x[1:]
    pk = x[peaks]
    return pk[:-1], pk[1:]


# --------------------------------------------------------------------------- #
#  Bifurcation diagram
# --------------------------------------------------------------------------- #
def bifurcation_diagram(make_system, integrate_fn, param_values, y0,
                        observable_index=1, t_max=120.0, t_keep=40.0,
                        dt_out=0.01):
    """
    Sweep ``param_values``; for each build a system via ``make_system(p)``,
    integrate, and record the local extrema of the chosen observable over the
    final ``t_keep`` seconds.

    Returns
    -------
    P : ndarray  parameter value repeated per recorded point
    V : ndarray  observable extrema
    """
    P, V = [], []
    for p in param_values:
        sysf = make_system(p)
        sol = integrate_fn(sysf, y0, t_max=t_max, dt_out=dt_out)
        t, Y = sol.t, sol.y
        keep = t >= (t_max - t_keep)
        x = Y[keep, observable_index]
        mx, _ = signal.find_peaks(x)
        mn, _ = signal.find_peaks(-x)
        ex = np.concatenate([x[mx], x[mn]]) if (len(mx)+len(mn)) else x[::25]
        P.extend([p] * len(ex))
        V.extend(ex.tolist())
    return np.asarray(P), np.asarray(V)


# --------------------------------------------------------------------------- #
#  Correlation dimension (Grassberger-Procaccia)
# --------------------------------------------------------------------------- #
def correlation_dimension(Y, max_points=2000, n_r=20):
    Y = np.asarray(Y, float)
    if len(Y) > max_points:
        idx = np.linspace(0, len(Y)-1, max_points).astype(int)
        Y = Y[idx]
    # pairwise distances
    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    iu = np.triu_indices(len(Y), k=1)
    d = D[iu]
    d = d[d > 0]
    rmin, rmax = np.percentile(d, 1), np.percentile(d, 50)
    rs = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([np.mean(d < r) for r in rs])
    good = C > 0
    lr, lc = np.log(rs[good]), np.log(C[good])
    if len(lr) < 3:
        return np.nan, rs, C
    slope = np.polyfit(lr, lc, 1)[0]
    return float(slope), rs, C
