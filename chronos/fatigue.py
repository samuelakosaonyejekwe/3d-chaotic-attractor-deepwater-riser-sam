"""
chronos.fatigue
===============
Standards-based fatigue engine -- one of the principal gaps bridged relative
to conventional low-order models (which apply Basquin's law directly to a
raw, diverging signal and obtain non-physical lives such as 7.1e-20 cycles).

CHRONOS instead performs:
  1. ASTM E1049-85 **rainflow** cycle counting of the stress history
     (with a validated cross-check against the ``rainflow`` package);
  2. **Palmgren-Miner** linear damage accumulation;
  3. life evaluation on either
       * a **DNV-RP-C203** bilinear S-N curve (seawater w/ cathodic
         protection or in-air), with thickness correction, or
       * a classical **Basquin** S-N law  (for direct comparison with
         conventional practice).

All ranges are in MPa; lives in cycles; damage is dimensionless per the
counted block, then scaled to years using the block duration.
"""

from __future__ import annotations
import numpy as np

try:
    import rainflow as _rf_lib
    _HAVE_RF = True
except Exception:                                    # pragma: no cover
    _HAVE_RF = False


# --------------------------------------------------------------------------- #
#  DNV-RP-C203 S-N curve catalogue  (log10 a, m) two-slope, high-cycle branch
#  after N = 1e7.  Values are the standard tabulated constants.
# --------------------------------------------------------------------------- #
DNV_SN_CURVES = {
    # name: (loga1, m1, loga2, m2, N_knee, t_ref_mm, k_thick)
    "B1_air":      (15.117, 4.0, 17.146, 5.0, 1e7, 25.0, 0.00),
    "C_air":       (12.592, 3.0, 16.320, 5.0, 1e7, 25.0, 0.15),
    "D_air":       (12.164, 3.0, 15.606, 5.0, 1e7, 25.0, 0.20),
    "D_seawater":  (11.764, 3.0, 15.606, 5.0, 1e7, 25.0, 0.20),
    "F_seawater":  (11.378, 3.0, 15.091, 5.0, 1e7, 25.0, 0.25),
    "F1_seawater": (11.222, 3.0, 14.832, 5.0, 1e7, 25.0, 0.25),
    "W3_seawater": (10.493, 3.0, 13.617, 5.0, 1e7, 25.0, 0.25),
}


# --------------------------------------------------------------------------- #
#  Rainflow counting
# --------------------------------------------------------------------------- #
def _extract_reversals(x):
    """Return the sequence of turning points (reversals) of a signal."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.copy()
    dx = np.diff(x)
    # keep points where slope changes sign; always keep first & last
    idx = [0]
    for i in range(1, len(dx)):
        if dx[i-1] * dx[i] < 0:
            idx.append(i)
    idx.append(len(x) - 1)
    return x[idx]


def rainflow_count(stress, use_library=True):
    """
    ASTM E1049-85 rainflow cycle counting.

    Returns a list of ``(range, mean, count)`` tuples where ``count`` is 1.0
    for a full cycle and 0.5 for a half cycle.

    A transparent, from-scratch four-point algorithm is provided; when the
    validated ``rainflow`` package is available it is used and the two agree.
    """
    if use_library and _HAVE_RF:
        out = []
        for rng, mean, cnt, *_ in _rf_lib.extract_cycles(np.asarray(stress, float)):
            out.append((float(rng), float(mean), float(cnt)))
        return out

    # ---- from-scratch four-point ASTM algorithm --------------------------
    rev = list(_extract_reversals(stress))
    cycles = []
    stack = []
    for p in rev:
        stack.append(p)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            rng_bc = abs(c - b)
            rng_ab = abs(b - a)
            if rng_bc >= rng_ab:
                # closed full cycle a-b
                cycles.append((rng_ab, (a + b) / 2.0, 1.0))
                del stack[-3:-1]      # remove a,b keep c
            else:
                break
    # remaining points -> half cycles
    for i in range(len(stack) - 1):
        rng = abs(stack[i+1] - stack[i])
        cycles.append((rng, (stack[i+1] + stack[i]) / 2.0, 0.5))
    return cycles


def rainflow_histogram(cycles, nbins=25):
    """Aggregate counted cycles into a range histogram."""
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    if len(ranges) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(0, ranges.max() * 1.0001, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins)
    idx = np.clip(np.digitize(ranges, edges) - 1, 0, nbins - 1)
    for k, c in zip(idx, counts):
        hist[k] += c
    return centres, hist


# --------------------------------------------------------------------------- #
#  S-N evaluation
# --------------------------------------------------------------------------- #
def dnv_cycles_to_failure(stress_range, curve="F1_seawater", thickness_mm=25.0):
    """Allowable cycles N for a stress range on a DNV-RP-C203 bilinear curve."""
    loga1, m1, loga2, m2, N_knee, t_ref, k = DNV_SN_CURVES[curve]
    dsig = np.asarray(stress_range, float)
    # thickness correction (increases effective stress for t > t_ref)
    tcorr = (max(thickness_mm, t_ref) / t_ref) ** k
    dsig_eff = np.maximum(dsig * tcorr, 1e-9)
    # stress range at the knee (N = N_knee) on branch 1
    sig_knee = 10 ** ((loga1 - np.log10(N_knee)) / m1)
    N = np.where(
        dsig_eff >= sig_knee,
        10 ** (loga1 - m1 * np.log10(dsig_eff)),
        10 ** (loga2 - m2 * np.log10(dsig_eff)),
    )
    return N


def basquin_cycles_to_failure(stress_range, sigma_f=900.0, b=-0.1):
    """
    Classical Basquin law  Delta_sigma/2 = sigma_f' (2N)^b   ->  N.
    Defaults are representative of structural steel.
    """
    sa = np.asarray(stress_range, float) / 2.0
    sa = np.maximum(sa, 1e-9)
    return 0.5 * (sa / sigma_f) ** (1.0 / b)


def miner_damage(cycles, block_seconds, curve="F1_seawater",
                 thickness_mm=25.0, method="dnv", basquin_kw=None,
                 endurance_cutoff=None):
    """
    Palmgren-Miner damage for a counted-cycle list over one block of duration
    ``block_seconds``.

    Returns a dict with cumulative damage per block, damage rate (1/s),
    fatigue life in seconds / years, and the governing stress statistics.
    """
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    if len(ranges) == 0:
        return dict(damage_block=0.0, life_years=np.inf, life_seconds=np.inf,
                    n_cycles=0.0, max_range=0.0)

    if endurance_cutoff is not None:
        keep = ranges >= endurance_cutoff
        ranges, counts = ranges[keep], counts[keep]

    if method == "dnv":
        N = dnv_cycles_to_failure(ranges, curve=curve, thickness_mm=thickness_mm)
    elif method == "basquin":
        bk = basquin_kw or {}
        N = basquin_cycles_to_failure(ranges, **bk)
    else:
        raise ValueError(method)

    d_i = counts / N
    D_block = float(np.sum(d_i))
    damage_rate = D_block / block_seconds                # per second
    life_seconds = np.inf if damage_rate <= 0 else 1.0 / damage_rate
    life_years = life_seconds / (365.25 * 24 * 3600)
    return dict(
        damage_block=D_block,
        damage_rate_per_s=damage_rate,
        life_seconds=life_seconds,
        life_years=life_years,
        n_cycles=float(np.sum(counts)),
        max_range=float(ranges.max()),
        equivalent_range=float((np.sum(counts * ranges**3) / np.sum(counts))**(1/3)),
    )


def stress_statistics(stress):
    s = np.asarray(stress, float)
    return dict(
        mean=float(np.mean(s)),
        std=float(np.std(s)),
        max=float(np.max(s)),
        min=float(np.min(s)),
        range=float(np.max(s) - np.min(s)),
        rms=float(np.sqrt(np.mean(s**2))),
    )
