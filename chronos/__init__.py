"""
CHRONOS
=======
Chaotic Hydro-Resonant Oscillation & Nonlinear-fatigue Operational Solver.

A *universal* solver for nonlinear / chaotic dynamical systems with an
engineering post-processing suite (phase-space, spectral, wavelet, recurrence,
Lyapunov, bifurcation) and a physically-grounded fatigue engine
(ASTM E1049 rainflow + Palmgren-Miner + DNV-RP-C203 S-N + Basquin).

The package ships with:
  * a from-scratch adaptive Runge-Kutta-Fehlberg (RKF45) integrator and a
    Dormand-Prince (DOPRI5) integrator  (``chronos.integrators``);
  * a generic ``DynamicalSystem`` container with finite-difference Jacobians
    (``chronos.dynsys``);
  * a full diagnostics library (``chronos.diagnostics``);
  * a standards-based fatigue module (``chronos.fatigue``);
  * a library of reference systems and the novel *Bounded Chaotic
    Wake-Structure Attractor* (BCWSA) for deepwater risers
    (``chronos.models``).

Author  : CHRONOS engine (A. S. Onyejekwe)
Licence : research / evaluation
"""

__version__ = "1.0.0"
__all__ = [
    "integrators",
    "dynsys",
    "diagnostics",
    "fatigue",
    "models",
    "metocean",
    "plots",
    "report",
]
