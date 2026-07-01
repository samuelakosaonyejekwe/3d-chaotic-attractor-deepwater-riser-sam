# CHRONOS — 3D Chaotic Attractor for Deepwater-Riser VIV & Fatigue

**Chaotic Hydro-Resonant Oscillation & Nonlinear-fatigue Operational Solver**

A novel, universal Python solver for nonlinear / chaotic dynamical systems, paired
with a standards-based engineering post-processing and fatigue suite. It was
developed for a deepwater marine-riser **vortex-induced vibration (VIV) fatigue**
case study by **Akosa Samuel Onyejekwe**.

---

## Motivation — the gap this bridges

Conventional low-order chaotic VIV attractors **diverge**: they blow up to
~1×10⁶–1×10⁸ within roughly 8 s of simulated time and return physically
meaningless results (e.g. axial stress −853 MPa, torsional stress 4.8×10⁸ MPa,
fatigue life 7.1×10⁻²⁰ cycles).

CHRONOS replaces that behaviour with a physically-grounded, **provably bounded**
model and a rigorous, validated toolchain, so that every stress, spectrum, and
fatigue estimate remains finite and physically interpretable for all parameter
choices.

---

## The novel model — BCWSA

The **Bounded Chaotic Wake–Structure Attractor** (7 states) couples three
sub-systems:

- a **Duffing** structural oscillator (cross-flow displacement);
- a **Facchinetti–de Langre–Biolley (2004)** acceleration-coupled **van der Pol
  wake oscillator**, which reproduces VIV lock-in at ~1-diameter amplitude; and
- a **Lorenz-type near-wake convective core**, which guarantees sustained but
  bounded chaos.

Coupling between the sub-systems is **two-way and `tanh`-saturated (bounded)**, and
an explicit dissipative **trapping-region argument** guarantees finite stresses for
all parameters.

---

## Repository layout

```
chronos/                     Core solver + analysis library (pure Python)
  integrators.py             RKF45 + Dormand–Prince (from scratch, adaptive step)
  dynsys.py                  Universal DynamicalSystem (analytic / finite-diff Jacobian)
  diagnostics.py             Lyapunov spectrum (Benettin/QR), Kaplan–Yorke &
                             correlation dimensions, Poincaré sections, PSD/FFT,
                             wavelet, recurrence plots, return maps, bifurcation
  fatigue.py                 ASTM E1049 rainflow + Palmgren–Miner + DNV-RP-C203 + Basquin
  models.py                  Lorenz, conventional low-order attractor,
                             RiserProperties, BCWSA
  metocean.py                Site-specific environmental (metocean) data
  plots.py                   Figure generation + LaTeX equation rendering
  report.py                  Structured report builder
  validate.py                Lorenz + VIV lock-in validation (with sources)

outputs/
  data/                      Every metric, table, curve and contour as *.csv
  figures/                   Publication-quality PNG figures
  figures/csv_plots/         One plot per exported CSV
  metrics_summary.json       Machine-readable summary of headline results
  run.log                    Execution log of the case-study run

case.report.pdf              Full written technical report of the case study
```

---

## The case study

The `outputs/` directory contains the complete, reproducible results of the
deepwater-riser VIV-fatigue case study:

- **State & attractor** — time series, 3-D attractor and 2-D projections,
  Poincaré sections, return maps.
- **Spectral** — axial / bending / torsional PSD, bending FFT, wavelet analysis.
- **Chaos diagnostics** — Lyapunov-exponent convergence, recurrence plots,
  bifurcation diagrams, stability maps.
- **Engineering** — physical stress time series, rainflow histograms, S–N curves,
  critical-velocity and sensitivity studies, long-term stress, material
  comparison, energy dissipation, and before/after optimisation.

Every metric, table, curve and contour is exported to `outputs/data/*.csv`, each
CSV is plotted under `outputs/figures/csv_plots/`, and the full narrative is given
in **`case.report.pdf`**.

---

## Validation

Sources are recorded alongside the validation outputs.

- **Lorenz spectrum** (0.906, 0, −14.572), Kaplan–Yorke dimension **2.062** —
  Sprott; Wolf et al. (1985).
- **VIV lock-in** peak **A/D ≈ 1.0** — Khalak & Williamson (1999).
- **Wake model** — Facchinetti, de Langre & Biolley (2004).
- **Fatigue** — DNV-RP-C203.

---

## Standards & methods

| Domain            | Basis                                              |
|-------------------|----------------------------------------------------|
| Cycle counting    | ASTM E1049 rainflow                                |
| Damage accumulation | Palmgren–Miner linear rule                        |
| Fatigue curves    | DNV-RP-C203, Basquin S–N                            |
| Wake dynamics     | Facchinetti–de Langre–Biolley van der Pol wake     |
| Chaos metrics     | Benettin/QR Lyapunov spectrum, Kaplan–Yorke dim    |
| Integration       | Adaptive RKF45 / Dormand–Prince (implemented from scratch) |

---

## Requirements

- Python 3.12+
- NumPy, SciPy, Matplotlib

The `chronos` package is pure Python and can be imported directly for use with any
dynamical system:

```python
from chronos.models import BCWSA
from chronos.dynsys import DynamicalSystem
from chronos.diagnostics import lyapunov_spectrum
```

---

## Author

**Akosa Samuel Onyejekwe**

---

*Research code. Provided as-is for scientific and engineering evaluation.*
