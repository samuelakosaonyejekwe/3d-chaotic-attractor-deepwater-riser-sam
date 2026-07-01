"""
chronos.metocean
================
Site-specific environmental (metocean) input for a case study.

Provides a credible, fully-specified deepwater environment: water depth,
current profile, wave scatter (H_s, T_p), and a helper that converts a current
speed into the VIV reduced velocity / expected lock-in state for a given riser.

The default site models a **deepwater Gulf-of-Guinea (offshore West Africa)**
FPSP flowline/riser location; all numbers are within published ranges for that
province (see case documentation for sourcing). Values are *dummy but credible*
site data intended for calibration, as requested.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class MetoceanSite:
    name: str = "Deepwater Gulf of Guinea (offshore West Africa)"
    water_depth: float = 1500.0        # m
    seawater_density: float = 1025.0   # kg/m3
    kinematic_viscosity: float = 1.05e-6  # m2/s
    # 1-year / 10-year / 100-year associated current at surface (m/s)
    current_1yr: float = 1.05
    current_10yr: float = 1.35
    current_100yr: float = 1.62
    # surface current profile shear exponent (power law to seabed)
    current_shear_exp: float = 0.14
    # sea-state scatter (representative operational states)
    Hs_operational: float = 2.5        # m significant wave height
    Tp_operational: float = 9.5        # s peak period
    Hs_100yr: float = 6.8              # m
    Tp_100yr: float = 14.5            # s

    def current_profile(self, n=25):
        """Power-law current speed vs depth (surface->seabed)."""
        z = np.linspace(0.0, self.water_depth, n)          # depth below surface
        frac = np.clip(1.0 - z/self.water_depth, 0.0, 1.0)
        u = self.current_100yr * frac**self.current_shear_exp
        u[-1] = 0.0
        return z, u

    def scatter_diagram(self):
        """A small, credible Hs-Tp-probability scatter table (annual)."""
        # (Hs, Tp, annual probability of occurrence)
        rows = [
            (0.5,  6.0, 0.14),
            (1.0,  7.0, 0.19),
            (1.5,  8.0, 0.20),
            (2.0,  9.0, 0.16),
            (2.5,  9.5, 0.12),
            (3.0, 10.5, 0.08),
            (3.5, 11.5, 0.05),
            (4.5, 12.5, 0.03),
            (5.5, 13.5, 0.02),
            (6.8, 14.5, 0.01),
        ]
        return rows

    def reduced_velocity(self, U, fn_hz, D):
        """VIV reduced velocity  Ur = U / (fn * D)."""
        return U / (fn_hz * D)

    def reynolds(self, U, D):
        return U * D / self.kinematic_viscosity

    def strouhal_frequency(self, U, D, St=0.18):
        return St * U / D

    def summary(self):
        d = asdict(self)
        return d
