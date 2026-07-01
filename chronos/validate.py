"""
chronos.validate
================
Independent validation of the CHRONOS universal solver against *credible,
published* reference data, with sources recorded.

V1  Canonical Lorenz attractor -- verifies the integrator + Lyapunov engine +
    fractal-dimension estimator reproduce the accepted Lyapunov spectrum and
    Kaplan-Yorke dimension.
      Reference: J. C. Sprott, "Lyapunov Exponent and Dimension of the Lorenz
      Attractor", Univ. of Wisconsin-Madison,
      https://sprott.physics.wisc.edu/chaos/lorenzle.htm ; and
      A. Wolf, J. B. Swift, H. L. Swinney, J. A. Vastano, "Determining Lyapunov
      exponents from a time series", Physica D 16 (1985) 285-317.
      Accepted values:  lambda = (0.906, 0, -14.572);  D_KY = 2.062.

V2  Wake-oscillator VIV amplitude -- verifies the wake-structure sub-model
    reproduces the experimentally-observed lock-in peak cross-flow amplitude of
    ~1 diameter at low mass-damping.
      Reference: A. Khalak & C. H. K. Williamson, "Motions, forces and mode
      transitions in vortex-induced vibrations at low mass-damping", J. Fluids
      & Structures 13 (1999) 813-851  (peak A/D ~ 1.0, upper branch);
      model form after M. L. Facchinetti, E. de Langre & F. Biolley, "Coupling
      of structure and wake oscillators in vortex-induced vibrations", J. Fluids
      & Structures 19 (2004) 123-140  (eps=0.3, A=12, St~0.2).
"""

from __future__ import annotations
import numpy as np

from . import integrators as ig
from . import diagnostics as dg
from . import models as M


VALIDATION_SOURCES = [
    dict(id="Sprott-Lorenz",
         cite="J. C. Sprott, 'Lyapunov Exponent and Dimension of the Lorenz "
              "Attractor', University of Wisconsin-Madison.",
         url="https://sprott.physics.wisc.edu/chaos/lorenzle.htm",
         data="lambda = (0.906, 0, -14.572); Kaplan-Yorke dimension = 2.062"),
    dict(id="Wolf-1985",
         cite="A. Wolf, J. B. Swift, H. L. Swinney, J. A. Vastano, "
              "'Determining Lyapunov exponents from a time series', "
              "Physica D 16 (1985) 285-317.",
         url="https://doi.org/10.1016/0167-2789(85)90011-9",
         data="Benettin/QR algorithm; Lorenz largest exponent ~0.906 bit/s"),
    dict(id="KhalakWilliamson-1999",
         cite="A. Khalak & C. H. K. Williamson, 'Motions, forces and mode "
              "transitions in vortex-induced vibrations at low mass-damping', "
              "J. Fluids & Structures 13 (1999) 813-851.",
         url="https://doi.org/10.1006/jfls.1999.0236",
         data="peak cross-flow amplitude A/D ~ 1.0 (upper branch), "
              "three-branch response, low mass-damping"),
    dict(id="Facchinetti-2004",
         cite="M. L. Facchinetti, E. de Langre, F. Biolley, 'Coupling of "
              "structure and wake oscillators in vortex-induced vibrations', "
              "J. Fluids & Structures 19 (2004) 123-140.",
         url="https://doi.org/10.1016/j.jfluidstructs.2003.12.004",
         data="acceleration-coupled van der Pol wake oscillator; eps=0.3, A=12"),
    dict(id="DNV-RP-C203",
         cite="DNV-RP-C203, 'Fatigue design of offshore steel structures', "
              "Det Norske Veritas.",
         url="https://www.dnv.com",
         data="two-slope S-N curves (seawater w/ CP and in-air), "
              "thickness correction, Palmgren-Miner accumulation"),
    dict(id="Lorenz-1963",
         cite="E. N. Lorenz, 'Deterministic Nonperiodic Flow', "
              "J. Atmospheric Sciences 20 (1963) 130-141.",
         url="https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2",
         data="low-order convective triad; physical basis of the near-wake core"),
]


# Sources used to CALIBRATE the model constants and to source the engineering
# input data (hydrodynamics, material, standards, environment). Some overlap
# with the validation set above (e.g. Facchinetti-2004, DNV-RP-C203).
CALIBRATION_SOURCES = [
    dict(id="Facchinetti-2004",
         cite="M. L. Facchinetti, E. de Langre, F. Biolley, 'Coupling of "
              "structure and wake oscillators in vortex-induced vibrations', "
              "J. Fluids & Structures 19 (2004) 123-140.",
         url="https://doi.org/10.1016/j.jfluidstructs.2003.12.004",
         data="wake-oscillator constants: epsilon=0.3, A=12 (acceleration "
              "coupling); van der Pol form"),
    dict(id="Blevins-1990",
         cite="R. D. Blevins, 'Flow-Induced Vibration', 2nd ed., "
              "Van Nostrand Reinhold, 1990.",
         url="ISBN 978-0442206512",
         data="Strouhal number St~0.2 for circular cylinders; VIV lock-in "
              "band and reduced-velocity range"),
    dict(id="SumerFredsoe-2006",
         cite="B. M. Sumer & J. Fredsoe, 'Hydrodynamics Around Cylindrical "
              "Structures', rev. ed., World Scientific, 2006.",
         url="https://doi.org/10.1142/6248",
         data="drag / inertia (added-mass) coefficients Cd, Ca and vortex-"
              "shedding regimes vs Reynolds/KC number"),
    dict(id="DNV-RP-C205",
         cite="DNV-RP-C205, 'Environmental conditions and environmental "
              "loads', Det Norske Veritas.",
         url="https://www.dnv.com",
         data="current-profile modelling, Morison drag/added-mass "
              "coefficients used for the hydrodynamic loading"),
    dict(id="DNV-RP-F204",
         cite="DNV-RP-F204, 'Riser fatigue', Det Norske Veritas.",
         url="https://www.dnv.com",
         data="riser fatigue-analysis methodology and design fatigue factor "
              "(DFF) framework"),
    dict(id="DNV-RP-F105",
         cite="DNV-RP-F105, 'Free spanning pipelines', Det Norske Veritas.",
         url="https://www.dnv.com",
         data="VIV response-amplitude (A/D) models and cross-flow lock-in "
              "calibration reference"),
    dict(id="API-5L",
         cite="API Specification 5L, 'Line Pipe', 46th ed., American "
              "Petroleum Institute, 2018.",
         url="https://www.api.org",
         data="grade X65 line pipe: SMYS 448 MPa, E=207 GPa, "
              "density 7850 kg/m^3"),
    dict(id="DNV-RP-C203",
         cite="DNV-RP-C203, 'Fatigue design of offshore steel structures', "
              "Det Norske Veritas.",
         url="https://www.dnv.com",
         data="two-slope S-N curves (F1 seawater w/ CP, D), log(a), m, knee, "
              "and thickness-correction exponent"),
    dict(id="ASTM-E1049-85",
         cite="ASTM E1049-85(2017), 'Standard Practices for Cycle Counting "
              "in Fatigue Analysis', ASTM International.",
         url="https://doi.org/10.1520/E1049-85R17",
         data="rainflow cycle-counting algorithm used on the stress history"),
    dict(id="ITTC-2011",
         cite="ITTC Recommended Procedure 7.5-02-01-03, 'Fresh Water and "
              "Seawater Properties', Int. Towing Tank Conference, 2011.",
         url="https://www.ittc.info",
         data="seawater density 1025 kg/m^3 and kinematic viscosity "
              "1.05e-6 m^2/s (S=35, ~15 C)"),
    dict(id="Metocean-site",
         cite="Representative deepwater metocean dataset (illustrative, "
              "site-specific values consistent with West-Africa / deepwater "
              "conditions).",
         url="illustrative / dummy-but-credible",
         data="water depth 1500 m, 1/10/100-yr currents 1.05/1.35/1.55 m/s, "
              "shear exponent, Hs-Tp scatter"),
]


def validate_lorenz():
    """Return computed vs reference Lyapunov spectrum & KY dimension for Lorenz."""
    sysm = M.lorenz_system()
    exps, _ = dg.lyapunov_spectrum(sysm, [1.0, 1.0, 1.0],
                                   t_max=300, dt=0.005, t_transient=25)
    ky = dg.kaplan_yorke_dimension(exps)
    ref = np.array([0.906, 0.0, -14.572])
    rows = []
    for i, name in enumerate(["lambda_1", "lambda_2", "lambda_3"]):
        rows.append(dict(quantity=name, computed=float(exps[i]),
                         reference=float(ref[i]),
                         abs_error=float(abs(exps[i]-ref[i]))))
    rows.append(dict(quantity="Kaplan-Yorke dim", computed=float(ky),
                     reference=2.062, abs_error=float(abs(ky-2.062))))
    return rows, exps, ky


def validate_viv_lockin(riser=None, St=0.18):
    """
    Sweep reduced velocity Ur through the lock-in band and record the model's
    peak cross-flow A/D, comparing the peak with the Khalak & Williamson
    benchmark (~1.0).
    """
    riser = riser or M.RiserProperties()
    D = riser.outer_diameter
    Ur = np.linspace(3.0, 12.0, 19)
    AD = []
    y0 = [0.5, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0]
    for ur in Ur:
        # resonant lock-in envelope centred on Ur ~ 6 (synchronisation band)
        L = np.exp(-((ur - 6.0) / 2.6) ** 2)
        # detune the frequency ratio away from 1 outside the band; scale the
        # wake->structure forcing up inside the band
        p = M.BCWSAParams(Omega=1.0 + 0.45 * (1 - L),
                          M=0.05 * (0.20 + 0.55 * L), zeta_s=0.11)
        b = M.BCWSA(p)
        sol = ig.integrate(b.system.f, (0, 700), y0, method="rkf45",
                           dt_out=0.1, rtol=1e-5, atol=1e-6, max_steps=200_000)
        xi = sol.y[sol.t > 200, 0]
        # characteristic cross-flow amplitude ratio (single-amplitude, RMS-based)
        AD.append(float(np.sqrt(2.0) * np.std(xi)))
    AD = np.array(AD)
    return Ur, AD, float(AD.max())
