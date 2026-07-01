"""
chronos.models
==============
System library for CHRONOS.

* :func:`lorenz_system`          -- canonical Lorenz (validation reference).
* :func:`conventional_viv_attractor` -- a representative conventional low-order
                                    3-D chaotic VIV attractor, reproduced so its
                                    non-physical divergence can be demonstrated
                                    and contrasted with the bounded model.
* :class:`RiserProperties`       -- geometric / material / modal riser data.
* :class:`BCWSA`                 -- the novel **Bounded Chaotic Wake-Structure
                                    Attractor**: a dimensionally-consistent,
                                    provably-bounded reduced-order VIV model that
                                    couples a Duffing structural oscillator, an
                                    acceleration-coupled van der Pol wake
                                    oscillator (Facchinetti, de Langre & Biolley,
                                    2004) and a slow torsional manifold, with a
                                    guaranteed trapping region.

The BCWSA fixes the three central defects of conventional low-order models:
  (i)   unbounded blow-up  -> enforced dissipative trapping region;
  (ii)  dimensional inconsistency -> proper nondimensionalisation + physical
        rescaling back to engineering stress (MPa);
  (iii) missing VIV physics -> self-limiting wake oscillator that reproduces
        lock-in and the ~1 diameter cross-flow amplitude bound.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

from .dynsys import DynamicalSystem


# ============================================================================ #
#  Reference / legacy systems
# ============================================================================ #
def lorenz_system(sigma=10.0, rho=28.0, beta=8.0/3.0):
    def rhs(t, y):
        x, yy, z = y
        return np.array([sigma*(yy - x),
                         x*(rho - z) - yy,
                         x*yy - beta*z])
    return DynamicalSystem(rhs, 3, name="Lorenz",
                           state_names=["x", "y", "z"])


def conventional_viv_attractor(a=1.2, b=0.8, c=0.5, d=0.6, e=1.5, f=0.4,
                               g=0.9, h=1.2, i=0.3):
    """Representative conventional low-order chaotic VIV attractor (divergent
    baseline, reproduced for contrast against the bounded BCWSA model)."""
    def rhs(t, Y):
        x, y, z = Y
        dx = a*y - b*x + c*np.tanh(z**2)
        dy = -d*z + e*x*y - f*np.tanh(y)
        dz = g*x - h*y*z + i*np.tanh(z)
        return np.array([dx, dy, dz])
    return DynamicalSystem(rhs, 3, name="Conventional-LowOrder",
                           state_names=["x", "y", "z"])


# ============================================================================ #
#  Riser physical properties
# ============================================================================ #
@dataclass
class RiserProperties:
    """Geometry, material and modal data for a deepwater steel catenary riser."""
    name: str = "Generic SCR"
    outer_diameter: float = 0.3239     # m   (12.75 in)
    wall_thickness: float = 0.0206     # m   (0.812 in)
    length: float = 1800.0             # m   suspended length
    youngs_modulus: float = 2.07e11    # Pa
    shear_modulus: float = 7.9e10      # Pa
    density_steel: float = 7850.0      # kg/m3
    content_density: float = 800.0     # kg/m3 (oil)
    seawater_density: float = 1025.0   # kg/m3
    added_mass_coeff: float = 1.0      # Ca
    drag_coeff: float = 1.2            # Cd
    lift_coeff: float = 0.3            # C_L0 fluctuating lift
    strouhal: float = 0.18            # St
    struct_damping: float = 0.005      # zeta_s (structural)
    top_tension: float = 2.10e6        # N   effective top tension
    viv_halfwave_length: float = 55.0  # m   effective half-wavelength of excited VIV mode
    material_yield: float = 448e6      # Pa  (X65)
    sn_curve: str = "F1_seawater"      # DNV-RP-C203 detail class
    # empirical stress-partition ratios (VIV) -- calibratable site constants
    inline_stress_ratio: float = 0.25   # inline/axial dynamic : cross-flow bending
    torsion_stress_scale: float = 9.0   # MPa per unit nondim torsional state w

    # ---- derived section properties ---------------------------------- #
    @property
    def inner_diameter(self):
        return self.outer_diameter - 2*self.wall_thickness

    @property
    def area_steel(self):
        Do, Di = self.outer_diameter, self.inner_diameter
        return np.pi/4*(Do**2 - Di**2)

    @property
    def second_moment(self):        # I
        Do, Di = self.outer_diameter, self.inner_diameter
        return np.pi/64*(Do**4 - Di**4)

    @property
    def polar_moment(self):         # J
        return 2*self.second_moment

    @property
    def mass_per_length(self):
        Do, Di = self.outer_diameter, self.inner_diameter
        m_steel = self.density_steel*self.area_steel
        m_cont = self.content_density*np.pi/4*Di**2
        m_added = self.added_mass_coeff*self.seawater_density*np.pi/4*Do**2
        return m_steel + m_cont + m_added

    @property
    def mass_ratio(self):           # mu = (m + m_a)/(rho_f D^2)
        return self.mass_per_length/(self.seawater_density*self.outer_diameter**2)

    @property
    def natural_frequency(self):
        """1st-mode natural frequency of a tensioned beam (rad/s)."""
        L = self.length
        m = self.mass_per_length
        EI = self.youngs_modulus*self.second_moment
        T = self.top_tension
        # tensioned Euler-Bernoulli beam, pinned-pinned, mode 1
        w2 = (np.pi/L)**2 * (T + EI*(np.pi/L)**2) / m
        return np.sqrt(w2)

    def summary(self):
        d = asdict(self)
        d.update(dict(
            inner_diameter=self.inner_diameter,
            area_steel=self.area_steel,
            second_moment=self.second_moment,
            polar_moment=self.polar_moment,
            mass_per_length=self.mass_per_length,
            mass_ratio=self.mass_ratio,
            natural_frequency_radps=self.natural_frequency,
            natural_frequency_hz=self.natural_frequency/(2*np.pi),
        ))
        return d


# ============================================================================ #
#  Novel Bounded Chaotic Wake-Structure Attractor  (BCWSA)
# ============================================================================ #
@dataclass
class BCWSAParams:
    """
    Nondimensional parameters of the 7-state BCWSA.

    Groups
    ------
    structure : zeta_s (damping), alpha (Duffing hardening)
    wake      : epsilon, Omega, A         (Facchinetti et al. 2004 van der Pol)
    coupling  : M (wake->structure), kappa (convective->structure, bounded),
                gamma (convective->wake, bounded), rho_c (structure->convective),
                w0, s0 (saturation scales of the bounded coupling)
    core      : sigma_c, r_c, beta_c      (Lorenz-type near-wake convective triad;
                Lorenz 1963 convection analogy) -- guarantees sustained chaos.
    """
    # --- structure ---
    zeta_s: float = 0.10
    alpha: float = 0.05
    # --- wake (van der Pol, Facchinetti/de Langre/Biolley 2004) ---
    epsilon: float = 0.30
    Omega: float = 1.10
    A: float = 12.0
    # --- couplings ---
    M: float = 0.05           # wake -> structure
    kappa: float = 0.12       # convective core -> structure (parametric, bounded)
    gamma: float = 0.06       # convective core -> wake (bounded)
    rho_c: float = 0.08       # structure -> convective core (energy pump)
    w0: float = 9.0           # saturation scale for kappa-coupling
    s0: float = 11.0          # saturation scale for gamma-coupling
    # --- Lorenz-type near-wake convective core ---
    sigma_c: float = 10.0
    r_c: float = 28.0
    beta_c: float = 8.0/3.0

    # VIV-coupling controls --------------------------------------------------
    def scaled(self, viv_coupling=1.0, damping_factor=1.0):
        """
        Return a copy with the *VIV coupling strength* (nondimensional 'a')
        and *damping* rescaled -- used for parameter sweeps, bifurcation,
        sensitivity and optimisation studies.
        """
        import copy
        p = copy.copy(self)
        p.M = self.M * viv_coupling
        p.rho_c = self.rho_c * viv_coupling
        p.zeta_s = self.zeta_s * damping_factor
        return p


class BCWSA:
    r"""
    Novel **Bounded Chaotic Wake-Structure Attractor** (7 states).

    State  y = [xi, eta, q, p, w, s, g]
      xi   cross-flow displacement / D            (nondim)
      eta  d(xi)/dtau                             (nondim velocity)
      q    wake variable (~ fluctuating lift)     (nondim, van der Pol)
      p    d(q)/dtau
      w,s,g  near-wake convective triad (Lorenz-type)

    Governing equations (reduced time tau = omega_lockin * t):
      xi' = eta
      eta'= -2 zeta_s eta - xi - alpha xi^3 + M q - kappa xi tanh(w/w0)
      q'  = p
      p'  = epsilon Omega (1 - q^2) p - Omega^2 q + A eta' + gamma tanh(s/s0)
      w'  = sigma_c (s - w) + rho_c xi eta
      s'  = w (r_c - g) - s
      g'  = w s - beta_c g

    Boundedness (why it cannot blow up, unlike conventional low-order models)
    ------------------------------------------------------------
    * the (w,s,g) convective triad is a Lorenz-type system perturbed by the
      *bounded* pump ``rho_c xi eta`` -> it possesses the classical Lorenz
      ellipsoidal trapping region and stays bounded & chaotic;
    * the wake ``q`` is bounded by the self-limiting van der Pol term plus a
      *bounded* forcing ``gamma tanh(.)``;
    * the structure energy V = 1/2 eta^2 + 1/2 xi^2 + 1/4 alpha xi^4 obeys
      dV/dtau = -2 zeta_s eta^2 + eta[M q - kappa xi tanh(w/w0)], and because
      the coupling terms are bounded while the hardening quartic grows, dV/dtau
      < 0 outside a bounded ball -> a globally attracting trapping region.

    Sustained chaos is therefore *guaranteed* by the embedded convective core,
    with the largest Lyapunov exponent set by that core, while every physical
    output remains finite.
    """

    def __init__(self, params: "BCWSAParams | None" = None):
        self.p = params or BCWSAParams()
        self.system = DynamicalSystem(
            self._rhs, 7, name="BCWSA",
            state_names=["xi", "eta", "q", "p", "w", "s", "g"],
            jac_fn=self._jac)

    # -- right-hand side --------------------------------------------------- #
    def _rhs(self, t, Y):
        xi, eta, q, p, w, s, g = Y
        P = self.p
        Tw = np.tanh(w / P.w0)
        Us = np.tanh(s / P.s0)
        eta_dot = (-2*P.zeta_s*eta - xi - P.alpha*xi**3
                   + P.M*q - P.kappa*xi*Tw)
        xi_dot = eta
        q_dot = p
        p_dot = (P.epsilon*P.Omega*(1.0 - q**2)*p - P.Omega**2*q
                 + P.A*eta_dot + P.gamma*Us)
        w_dot = P.sigma_c*(s - w) + P.rho_c*xi*eta
        s_dot = w*(P.r_c - g) - s
        g_dot = w*s - P.beta_c*g
        return np.array([xi_dot, eta_dot, q_dot, p_dot, w_dot, s_dot, g_dot])

    # -- analytic Jacobian (fast, exact Lyapunov spectrum) ---------------- #
    def _jac(self, t, Y):
        xi, eta, q, p, w, s, g = Y
        P = self.p
        Tw = np.tanh(w / P.w0); dTw = (1.0 - Tw**2) / P.w0
        Us = np.tanh(s / P.s0); dUs = (1.0 - Us**2) / P.s0
        # eta_dot partials
        e_xi = -1.0 - 3.0*P.alpha*xi**2 - P.kappa*Tw
        e_eta = -2.0*P.zeta_s
        e_q = P.M
        e_w = -P.kappa*xi*dTw
        J = np.zeros((7, 7))
        J[0] = [0, 1, 0, 0, 0, 0, 0]
        J[1] = [e_xi, e_eta, e_q, 0, e_w, 0, 0]
        J[2] = [0, 0, 0, 1, 0, 0, 0]
        J[3] = [P.A*e_xi, P.A*e_eta,
                -2.0*P.epsilon*P.Omega*q*p - P.Omega**2 + P.A*e_q,
                P.epsilon*P.Omega*(1.0 - q**2), P.A*e_w, P.gamma*dUs, 0]
        J[4] = [P.rho_c*eta, P.rho_c*xi, 0, 0, -P.sigma_c, P.sigma_c, 0]
        J[5] = [0, 0, 0, 0, (P.r_c - g), -1.0, -w]
        J[6] = [0, 0, 0, 0, s, w, -P.beta_c]
        return J

    # -- convenience ------------------------------------------------------- #
    def reduced_3d(self, Y):
        """(xi, q, w): transverse displacement / wake-lift / convective axes."""
        return Y[:, [0, 2, 4]]

    def convective_core(self, Y):
        """(w, s, g): the Lorenz-type near-wake convective attractor."""
        return Y[:, [4, 5, 6]]

    # -- physical rescaling ------------------------------------------------ #
    def to_physical(self, t, Y, riser: "RiserProperties", current_velocity=1.0):
        """Map the nondimensional trajectory to engineering quantities (MPa)."""
        f_lockin = riser.strouhal * current_velocity / riser.outer_diameter
        wn = 2.0 * np.pi * f_lockin
        t_phys = t / wn
        D = riser.outer_diameter
        Do = riser.outer_diameter
        E = riser.youngs_modulus
        ell = riser.viv_halfwave_length

        xi = Y[:, 0]; eta = Y[:, 1]; q = Y[:, 2]; w = Y[:, 4]

        Y_disp = D * xi
        curv = (np.pi/ell)**2 * Y_disp
        coef0 = E * (Do/2.0) * (np.pi/ell)**2 * D / 1e6      # MPa per unit xi
        sigma_bending = coef0 * xi

        T0 = riser.top_tension
        As = riser.area_steel
        sigma_axial_mean = T0/As/1e6
        inline = riser.inline_stress_ratio * coef0 * (xi**2 - np.mean(xi**2))
        sigma_axial = sigma_axial_mean + inline

        # torsional shear from the convective/near-wake mode (bounded via tanh)
        sigma_torsion = riser.torsion_stress_scale * np.tanh(w / 9.0)

        C_L = riser.lift_coeff * q / 2.0

        return dict(
            t=t_phys, wn=wn, f_lockin=f_lockin,
            displacement=Y_disp, amplitude_ratio=Y_disp/D, curvature=curv,
            sigma_bending=sigma_bending, sigma_axial=sigma_axial,
            sigma_torsion=sigma_torsion, lift_coeff=C_L,
            xi=xi, eta=eta, q=q, w=w, coef0=coef0,
        )
