"""
_hot_storage.py — Module 2: High-Temperature Packed-Bed Tank
══════════════════════════════════════════════════════════════
Physical description
────────────────────
Charging : hot compressed gas (T2) flows in, heats the rock bed,
           cooled gas exits at T_HR_out toward recuperator / turbine inlet.
Discharging : reversed — cold gas from compressor outlet heats up through
              the bed, exits at turbine inlet temperature T_MAX.

Governing equations
───────────────────
Static (approach-ΔT model, Du2025):
  T_HR_out = T_in_gas − DT_HR

Dynamic — Schumann two-phase PDE (Desrues2010 Eq.25-26):
  ∂T_f/∂t + u_int·∂T_f/∂x = h_v/(ρ_f·cp_f·ε) · (T_s − T_f)
  ∂T_s/∂t                  = h_v/(ρ_s·cp_s·(1−ε)) · (T_f − T_s)
  ∂P/∂x = −A·u_s − B·u_s²          (Ergun)

Linearised first-order lag (§2.5 Step5, Zhang2020 §1.1):
  d(δT_out)/dt = (δT_in − δT_out) / τ_hot   [τ_hot ≈ 89 s]

Numerical scheme: operator splitting — upwind advection + semi-implicit
heat exchange (unconditionally stable, Desrues2010 §4.4).
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from brayton_v3_physics import packed_bed_step_v3


class PackedBedTank:
    """
    Packed-bed thermal storage tank.

    Parameters
    ──────────
    tank_id  : str   'hot' or 'cold'  (controls initial temp & flow sign)
    tau_lag  : float first-order lag time constant [s]
               (default: TAU_HOT_STORAGE or TAU_COLD_STORAGE from config)
    T_init   : float initial uniform temperature [K]
               (default: T_ENV_K)
    """

    def __init__(self,
                 tank_id: str = 'hot',
                 tau_lag: float = None,
                 T_init: float = None):
        self.tank_id = tank_id

        if tau_lag is None:
            tau_lag = (C.TAU_HOT_STORAGE  if tank_id == 'hot'
                       else C.TAU_COLD_STORAGE)
        if T_init is None:
            T_init = C.T_ENV_K

        self.tau_lag = tau_lag
        self.T_init  = T_init

        # PDE state fields (initialised on first call to reset_pde or step)
        self.T_f = np.full(C.N_X, T_init)
        self.T_s = np.full(C.N_X, T_init)

        # Linearised lag state (§2.5)
        self.dT_out = 0.0   # perturbation of outlet temperature [K]

    # ──────────────────────────────────────────────────────────────
    # Static model  (quasi-static, approach-ΔT)
    # ──────────────────────────────────────────────────────────────

    def static_outlet_temp(self, T_in: float, m_dot: float = None) -> float:
        """
        Quasi-static outlet temperature using approach ΔT (Du2025 style).

        For hot tank  : T_out = T_in − DT_HR  (gas cools down)
        For cold tank : T_out = T_in + DT_CR  (gas warms up slightly)

        Parameters
        ──────────
        T_in  : gas inlet temperature [K]
        m_dot : unused in static model (retained for API symmetry)
        """
        if self.tank_id == 'hot':
            return float(T_in - C.DT_HR)
        else:
            return float(T_in + C.DT_CR)

    def static_stored_energy(self, T_ref: float = C.T_ENV_K) -> float:
        """
        Thermal energy stored in solid bed above reference temperature [J].
          E = (1−ε)·ρ_s·cp_s·V_tank·(T_s_mean − T_ref)
        """
        T_s_mean = self.T_s.mean()
        return (1.0 - C.EPS_BED) * C.RHO_SOLID * C.CP_SOLID * C.V_TANK * (T_s_mean - T_ref)

    # ──────────────────────────────────────────────────────────────
    # Dynamic model — Schumann PDE step
    # ──────────────────────────────────────────────────────────────

    def reset_pde(self, T_init: float = None) -> None:
        """Reset PDE temperature fields to a uniform initial temperature."""
        T = T_init if T_init is not None else self.T_init
        self.T_f[:] = T
        self.T_s[:] = T

    def pde_step(self,
                 u_int: float,
                 T_inlet: float,
                 P_f: float,
                 dt: float) -> tuple:
        """
        Advance Schumann PDE by one timestep dt.

        Wraps packed_bed_step_v3 from brayton_v3_physics.py.
        Updates self.T_f, self.T_s in-place.

        Parameters
        ──────────
        u_int   : interstitial velocity [m/s]  (+ve = left→right)
        T_inlet : inlet gas temperature [K]
        P_f     : mean gas pressure [Pa]
        dt      : timestep [s]

        Returns
        ───────
        T_out  : outlet gas temperature [K]
        dP     : total pressure drop across tank [Pa]
        """
        # Stability: automatically sub-step if CFL > 0.95
        cfl = abs(u_int) * dt / C.DX
        if cfl > 0.95:
            n_sub = int(np.ceil(cfl / 0.9))
            dt_sub = dt / n_sub
        else:
            n_sub  = 1
            dt_sub = dt

        dP_total = 0.0
        for _ in range(n_sub):
            self.T_f, self.T_s, dP = packed_bed_step_v3(
                self.T_f, self.T_s, u_int, T_inlet, P_f, dt_sub)
            dP_total += dP

        # Outlet: last node if u_int>0, first node if u_int<0
        T_out = float(self.T_f[-1] if u_int >= 0.0 else self.T_f[0])
        return T_out, dP_total

    # ──────────────────────────────────────────────────────────────
    # Dynamic model — linearised first-order lag ODE (§2.5 Step5/7)
    # ──────────────────────────────────────────────────────────────

    def lag_ode_rhs(self, dT_inlet: float) -> float:
        """
        First-order lag ODE RHS for outlet temperature perturbation:
          d(δT_out)/dt = (δT_in − δT_out) / τ_lag

        Parameters
        ──────────
        dT_inlet : current inlet temperature perturbation δT_in [K]

        Returns dδT_out/dt [K/s]
        """
        return (dT_inlet - self.dT_out) / self.tau_lag

    def lag_step(self, dT_inlet: float, dt: float) -> float:
        """
        Euler advance of the lag ODE; updates self.dT_out.

        Returns updated dT_out [K].
        """
        self.dT_out += self.lag_ode_rhs(dT_inlet) * dt
        return self.dT_out

    @property
    def T_out_lag(self) -> float:
        """Absolute outlet temperature using lag model: T_out_0 + δT_out."""
        T_out_0 = self.T_init - (C.DT_HR if self.tank_id == 'hot' else -C.DT_CR)
        return T_out_0 + self.dT_out

    # ──────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────

    def mean_solid_temp(self) -> float:
        return float(self.T_s.mean())

    def mean_fluid_temp(self) -> float:
        return float(self.T_f.mean())
