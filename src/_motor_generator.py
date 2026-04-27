"""
_motor_generator.py — Module 4: Electric Motor / Generator
════════════════════════════════════════════════════════════
Physical description
────────────────────
Charging mode  : Electric motor drives the co-shaft (compressor + turbine).
                 Operator commands P_EC [W]; actual shaft torque is
                 τ_motor = min(P_EC / ω, τ_max).
                 Motor power command tracks operator via governor lag τ_gov.
                 Inventory control adjusts system pressure α_P with lag τ_inv.

Discharging mode: Generator harvests shaft power.
                  W_gen = η_gen · (W_exp − W_comp − W_fric)

Static model (algebraic)
────────────────────────
  τ_motor = min(P_cmd / ω, τ_max)              [charging]
  W_gen   = η_gen · (W_exp − W_comp − W_fric)  [discharging]

Dynamic model (ODEs)
────────────────────
  dP_motor/dt = (P_cmd − P_motor) / τ_gov      [governor lag, τ_gov=0.5 s]
  dα_P/dt     = (α_cmd  − α_P)   / τ_inv      [inventory control, τ_inv=30 s]

  Motor power command (operator schedule → immediate or via ramp):
    P_cmd(t) = α_cmd(t) · P_cmd0
  where P_cmd0 is the steady-state balance: W_comp0 − W_exp0 + W_fric0

  Inventory pressure scale α_P controls:
    P_lo = P_lo0 · α_P,  P_hi = P_hi0 · α_P  → π = const ✓
    ṁ    = ρ(T1, P_lo) · V̇_ref              → V̇ = const ✓
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C


class MotorGenerator:
    """
    Motor (charging) / Generator (discharging) module.

    Parameters
    ──────────
    mode       : 'charge' or 'discharge'
    eta_motor  : motor efficiency (electrical → mechanical)
    eta_gen    : generator efficiency (mechanical → electrical)
    tau_gov    : governor time constant [s]
    tau_inv    : inventory control time constant [s]
    P_cmd0     : steady-state motor power command [W]
                 (set by calibrate() from design-point cycle)
    """

    def __init__(self,
                 mode: str      = 'charge',
                 eta_motor: float = C.ETA_MOTOR,
                 eta_gen: float   = C.ETA_GEN,
                 tau_gov: float   = C.TAU_GOV,
                 tau_inv: float   = C.TAU_INV):
        self.mode      = mode
        self.eta_motor = eta_motor
        self.eta_gen   = eta_gen
        self.tau_gov   = tau_gov
        self.tau_inv   = tau_inv

        # Set by calibrate()
        self._P_cmd0    = None   # W — steady-state power balance
        self._V_dot_ref = None   # m³/s — reference volume flow
        self._P_lo0     = C.P_LOW_PA
        self._P_hi0     = C.P_HIGH_PA

    # ──────────────────────────────────────────────────────────────
    # Calibration  (call once after steady_state solve)
    # ──────────────────────────────────────────────────────────────

    def calibrate(self,
                  W_comp0: float, W_exp0: float,
                  V_dot_ref: float,
                  P_lo0: float = C.P_LOW_PA,
                  P_hi0: float = C.P_HIGH_PA) -> None:
        """
        Set steady-state references.

        P_cmd0 accounts for motor efficiency so that at steady state:
          τ_motor = η_motor × P_cmd0 / ω = (W_comp − W_exp + W_fric) / ω
        i.e. P_cmd0 = (W_comp − W_exp + W_fric) / η_motor

        Parameters
        ──────────
        W_comp0   : design-point compressor mechanical power [W]
        W_exp0    : design-point expander mechanical power [W]
        V_dot_ref : design-point volume flow rate [m³/s]
        """
        W_fric0 = C.K_FRIC * C.TAU_RATED * C.OMEGA_DESIGN
        W_mech  = W_comp0 - W_exp0 + W_fric0   # net mechanical demand
        self._P_cmd0    = W_mech / self.eta_motor  # electrical command for balance
        self._V_dot_ref = V_dot_ref
        self._P_lo0     = P_lo0
        self._P_hi0     = P_hi0

    # ──────────────────────────────────────────────────────────────
    # Static model
    # ──────────────────────────────────────────────────────────────

    def motor_torque(self, P_cmd: float, omega: float) -> float:
        """
        Motor output torque [N·m].
        τ = min(η_motor · P_cmd / ω,  τ_max)
        """
        w = max(abs(omega), 1.0)
        return float(min(self.eta_motor * P_cmd / w, C.TAU_MAX))

    def friction_power(self, omega: float) -> float:
        """Friction dissipation W_fric = τ_fric · ω [W]."""
        return float(C.K_FRIC * C.TAU_RATED * abs(omega))

    def generator_power(self,
                        W_exp: float,
                        W_comp: float,
                        omega: float) -> float:
        """
        Net electrical output [W] in discharge mode.
        W_gen = η_gen · (W_exp − W_comp − W_fric)
        """
        W_fric = self.friction_power(omega)
        return float(self.eta_gen * max(W_exp - W_comp - W_fric, 0.0))

    def mass_flow_from_alpha(self, alpha_P: float, T1: float) -> float:
        """
        Inventory-control mass flow: ṁ = ρ(P_lo) · V̇_ref
        Maintains V̇ = const → ṁ ∝ α_P.
        """
        if self._V_dot_ref is None:
            raise RuntimeError("Call calibrate() before mass_flow_from_alpha().")
        P_lo  = self._P_lo0 * alpha_P
        rho   = P_lo / (C.R_GAS * T1)
        m_dot = rho * self._V_dot_ref
        return float(np.clip(m_dot,
                              C.M_DOT_DESIGN * C.ALPHA_MIN,
                              C.M_DOT_DESIGN * C.ALPHA_MAX))

    def operator_power_cmd(self, alpha_cmd: float) -> float:
        """
        Operator power command [W].
        P_cmd = α_cmd · P_cmd0  (scales linearly with load fraction)
        """
        if self._P_cmd0 is None:
            raise RuntimeError("Call calibrate() before operator_power_cmd().")
        return float(np.clip(alpha_cmd * self._P_cmd0, 0.0, C.P_MOTOR_RATED))

    # ──────────────────────────────────────────────────────────────
    # Dynamic model ODE RHS
    # ──────────────────────────────────────────────────────────────

    def ode_rhs(self, state: dict, inputs: dict) -> dict:
        """
        ODE right-hand sides for motor governor and inventory control.

        State variables
        ───────────────
        state['P_motor'] : actual motor shaft power [W]
        state['alpha_P'] : current pressure scale factor [-]

        Inputs
        ──────
        inputs['alpha_cmd'] : operator load command (0–1.5)

        Returns dict
        ────────────
        d_P_motor : dP_motor/dt [W/s]
        d_alpha_P : dα_P/dt    [1/s]
        P_cmd     : target motor power (for record keeping) [W]
        """
        P_motor  = state['P_motor']
        alpha_P  = state['alpha_P']
        alpha_cmd = inputs['alpha_cmd']

        P_cmd = self.operator_power_cmd(alpha_cmd)

        d_P_motor = (P_cmd - P_motor) / self.tau_gov
        d_alpha_P = (alpha_cmd - alpha_P) / self.tau_inv

        return dict(d_P_motor=d_P_motor,
                    d_alpha_P=float(np.clip(d_alpha_P,
                                            (C.ALPHA_MIN - alpha_P) / self.tau_inv,
                                            (C.ALPHA_MAX - alpha_P) / self.tau_inv)),
                    P_cmd=P_cmd)

    # ──────────────────────────────────────────────────────────────
    # Convenience: schedule helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def step_schedule(t: float, t_step: float, alpha_final: float) -> float:
        """Instantaneous step: α = 1 before t_step, α_final after."""
        return alpha_final if t >= t_step else 1.0

    @staticmethod
    def ramp_schedule(t: float, t_start: float, t_end: float,
                      alpha_final: float) -> float:
        """Linear ramp from 1 to alpha_final over [t_start, t_end]."""
        if t < t_start:
            return 1.0
        if t > t_end:
            return alpha_final
        frac = (t - t_start) / (t_end - t_start)
        return 1.0 + frac * (alpha_final - 1.0)
