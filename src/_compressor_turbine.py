"""
_compressor_turbine.py — Module 1: Co-shaft Compressor & Turbine
════════════════════════════════════════════════════════════════════
Physical description
────────────────────
Compressor and turbine share one rigid shaft (same ω, torques coupled).

Charging mode  : compressor raises P_lo→P_hi; turbine expands P_hi→P_lo
                 (partial work recovery on same shaft)
Discharging mode: turbine drives output; compressor re-pressurises cold gas
                  (both contribute to shaft torque balance)

Static model (algebraic, for steady_state.py)
─────────────────────────────────────────────
  T2 = T1·β_c^(κ/η_p)            (polytropic compression, REFPROP Heun)
  T5 = T4·β_t^(−η_p·κ)          (polytropic expansion,   REFPROP Heun)
  W_comp = ṁ·cp·(T2−T1)
  W_exp  = ṁ·cp·(T4−T5)
  β_e    = Du2025 Eq.11 coupling

Dynamic model (ODEs, for dynamic_perturbation.py)
──────────────────────────────────────────────────
  dω/dt     = (τ_motor − τ_comp + τ_exp − τ_fric) / J   [Zhang2020 Eq.11]
  dP_sys/dt = R·T/V·(ṁ_in − ṁ_out)                      [Zhang2020 Eq.7]
  dβ_t/dt   = (β_c − β_t) / τ_vol                        [§2.5 Step3]

Compressor characteristic map (§2.4, Desrues2010)
──────────────────────────────────────────────────
  β_c(N_r) = β_c0·[1 + a_π·(N_r²−1)]   (along operating line ṁ_r=const)
  (linearised: Δβ_c = 2·a_π·(β_c0−1)·Δn/n0)
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from brayton_v3_physics import (
    polytropic_machine,
    charge_expansion_ratio,
    volume_ode,
)


class CompressorTurbine:
    """
    Co-shaft compressor + turbine module.

    Parameters (all optional, override config defaults)
    ───────────────────────────────────────────────────
    eta_p_c : float  polytropic efficiency, compressor
    eta_p_e : float  polytropic efficiency, expander
    J       : float  shaft rotational inertia [kg·m²]
    a_pi    : float  compressor map curvature  (Desrues Eq.30)
    tau_vol : float  volumetric lag time const [s]
    """

    def __init__(self,
                 eta_p_c: float = C.ETA_POLY_C,
                 eta_p_e: float = C.ETA_POLY_E,
                 J: float = C.J_ROTOR,
                 a_pi: float = C.A_PI,
                 tau_vol: float = C.TAU_VOL):
        self.eta_p_c = eta_p_c
        self.eta_p_e = eta_p_e
        self.J       = J
        self.a_pi    = a_pi
        self.tau_vol = tau_vol

        # Design-point calibration (set by calibrate() in steady_state.py)
        self._F_cal_C = 1.0
        self._F_cal_E = 1.0
        self._beta_c0 = C.PI_DESIGN
        self._T1_0    = C.T_ENV_K
        self._T4_0    = C.T_ENV_K

    # ──────────────────────────────────────────────────────────────
    # Static model  (called by steady_state.py per iteration)
    # ──────────────────────────────────────────────────────────────

    def static_state(self,
                     T1: float, P_lo: float, P_hi: float,
                     T4: float, m_dot: float) -> dict:
        """
        Quasi-static thermodynamic state (REFPROP Heun's method).

        Parameters
        ──────────
        T1    : compressor inlet temperature [K]
        P_lo  : low-pressure side [Pa]
        P_hi  : high-pressure side [Pa]
        T4    : turbine inlet temperature [K]  (= hot-storage outlet)
        m_dot : mass flow rate [kg/s]

        Returns dict
        ────────────
        T2, T5          : outlet temperatures [K]
        w_comp, w_exp   : specific work [J/kg]
        W_comp, W_exp   : shaft power [W]
        W_net           : net charging power W_comp − W_exp [W]
        beta_c, beta_e  : pressure ratios [-]
        """
        comp = polytropic_machine(T1, P_lo, P_hi, self.eta_p_c)
        exp  = polytropic_machine(T4, P_hi, P_lo, self.eta_p_e)

        T2     = comp['T_out']
        T5     = exp['T_out']
        w_comp = comp['w_spec']
        w_exp  = exp['w_spec']
        W_comp = m_dot * w_comp
        W_exp  = m_dot * w_exp
        W_net  = W_comp - W_exp

        beta_c = P_hi / P_lo
        beta_e = charge_expansion_ratio(beta_c,
                                         p1=P_lo,
                                         dp_HR=C.DP_HR * (P_lo / C.P_LOW_PA),
                                         dp_CR=C.DP_CR * (P_lo / C.P_LOW_PA))
        return dict(
            T2=T2, T5=T5,
            w_comp=w_comp, w_exp=w_exp,
            W_comp=W_comp, W_exp=W_exp, W_net=W_net,
            beta_c=beta_c, beta_e=beta_e,
        )

    def calibrate(self, T1_0: float, T4_0: float,
                  P_lo: float, P_hi: float, m_dot: float) -> None:
        """
        Compute calibration factors so that ideal-gas fast formulae
        reproduce REFPROP design-point values exactly.
        Called once by dynamic_perturbation.py at startup.
        """
        ss = self.static_state(T1_0, P_lo, P_hi, T4_0, m_dot)
        beta_c0 = P_hi / P_lo

        W_c_ideal = (m_dot * C.CP_GAS * T1_0
                     * (beta_c0**(C.KAPPA / self.eta_p_c) - 1.0))
        W_e_ideal = (m_dot * C.CP_GAS * T4_0
                     * (1.0 - beta_c0**(-self.eta_p_e * C.KAPPA)))

        self._F_cal_C = ss['W_comp'] / max(W_c_ideal, 1.0)
        self._F_cal_E = ss['W_exp']  / max(W_e_ideal, 1.0)
        self._beta_c0 = beta_c0
        self._T1_0    = T1_0
        self._T4_0    = T4_0

    # ──────────────────────────────────────────────────────────────
    # Compressor map  (§2.4 operating line, Desrues2010)
    # ──────────────────────────────────────────────────────────────

    def beta_c_from_speed(self, n_rpm: float,
                          n0: float = C.N_DESIGN_RPM,
                          m_dot: float = None) -> float:
        """
        §2.5 Step2: compressor pressure ratio from shaft speed and mass flow.

        Along the equal-corrected-flow operating line (inventory control, ṁ_r=const):
          β_c(N_r) = β_c0 + a_π·(β_c0−1)·(2r + r²),  r = Δn/n0

        Under inventory control, ṁ_r = ṁ√T/P = const, so operating point
        moves with N_r along the characteristic. The pressure ratio also
        scales with the inventory fraction α_P (since P scales with α_P
        and the compressor map produces the same β at same N_r and ṁ_r):
          β_c = β_c0 × α_P^0  (ratio doesn't change with pressure level)

        The speed-based correction captures deviations from the inventory
        operating line due to shaft speed transients (§2.5 Step 2).
        """
        r  = (n_rpm - n0) / n0
        db = self.a_pi * (self._beta_c0 - 1.0) * (2.0 * r + r * r)
        return float(np.clip(self._beta_c0 + db, 1.5, self._beta_c0 * 1.15))

    # ──────────────────────────────────────────────────────────────
    # Fast power formulae  (ideal gas + calibration, no CoolProp)
    # ──────────────────────────────────────────────────────────────

    def _W_comp_fast(self, beta_c: float, m_dot: float, T1: float) -> float:
        return float(self._F_cal_C * m_dot * C.CP_GAS * T1
                     * (beta_c**(C.KAPPA / self.eta_p_c) - 1.0))

    def _W_exp_fast(self, beta_t: float, m_dot: float, T4: float) -> float:
        return float(self._F_cal_E * m_dot * C.CP_GAS * T4
                     * (1.0 - beta_t**(-self.eta_p_e * C.KAPPA)))

    # ──────────────────────────────────────────────────────────────
    # Dynamic model ODE RHS  (called each timestep by dynamic_perturbation.py)
    # ──────────────────────────────────────────────────────────────

    def ode_rhs(self, state: dict, inputs: dict) -> tuple:
        """
        Compute ODE right-hand sides for shaft + dead volume + β_t lag.

        State variables
        ───────────────
        state['omega']   : shaft angular velocity ω [rad/s]
        state['P_sys']   : dead-volume pressure [Pa]
        state['beta_t']  : current turbine expansion ratio [-]

        Inputs (algebraic, computed upstream each step)
        ───────────────────────────────────────────────
        inputs['P_motor'] : motor/generator shaft power [W]
        inputs['m_dot']   : mass flow rate [kg/s]
        inputs['T1']      : compressor inlet temperature [K]
        inputs['T4']      : turbine inlet temperature [K]
        inputs['P_lo']    : low-pressure side [Pa]
        inputs['P_hi']    : high-pressure side [Pa]

        Returns
        ───────
        deriv : dict  {d_omega, d_P_sys, d_beta_t}
        alg   : dict  algebraic outputs {n_rpm, beta_c, T2, T5, W_comp, W_exp, W_net, tau_net}
        """
        omega   = state['omega']
        P_sys   = state['P_sys']
        beta_t  = state['beta_t']

        P_motor = inputs['P_motor']
        m_dot   = inputs['m_dot']
        T1      = inputs['T1']
        T4      = inputs['T4']
        P_lo    = inputs['P_lo']
        P_hi    = inputs['P_hi']

        # ── Shaft speed ──────────────────────────────────────────
        n_rpm = omega * 60.0 / (2.0 * np.pi)

        # ── Compressor map → β_c, T2 ────────────────────────────
        beta_c = self.beta_c_from_speed(n_rpm)
        T2     = T1 * beta_c**(C.KAPPA / self.eta_p_c)

        # ── Fast power (calibrated ideal gas) ────────────────────
        W_comp = self._W_comp_fast(beta_c, m_dot, T1)
        W_exp  = self._W_exp_fast(beta_t, m_dot, T4)
        W_net  = W_comp - W_exp

        # ── Turbine outlet temperature ────────────────────────────
        T5 = T4 * beta_t**(-self.eta_p_e * C.KAPPA)

        # ── Torques ──────────────────────────────────────────────
        # P_motor is the electrical power command; mechanical output torque
        # includes motor efficiency: τ_motor = η_motor × P_motor / ω
        w_safe   = max(abs(omega), 1.0)
        tau_m    = min(C.ETA_MOTOR * P_motor / w_safe, C.TAU_MAX)
        tau_c    = W_comp / w_safe
        tau_e    = W_exp  / w_safe
        tau_f    = C.K_FRIC * C.TAU_RATED * (w_safe / C.OMEGA_DESIGN)
        tau_net  = tau_m - tau_c + tau_e - tau_f

        # ── ODE 1: shaft angular momentum (Zhang2020 Eq.11) ──────
        d_omega = tau_net / self.J

        # ── ODE 2: dead-volume pressure lag (Zhang2020 Eq.7) ─────
        # Approximate: ṁ_in = m_dot (compressor inlet),
        #              ṁ_out = m_dot (expander outlet) → steady state dP≈0
        # Perturbation source: α_P changes ṁ_in ≠ ṁ_out transiently
        # (m_dot_in − m_dot_out) is passed in inputs if available
        m_in  = inputs.get('m_dot_in',  m_dot)
        m_out = inputs.get('m_dot_out', m_dot)
        T_avg = (T1 + T4) / 2.0
        d_P_sys = volume_ode(P_sys, m_in, m_out, T_avg,
                              V_dead=C.V_SYS_TOTAL)

        # ── ODE 3: expander ratio volumetric lag (§2.5 Step3) ────
        d_beta_t = (beta_c - beta_t) / self.tau_vol

        deriv = dict(d_omega=d_omega, d_P_sys=d_P_sys, d_beta_t=d_beta_t)
        alg   = dict(n_rpm=n_rpm, beta_c=beta_c, T2=T2, T5=T5,
                     W_comp=W_comp, W_exp=W_exp, W_net=W_net,
                     tau_net=tau_net, tau_m=tau_m, tau_c=tau_c,
                     tau_e=tau_e, tau_f=tau_f)
        return deriv, alg
