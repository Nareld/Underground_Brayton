"""
config.py — PTES Simulation Configuration Loader
══════════════════════════════════════════════════
Reads a JSON config file and exposes all parameters as attributes
of a Cfg object. All module scripts import from this loader.

Usage
─────
  import config as C          # loads default.json automatically
  C.load('configs/inv_pid.json')  # switch to a different config

  # All parameters accessible as attributes:
  C.P_LOW_PA, C.J_ROTOR, C.TAU_INV, ...

JSON schema: see configs/default.json for the canonical structure.
"""

import json, os, math
import numpy as np

# ── Internal state ────────────────────────────────────────────────
_cfg_dict: dict = {}
_cfg_path: str  = ''


class _Cfg:
    """Namespace object — all config parameters as attributes."""
    pass


def load(path: str) -> None:
    """
    Load a JSON config file and populate all module-level attributes.

    Parameters
    ──────────
    path : path to JSON config file (absolute or relative to CWD)
    """
    global _cfg_dict, _cfg_path

    abs_path = os.path.abspath(path)
    with open(abs_path, encoding='utf-8') as f:
        raw = json.load(f)
    _cfg_dict = raw
    _cfg_path = abs_path
    _apply(raw)


def _apply(raw: dict) -> None:
    """Compute all derived parameters and inject into module namespace."""
    import sys
    m = sys.modules[__name__]

    fl = raw['fluid']
    th = raw['thermodynamics']
    hx = raw['heat_exchangers']
    pb = raw['packed_bed']
    sh = raw['shaft']
    ic = raw['inventory_control']
    mo = raw['motor']
    si = raw['simulation']

    # ── run identity ─────────────────────────────────────────────
    m.RUN_ID = raw.get('run_id', 'unnamed')

    # ── fluid ────────────────────────────────────────────────────
    m.FLUID    = fl['name']
    m.FLUID_FB = fl['fallback']
    m.GAMMA    = fl['gamma']
    m.R_GAS    = fl['R_gas']
    m.CP_GAS   = fl['cp_gas']
    m.KAPPA    = (fl['gamma'] - 1.0) / fl['gamma']

    # ── thermodynamics ───────────────────────────────────────────
    m.T_MAX_K      = th['T_max_K']
    m.T_MIN_K      = th['T_min_K']
    m.T_ENV_K      = th['T_env_K']
    m.P_LOW_PA     = th['P_low_Pa']
    m.P_HIGH_PA    = th['P_high_Pa']
    m.PI_DESIGN    = th['P_high_Pa'] / th['P_low_Pa']
    m.ETA_POLY_C   = th['eta_poly_c']
    m.ETA_POLY_E   = th['eta_poly_e']
    m.M_DOT_DESIGN = th['m_dot_design']

    # ── heat exchangers ──────────────────────────────────────────
    m.EPS_RE = hx['eps_re']
    m.FP_RE  = hx['fp_re']
    m.FP_EX  = hx['fp_ex']
    m.DT_HR  = hx['dT_HR']
    m.DT_CR  = hx['dT_CR']
    m.DP_HR  = hx['dP_HR']
    m.DP_CR  = hx['dP_CR']

    # ── packed bed ───────────────────────────────────────────────
    m.RHO_SOLID        = pb['rho_solid']
    m.CP_SOLID         = pb['cp_solid']
    m.LAM_SOLID        = pb['lam_solid']
    m.EPS_BED          = pb['eps_bed']
    m.D_PART           = pb['d_part']
    m.L_TANK           = pb['L_tank']
    m.D_TANK           = pb['D_tank']
    m.A_CROSS          = math.pi * pb['D_tank']**2 / 4.0
    m.V_TANK           = math.pi * pb['D_tank']**2 / 4.0 * pb['L_tank']
    m.MU_GAS           = pb['mu_gas']
    m.LAM_GAS          = pb['lam_gas']
    m.PR_GAS           = pb['Pr_gas']
    m.N_X              = pb['N_x']
    m.DX               = pb['L_tank'] / pb['N_x']
    m.TAU_HOT_STORAGE  = pb['tau_hot_storage']
    m.TAU_COLD_STORAGE = pb['tau_cold_storage']

    # ── shaft ────────────────────────────────────────────────────
    m.N_DESIGN_RPM  = sh['N_design_rpm']
    m.OMEGA_DESIGN  = sh['N_design_rpm'] * 2.0 * math.pi / 60.0
    m.P_MOTOR_RATED = sh['P_motor_rated']
    m.TAU_SHAFT     = sh['tau_shaft']
    m.J_ROTOR       = sh['tau_shaft'] * sh['P_motor_rated'] / (sh['N_design_rpm'] * 2*math.pi/60)**2
    m.K_FRIC        = sh['K_fric']
    m.K_OL          = sh['K_ol']
    m.TAU_RATED     = sh['P_motor_rated'] / (sh['N_design_rpm'] * 2*math.pi/60)
    m.TAU_MAX       = sh['K_ol'] * sh['P_motor_rated'] / (sh['N_design_rpm'] * 2*math.pi/60)
    m.V_DEAD_COMP   = sh['V_dead_comp']
    m.V_DEAD_EXP    = sh['V_dead_exp']
    m.V_PIPE        = sh['V_pipe']
    m.V_SYS_TOTAL   = sh['V_dead_comp'] + sh['V_dead_exp'] + sh['V_pipe']

    # ── inventory control ────────────────────────────────────────
    m.INV_MODE  = ic['mode']
    m.ALPHA_MIN = ic['alpha_min']
    m.ALPHA_MAX = ic['alpha_max']
    m.TAU_INV   = ic['tau_inv']
    m.TAU_GOV   = ic['tau_gov']
    m.TAU_VOL   = ic['tau_vol']
    m.A_PI      = ic['a_pi']
    m.V_INV     = ic['V_inv']
    m.INV_KP    = ic['Kp']
    m.INV_KI    = ic['Ki']
    m.INV_KD    = ic['Kd']
    m.INV_TAU_F = ic['tau_f']
    m.K_GOV     = ic['K_gov']

    # ── motor ────────────────────────────────────────────────────
    m.ETA_MOTOR = mo['eta_motor']
    m.ETA_GEN   = mo['eta_gen']

    # ── simulation ───────────────────────────────────────────────
    m.T_STEP         = si['t_step']
    m.T_TOTAL        = si['t_total']
    m.DT             = si['dt']
    m.DELTA_PCT_LIST = si['delta_pct_list']

    # ── legacy aliases (backward compat with existing module code) ─
    m.PID_KP         = ic['Kp']
    m.PID_KI         = ic['Ki']
    m.PID_KD         = ic['Kd']
    m.PID_TAU_FILTER = ic['tau_f']


def cfg_path() -> str:
    """Return the path of the currently loaded config file."""
    return _cfg_path


def cfg_dict() -> dict:
    """Return the raw config dict."""
    return _cfg_dict


# ── Auto-load default on import ───────────────────────────────────
_DEFAULT = os.path.join(os.path.dirname(__file__), 'configs', 'default.json')
if os.path.exists(_DEFAULT):
    load(_DEFAULT)
else:
    # Fallback: hardcoded defaults (keeps backward compat if configs/ missing)
    import numpy as _np
    RUN_ID = 'fallback'
    FLUID = "REFPROP::Nitrogen"; FLUID_FB = "Nitrogen"
    GAMMA = 1.4; R_GAS = 296.8; CP_GAS = 1039.0; KAPPA = 0.2857
    T_MAX_K = 700.0; T_MIN_K = 200.0; T_ENV_K = 293.15
    P_LOW_PA = 3e5; P_HIGH_PA = 30e5; PI_DESIGN = 10.0
    ETA_POLY_C = 0.88; ETA_POLY_E = 0.88; M_DOT_DESIGN = 2.0
    EPS_RE = 0.95; FP_RE = 0.01; FP_EX = 0.001
    DT_HR = 10.0; DT_CR = 10.0; DP_HR = 200.0; DP_CR = 200.0
    RHO_SOLID = 2600.0; CP_SOLID = 900.0; LAM_SOLID = 2.0
    EPS_BED = 0.38; D_PART = 0.025; L_TANK = 5.0; D_TANK = 1.2
    A_CROSS = _np.pi*1.2**2/4; V_TANK = A_CROSS*5.0
    MU_GAS = 2e-5; LAM_GAS = 0.032; PR_GAS = 0.71; N_X = 50; DX = 0.1
    TAU_HOT_STORAGE = 89.0; TAU_COLD_STORAGE = 104.0
    N_DESIGN_RPM = 3000.0; OMEGA_DESIGN = 314.159; P_MOTOR_RATED = 500e3
    TAU_SHAFT = 62.0; J_ROTOR = 314.1; K_FRIC = 0.012; K_OL = 1.5
    TAU_RATED = 1591.5; TAU_MAX = 2387.3
    V_DEAD_COMP = 0.05; V_DEAD_EXP = 0.05; V_PIPE = 0.10; V_SYS_TOTAL = 0.20
    INV_MODE = 'pi'; ALPHA_MIN = 0.20; ALPHA_MAX = 1.50
    TAU_INV = 30.0; TAU_GOV = 0.5; TAU_VOL = 5.0; A_PI = 0.85
    V_INV = 0.5; INV_KP = 6.0; INV_KI = 0.2; INV_KD = 0.0; INV_TAU_F = 5.0
    K_GOV = 2000.0; ETA_MOTOR = 0.97; ETA_GEN = 0.97
    T_STEP = 20.0; T_TOTAL = 700.0; DT = 0.5; DELTA_PCT_LIST = [5.0, 50.0]
    PID_KP = 6.0; PID_KI = 0.2; PID_KD = 0.0; PID_TAU_FILTER = 5.0


if __name__ == '__main__':
    print(f"Config loaded: {_cfg_path}")
    print(f"  run_id      = {RUN_ID}")
    print(f"  inv_mode    = {INV_MODE}")
    print(f"  P_lo/P_hi   = {P_LOW_PA/1e5:.0f}/{P_HIGH_PA/1e5:.0f} bar  π={PI_DESIGN:.0f}")
    print(f"  J_rotor     = {J_ROTOR:.1f} kg·m²")
    print(f"  K_gov       = {K_GOV:.0f} W/rpm")
    print(f"  delta_pct   = {DELTA_PCT_LIST}")
