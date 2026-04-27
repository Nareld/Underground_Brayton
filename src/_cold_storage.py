"""
_cold_storage.py — Module 3: Low-Temperature Packed-Bed Tank
═════════════════════════════════════════════════════════════
Physical description
────────────────────
Charging : cold expanded gas (T5) flows in, cools the rock bed,
           slightly warmed gas exits toward compressor inlet (T1).
Discharging : reversed — warm gas from turbine outlet cools down
              through the bed, exits toward compressor inlet.

Governing equations: identical to hot storage (_hot_storage.py).
Differences vs hot tank:
  • Flow direction is reversed in charging (u_int < 0 convention)
  • Temperature range: 200–300 K instead of 300–700 K
  • Lag time constant: τ_cold = 104 s (Zhang2020 §1.1 table)
  • Approach temperature: +DT_CR (gas warms slightly, not cools)

This module instantiates PackedBedTank with tank_id='cold'.
No equations are re-implemented — all physics reuses _hot_storage.py.
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from _hot_storage import PackedBedTank


def make_cold_tank(T_init: float = C.T_ENV_K) -> PackedBedTank:
    """
    Factory: return a PackedBedTank configured for the cold storage role.

    Parameters
    ──────────
    T_init : initial uniform temperature [K]  (default: T_ENV_K ≈ 293 K)

    Usage
    ─────
    In charging, call pde_step with u_int < 0 (right→left flow)
    so that cold gas enters from the right and exits from the left.
    """
    return PackedBedTank(
        tank_id='cold',
        tau_lag=C.TAU_COLD_STORAGE,
        T_init=T_init,
    )


# Convenience alias — allows:  from _cold_storage import ColdTank
ColdTank = make_cold_tank
