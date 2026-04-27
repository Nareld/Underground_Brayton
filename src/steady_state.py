"""
steady_state.py — Main Script A: Quasi-static Thermodynamic Solver
═══════════════════════════════════════════════════════════════════
Purpose
───────
Given a set of fixed parameters (T_MAX, T_MIN, π, η_p), iteratively
solve for all remaining parameters that close the thermodynamic cycle.

Algorithm
─────────
1. Load config.py design-point parameters.
2. Call charging_cycle_v3() (inner 30-iteration fixed-point) — reuses
   existing REFPROP/CoolProp physics from brayton_v3_physics.py.
3. Call discharging_cycle_v3() for the reverse (heat-engine) cycle.
4. Compute performance metrics: χ (round-trip eff.), ρ_E, ρ_P (Du2025).
5. Save results to /results/steady_state_results.json.
6. Generate T-s diagram via _Ts_plot.py → /results/fig_steady_state_Ts.png.
7. Generate operating-point parameter table → /results/fig_steady_state_params.png.

Outputs
───────
  /results/steady_state_results.json
  /results/fig_steady_state_Ts.png
  /results/fig_steady_state_params.png
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti TC',
                                       'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
_RES   = os.path.join(_ROOT, 'results')   # /Underground_Brayton/results/
os.makedirs(_RES, exist_ok=True)

sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import config as C
from _compressor_turbine import CompressorTurbine
from _hot_storage import PackedBedTank
from _cold_storage import make_cold_tank
from _motor_generator import MotorGenerator

from brayton_v3_physics import (
    charging_cycle_v3,
    discharging_cycle_v3,
    compute_performance_v3,
)
from _Ts_plot import plot_Ts

# ═════════════════════════════════════════════════════════════════════════
# 1. Solve steady-state cycles
# ═════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Steady-State Thermodynamic Solver")
print(f"  Working fluid : {C.FLUID}")
print(f"  π             : {C.PI_DESIGN:.0f}  "
      f"({C.P_LOW_PA/1e5:.0f}→{C.P_HIGH_PA/1e5:.0f} bar)")
print(f"  T_min / T_max : {C.T_MIN_K:.0f} / {C.T_MAX_K:.0f} K")
print(f"  η_p (comp/exp): {C.ETA_POLY_C:.2f} / {C.ETA_POLY_E:.2f}")
print("=" * 60)

# Charging cycle (heat-pump mode)
print("\n[1/2] Solving charging cycle (heat-pump)...")
ch = charging_cycle_v3(P_scale=1.0,
                        T_HR_out=C.T_ENV_K,
                        T_CR_out=C.T_ENV_K,
                        m_dot=C.M_DOT_DESIGN)

print(f"  State points (charging):")
print(f"    1→2 (compressor): {ch['T1']-273.15:.1f}°C → {ch['T2']-273.15:.1f}°C")
print(f"    2→3 (hot HX):     {ch['T2']-273.15:.1f}°C → {ch['T3']-273.15:.1f}°C")
print(f"    3→7 (recuperator HP): {ch['T3']-273.15:.1f}°C → {ch['T7']-273.15:.1f}°C")
print(f"    7→5 (expander):   {ch['T7']-273.15:.1f}°C → {ch['T5']-273.15:.1f}°C")
print(f"    5→6 (cold HX):    {ch['T5']-273.15:.1f}°C → {ch['T6']-273.15:.1f}°C")
print(f"    6→1 (recuperator LP): {ch['T6']-273.15:.1f}°C → {ch['T1']-273.15:.1f}°C")
print(f"  W_comp = {ch['W_comp']/1e3:.2f} kW   W_exp = {ch['W_exp']/1e3:.2f} kW")
print(f"  W_net  = {ch['W_net']/1e3:.2f} kW   Q_hot = {ch['Q_hot']/1e3:.2f} kW")
print(f"  β_c    = {ch['beta_c']:.3f}   β_e = {ch['beta_e']:.3f}")
print(f"  ṁ      = {ch['m_dot']:.3f} kg/s   V̇ = {ch['V_dot']*1e3:.1f} L/s")

# Discharging cycle (heat-engine mode)
print("\n[2/2] Solving discharging cycle (heat-engine)...")
dis = discharging_cycle_v3(P_scale=1.0,
                            T_HR_out=C.T_MAX_K,
                            T_CR_out=C.T_MIN_K,
                            m_dot=C.M_DOT_DESIGN)

print(f"  State points (discharging):")
print(f"    D1→D2 (turbine):  {dis['T_D1']-273.15:.1f}°C → {dis['T_D2']-273.15:.1f}°C")
print(f"    D2→D3 (cold HX):  {dis['T_D2']-273.15:.1f}°C → {dis['T_D3']-273.15:.1f}°C")
print(f"    D3→D4 (compressor): {dis['T_D3']-273.15:.1f}°C → {dis['T_D4']-273.15:.1f}°C")
print(f"    D4→D5 (hot HX):   {dis['T_D4']-273.15:.1f}°C → {dis['T_D5']-273.15:.1f}°C")
print(f"  W_turb = {dis['W_turb']/1e3:.2f} kW   W_comp = {dis['W_comp']/1e3:.2f} kW")
print(f"  W_net  = {dis['W_net']/1e3:.2f} kW")

# Performance metrics (Du2025)
print("\n[Perf] Computing Du2025 performance metrics...")
pf = compute_performance_v3(ch, dis, V_hot=C.V_TANK, V_cold=C.V_TANK)
print(f"  Round-trip eff. χ = {pf['chi']*100:.1f}%")
print(f"  Energy density ρ_E = {pf['rho_E']:.4f} kWh/m³")
print(f"  Power density  ρ_P = {pf['rho_P']:.1f} kW/(m³/s)")

# Motor/generator balance
mg = MotorGenerator(mode='charge')
mg.calibrate(ch['W_comp'], ch['W_exp'], ch['V_dot'])
W_fric = mg.friction_power(C.OMEGA_DESIGN)
print(f"\n  W_fric (shaft) = {W_fric/1e3:.3f} kW")
print(f"  P_cmd0 (motor) = {mg._P_cmd0/1e3:.2f} kW")

# ═════════════════════════════════════════════════════════════════════════
# 2. Calibrate CompressorTurbine module
# ═════════════════════════════════════════════════════════════════════════
ct = CompressorTurbine()
ct.calibrate(ch['T1'], ch['T7'], C.P_LOW_PA, C.P_HIGH_PA, C.M_DOT_DESIGN)
print(f"\n  F_cal_C = {ct._F_cal_C:.4f}   F_cal_E = {ct._F_cal_E:.4f}")

# ═════════════════════════════════════════════════════════════════════════
# 3. Save results JSON
# ═════════════════════════════════════════════════════════════════════════
results = {
    "config": {
        "fluid": C.FLUID,
        "P_low_bar":  C.P_LOW_PA  / 1e5,
        "P_high_bar": C.P_HIGH_PA / 1e5,
        "pi_design":  C.PI_DESIGN,
        "T_min_K":    C.T_MIN_K,
        "T_max_K":    C.T_MAX_K,
        "T_env_K":    C.T_ENV_K,
        "eta_poly_c": C.ETA_POLY_C,
        "eta_poly_e": C.ETA_POLY_E,
        "m_dot_design_kg_s": C.M_DOT_DESIGN,
    },
    "charging": {
        "T1_C": ch['T1'] - 273.15,
        "T2_C": ch['T2'] - 273.15,
        "T3_C": ch['T3'] - 273.15,
        "T7_C": ch['T7'] - 273.15,
        "T5_C": ch['T5'] - 273.15,
        "T6_C": ch['T6'] - 273.15,
        "P_lo_bar": ch['P_lo'] / 1e5,
        "P_hi_bar": ch['P_hi'] / 1e5,
        "beta_c": ch['beta_c'],
        "beta_e": ch['beta_e'],
        "W_comp_kW": ch['W_comp'] / 1e3,
        "W_exp_kW":  ch['W_exp']  / 1e3,
        "W_net_kW":  ch['W_net']  / 1e3,
        "Q_hot_kW":  ch['Q_hot']  / 1e3,
        "Q_cold_kW": ch['Q_cold'] / 1e3,
        "V_dot_L_s": ch['V_dot']  * 1e3,
    },
    "discharging": {
        "T_D1_C": dis['T_D1'] - 273.15,
        "T_D2_C": dis['T_D2'] - 273.15,
        "T_D3_C": dis['T_D3'] - 273.15,
        "T_D4_C": dis['T_D4'] - 273.15,
        "T_D5_C": dis['T_D5'] - 273.15,
        "W_turb_kW": dis['W_turb'] / 1e3,
        "W_comp_kW": dis['W_comp'] / 1e3,
        "W_net_kW":  dis['W_net']  / 1e3,
    },
    "performance": {
        "chi_pct":    pf['chi']   * 100.0,
        "rho_E_kWh_m3": pf['rho_E'],
        "rho_P_kW_m3s":  pf['rho_P'],
    },
    "motor": {
        "P_cmd0_kW": mg._P_cmd0 / 1e3,
        "W_fric_kW": W_fric / 1e3,
        "F_cal_C":   ct._F_cal_C,
        "F_cal_E":   ct._F_cal_E,
    },
}

json_path = os.path.join(_RES, 'steady_state_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved → {json_path}")

# ═════════════════════════════════════════════════════════════════════════
# 4. T-s Diagram  (via _Ts_plot.py)
# ═════════════════════════════════════════════════════════════════════════
ts_path = os.path.join(_RES, 'fig_steady_state_Ts.png')
plot_Ts(ch, dis, ts_path)

# ═════════════════════════════════════════════════════════════════════════
# 5. Parameter Summary Figure
# ═════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 6))
fig2.patch.set_facecolor('#F8F9FA')

# ── Panel 1: State-point temperatures ───────────────────────────────────
ax = axes2[0]
labels_ch = ['T1\n(comp in)', 'T2\n(comp out)', 'T3\n(hot HX out)',
              'T7\n(turb in)', 'T5\n(exp out)', 'T6\n(cold HX out)']
temps_ch  = [ch['T1']-273.15, ch['T2']-273.15, ch['T3']-273.15,
             ch['T7']-273.15, ch['T5']-273.15, ch['T6']-273.15]
colors_ch = ['#4A90D9', '#CC3300', '#CC3300', '#1A3E6F', '#00509E', '#00509E']
bars = ax.bar(labels_ch, temps_ch, color=colors_ch, alpha=0.82, edgecolor='white', lw=0.8)
for bar, val in zip(bars, temps_ch):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.0f}°C', ha='center', fontsize=8.5, fontweight='bold')
ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax.set_ylabel('Temperature [°C]', fontsize=10)
ax.set_title('Charging Cycle State Points', fontsize=10.5, fontweight='bold')
ax.set_facecolor('#FAFAFA')
ax.grid(axis='y', lw=0.35, alpha=0.45)

# ── Panel 2: Power flows ─────────────────────────────────────────────────
ax = axes2[1]
power_labels = ['W_comp\n(charge)', 'W_exp\n(charge)', 'W_net\n(charge)',
                'W_turb\n(disch.)', 'W_comp\n(disch.)', 'W_net\n(disch.)']
power_vals   = [ch['W_comp']/1e3, ch['W_exp']/1e3, ch['W_net']/1e3,
                dis['W_turb']/1e3, dis['W_comp']/1e3, dis['W_net']/1e3]
power_clrs   = ['#CC3300', '#1A3E6F', '#228B22',
                '#CC3300', '#1A3E6F', '#228B22']
bars2 = ax.bar(power_labels, power_vals, color=power_clrs, alpha=0.82, edgecolor='white')
for bar, val in zip(bars2, power_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}', ha='center', fontsize=8.5, fontweight='bold')
ax.set_ylabel('Power [kW]', fontsize=10)
ax.set_title('Power Flows (Charge / Discharge)', fontsize=10.5, fontweight='bold')
ax.set_facecolor('#FAFAFA')
ax.grid(axis='y', lw=0.35, alpha=0.45)

# ── Panel 3: Performance metrics + text summary ──────────────────────────
ax = axes2[2]
ax.axis('off')
summary = (
    "Design-Point Summary\n"
    "══════════════════════════════\n"
    f"Working fluid : N₂ (REFPROP)\n"
    f"Pressure ratio π = {C.PI_DESIGN:.0f}  "
    f"({C.P_LOW_PA/1e5:.0f}→{C.P_HIGH_PA/1e5:.0f} bar)\n"
    f"Temp. range : {C.T_MIN_K:.0f}–{C.T_MAX_K:.0f} K\n"
    f"η_p (C/E) : {C.ETA_POLY_C:.2f} / {C.ETA_POLY_E:.2f}\n"
    f"ṁ_design : {C.M_DOT_DESIGN:.2f} kg/s\n"
    f"V̇_design : {ch['V_dot']*1e3:.1f} L/s\n"
    "──────────────────────────────\n"
    f"Charging  W_net : {ch['W_net']/1e3:.2f} kW\n"
    f"Discharge W_net : {dis['W_net']/1e3:.2f} kW\n"
    f"Round-trip  χ   : {pf['chi']*100:.1f}%\n"
    f"Energy dens ρ_E : {pf['rho_E']:.4f} kWh/m³\n"
    f"Power dens  ρ_P : {pf['rho_P']:.1f} kW/(m³/s)\n"
    "──────────────────────────────\n"
    f"n_design : {C.N_DESIGN_RPM:.0f} rpm\n"
    f"J_rotor  : {C.J_ROTOR:.1f} kg·m²\n"
    f"  (τ_shaft={C.TAU_SHAFT:.0f}s, Zhang2020)\n"
    f"P_motor0 : {mg._P_cmd0/1e3:.2f} kW\n"
)
ax.text(0.05, 0.97, summary, transform=ax.transAxes,
        fontsize=9.5, va='top', family='monospace',
        bbox=dict(boxstyle='round', fc='#EAF0FB', ec='#1A3E6F', alpha=0.95))
ax.set_title('System Overview', fontsize=10.5, fontweight='bold')

fig2.suptitle('PTES Steady-State Operating Parameters', fontsize=13, fontweight='bold')
plt.tight_layout()

params_path = os.path.join(_RES, 'fig_steady_state_params.png')
plt.savefig(params_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Parameter figure saved → {params_path}")

print("\n✓ Steady-state analysis complete.")
print(f"  All results in: {_RES}/")
