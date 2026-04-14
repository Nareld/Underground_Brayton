"""
brayton_v3_config_fixed.py  ——  修正版系统参数配置
修正内容：
  ① n_design: 15000 rpm → 3000 rpm（匹配Zhang2020 / 50Hz电网同步速度）
  ② 重新推导转动惯量 J（基于Zhang2020观测的动态响应时间常数）
  ③ 库存控制参数调整（匹配3000rpm系统）
"""
import numpy as np

# ── 工质 ────────────────────────────────────────────────────────────
FLUID    = "REFPROP::Nitrogen"
FLUID_FB = "Nitrogen"
GAMMA    = 1.4
R_GAS    = 296.8     # J/(kg·K) — N₂
CP_GAS   = 1039.0    # J/(kg·K)
KAPPA    = (GAMMA-1)/GAMMA   # 0.2857

# ── 热力学设计点 ────────────────────────────────────────────────────
T_MAX_K  = 700.0    # [K]
T_MIN_K  = 200.0    # [K]
T_ENV_K  = 293.15   # [K]
P_LOW_PA  = 3.0e5   # 3 bar
P_HIGH_PA = 30.0e5  # 30 bar
PI_DESIGN = P_HIGH_PA / P_LOW_PA   # = 10
ETA_POLY_C = 0.88
ETA_POLY_E = 0.88
EPS_RE     = 0.95
FP_RE      = 0.010
FP_EX      = 0.001
DT_HR      = 10.0
DT_CR      = 10.0
DP_HR      = 0.002e6
DP_CR      = 0.002e6
M_DOT_DESIGN = 2.0   # kg/s

# ── 填充床 ──────────────────────────────────────────────────────────
RHO_SOLID = 2600.0
CP_SOLID  = 900.0
LAM_SOLID = 2.0
EPS_BED   = 0.38
D_PART    = 0.025
L_TANK    = 5.0
D_TANK    = 1.2
A_CROSS   = np.pi * D_TANK**2 / 4
V_TANK    = A_CROSS * L_TANK
MU_GAS    = 2.0e-5
LAM_GAS   = 0.032
PR_GAS    = 0.71
N_X       = 50
DX        = L_TANK / N_X

# ════════════════════════════════════════════════════════════════════
# 轴系参数——修正版
# ════════════════════════════════════════════════════════════════════
#
# 修正说明：
#   原值 N_DESIGN_RPM = 15000 rpm 是直接硬编码，缺乏物理依据。
#
#   修正依据（Zhang2020 / 文献标准）：
#     - 50Hz电网同步转速：n = 3000 rpm（2极同步电机）
#     - Zhang2020明确记载："转速/（r·min⁻¹）3000"（Table 1）
#     - 大规模 PTES 系统通过增速齿轮箱使透平机械在更高转速运行，
#       但电机轴（并网侧）固定为 3000 rpm
#     - 本仿真建模电机轴侧，故采用 3000 rpm
#
#   转动惯量 J 的推导：
#     Zhang2020 观测：功率阶跃-5%后，转速在约 t=62s 达到最小值
#     → 机械响应时间常数 τ_mech ≈ 62s（从阶跃到最低点）
#     → 但注意：这个 62s 是整个系统热-机耦合的结果，不是纯轴系惯量
#     → 纯轴系时间常数 τ_shaft = J·ω / P_rated
#     → Zhang2020（150MW, 3000rpm）：τ_shaft ≈ 1–5s（轴系本身）
#     → 流量调节时间常数约30s（库存控制外层）才是主要延迟
#
#     对于我们的系统（~500kW，3000rpm）：
#       P_rated = 500 kW
#       ω = 314.16 rad/s
#       τ_shaft目标 = 2s（与文献同量级）
#       J = τ_shaft × P_rated / ω² = 2 × 500000 / 314.16² ≈ 10 kg·m²
#
N_DESIGN_RPM  = 3000.0      # [rpm] — 50Hz电网同步速度（Zhang2020 Table 1）
OMEGA_DESIGN  = N_DESIGN_RPM * 2 * np.pi / 60  # = 314.16 rad/s
P_MOTOR_RATED = 500e3       # [W] — 须 ≥ W_net_charge

# J 由目标时间常数推导：τ_shaft = J·ω²/P = 2s
TAU_SHAFT_TARGET = 2.0      # [s] — 目标轴系时间常数
J_ROTOR = TAU_SHAFT_TARGET * P_MOTOR_RATED / OMEGA_DESIGN**2
# J = 2.0 × 500000 / 314.16² = 10.13 kg·m²

K_FRIC   = 0.012            # 摩擦系数（相对额定扭矩）
K_OL     = 1.5              # 过载系数
TAU_RATED = P_MOTOR_RATED / OMEGA_DESIGN
TAU_MAX   = K_OL * TAU_RATED

# ── 容积效应 ────────────────────────────────────────────────────────
V_DEAD_COMP = 0.05
V_DEAD_EXP  = 0.05
V_PIPE      = 0.10
V_SYS_TOTAL = V_DEAD_COMP + V_DEAD_EXP + V_PIPE

# ════════════════════════════════════════════════════════════════════
# 库存控制参数——修正版
# ════════════════════════════════════════════════════════════════════
#
# 正确的库存控制逻辑（McTigue 2024 §3.3.1 / 张谨奕 §3.1）：
#
#   核心原则：维持 体积流量 V̇ = 不变
#             即：折合流量系数 ṁ_c = ṁ√T/P = const
#
#   调节机制：
#     当功率目标变为 α × W_design：
#     ① 等比例调节系统压力：P_lo_new = α × P_lo,design
#                            P_hi_new = α × P_hi,design
#     ② 压比保持不变：π = P_hi/P_lo = const ✓
#     ③ 入口密度随压力缩放：ρ_new = α × ρ_design
#     ④ 质量流量缩放：ṁ_new = ρ_new × V̇_design = α × ṁ_design
#     ⑤ 体积流量不变：V̇ = ṁ_new/ρ_new = V̇_design = const ✓
#     ⑥ 功率线性跟随：W_net ≈ α × W_design ✓
#
#   电机控制：
#     快速跟踪机械平衡：P_motor = W_comp - W_exp + W_fric
#     使得转速始终≈n_design（小偏差由轴系ODE决定）
#
ALPHA_MIN    = 0.20
ALPHA_MAX    = 1.50
TAU_INV      = 30.0    # 库存控制时间常数（压力建立）[s]
TAU_GOV      = 0.5     # 调速器时间常数（电机功率调节）[s]  ← 修正：更快

# ── 仿真场景 ────────────────────────────────────────────────────────
ZH_T_STEP    = 100.0   # Zhang2020基准：100s后阶跃（稳态段足够）
ZH_STEP_FRAC = 0.95    # 降至95%
ZH_T_TOTAL   = 800.0

STEP_50_T    = 300.0   # 50%阶跃时刻
STEP_50_FRAC = 0.50
STEP_50_TOTAL= 1500.0

T_RAMP_S     = 60.0

if __name__ == '__main__':
    print("=== 修正后系统参数 ===")
    print(f"  n_design  = {N_DESIGN_RPM:.0f} rpm  (50Hz电网同步速度，Zhang2020 Table 1)")
    print(f"  ω_design  = {OMEGA_DESIGN:.2f} rad/s")
    print(f"  J_rotor   = {J_ROTOR:.2f} kg·m²  (τ_shaft = {TAU_SHAFT_TARGET}s)")
    print(f"  τ_rated   = {TAU_RATED:.1f} N·m")
    print(f"  τ_max     = {TAU_MAX:.1f} N·m")
    print(f"  τ_gov     = {TAU_GOV} s  (调速器响应)")
    print(f"  τ_inv     = {TAU_INV} s  (库存/压力建立)")
    print(f"  π         = {PI_DESIGN}")
    print(f"  T_shaft   = J·ω²/P = {J_ROTOR*OMEGA_DESIGN**2/P_MOTOR_RATED:.2f} s ✓")
