"""
brayton_v3_config.py — 完整动态建模系统参数配置
对标：Zhang2020（150 MW，Ar），Desrues2010（填充床），Du2025（N₂，双填充床）
"""
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 工质选择：本系统使用 N₂（REFPROP），兼容 Zhang2020/Desrues 的 Ar 参数对比
# ══════════════════════════════════════════════════════════════════════
FLUID   = "REFPROP::Nitrogen"
FLUID_FB= "Nitrogen"           # CoolProp HEOS 后备

# 理想气体参数（N₂）
GAMMA   = 1.4
R_GAS   = 296.8    # J/(kg·K)  — N₂ specific gas constant
CP_GAS  = 1039.0   # J/(kg·K)  — N₂ at ~400K
KAPPA   = (GAMMA-1)/GAMMA  # = 0.2857

# ══════════════════════════════════════════════════════════════════════
# 热力学设计点（对标 Du2025，N₂ + 填充床）
# ══════════════════════════════════════════════════════════════════════
T_MAX_K  = 700.0    # 压气机出口/热储最高温度 [K] (427°C)
T_MIN_K  = 200.0    # 透平出口/冷储最低温度 [K] (-73°C)
T_ENV_K  = 293.15   # 环境温度 [K] (20°C)

P_LOW_PA  = 3.0e5   # 低压侧 [Pa] = 3 bar
P_HIGH_PA = 30.0e5  # 高压侧 [Pa] = 30 bar
PI_DESIGN = P_HIGH_PA / P_LOW_PA   # = 10
PSI_DESIGN = PI_DESIGN**KAPPA       # = 10^0.2857 ≈ 1.931 (热压缩温比)

ETA_POLY_C = 0.88   # 压气机多变效率
ETA_POLY_E = 0.88   # 膨胀机多变效率
EPS_RE   = 0.95     # 回热器效能
FP_RE    = 0.010    # 回热器压损系数 (1%)
FP_EX    = 0.001    # 外部换热器压损 (0.1%)
DT_HR    = 10.0     # 热储端差 [K]  (Du2025)
DT_CR    = 10.0     # 冷储端差 [K]  (Du2025)
DP_HR    = 0.002e6  # 热储压降 [Pa] (Du2025)
DP_CR    = 0.002e6  # 冷储压降 [Pa]

M_DOT_DESIGN = 2.0  # 设计质量流量 [kg/s]

# ══════════════════════════════════════════════════════════════════════
# 填充床参数（Basalt）
# ══════════════════════════════════════════════════════════════════════
RHO_SOLID  = 2600.0    # 固体密度 [kg/m³]
CP_SOLID   = 900.0     # 固体比热容 [J/(kg·K)]
LAM_SOLID  = 2.0       # 固体导热系数 [W/(m·K)]
EPS_BED    = 0.38      # 孔隙率 [-]
D_PART     = 0.025     # 颗粒直径 [m]
L_TANK     = 5.0       # 储罐长度 [m]
D_TANK     = 1.2       # 储罐直径 [m]
A_CROSS    = np.pi * D_TANK**2 / 4   # 截面积 [m²]
V_TANK     = A_CROSS * L_TANK         # 单个储罐体积 [m³]

# 气体物性（用于 Ergun 方程）
MU_GAS     = 2.0e-5    # 动力黏度 [Pa·s]（N₂ @ 400K）
LAM_GAS    = 0.032     # 导热系数 [W/(m·K)]
PR_GAS     = 0.71      # 普朗特数

# ══════════════════════════════════════════════════════════════════════
# 机械系统（轴系）参数 — Zhang2020 Eq.11
# ══════════════════════════════════════════════════════════════════════
N_DESIGN_RPM = 15000.0  # 设计转速 [rpm]
OMEGA_DESIGN = N_DESIGN_RPM * 2 * np.pi / 60  # 角速度 [rad/s]
J_ROTOR   = 2.0         # 转子转动惯量 [kg·m²]
K_FRIC    = 0.012       # 摩擦系数 [-]（相对额定扭矩）
K_OL      = 1.5         # 电机过载系数（1.5×额定扭矩）
P_MOTOR_RATED = 500e3   # 电机额定功率 [W]（须 >= W_net_charge ≈ 460 kW）

# 额定扭矩
TAU_RATED = P_MOTOR_RATED / OMEGA_DESIGN  # [N·m]
TAU_MAX   = K_OL * TAU_RATED              # [N·m]
TAU_FRIC  = K_FRIC * TAU_RATED           # [N·m]（额定转速时）

# ══════════════════════════════════════════════════════════════════════
# 容积效应参数 — Zhang2020 Eq.7
# ══════════════════════════════════════════════════════════════════════
V_DEAD_COMP = 0.05   # 压气机死体积 [m³]
V_DEAD_EXP  = 0.05   # 透平死体积 [m³]
V_PIPE      = 0.10   # 管路总容积 [m³]
V_SYS_TOTAL = V_DEAD_COMP + V_DEAD_EXP + V_PIPE  # 总死体积 [m³]

# ══════════════════════════════════════════════════════════════════════
# 库存控制参数 — McTigue2024 §3.3.1
# ══════════════════════════════════════════════════════════════════════
V_DOT_DESIGN = None     # 由初始状态计算（ṁ₀/ρ₀）[m³/s]
ALPHA_MIN  = 0.20       # 最小压力比例（安全下限）
ALPHA_MAX  = 1.50       # 最大压力比例（耐压上限）

# PID 控制器参数
PID_KP = 0.50           # 比例增益
PID_KI = 0.05           # 积分增益
PID_KD = 0.10           # 微分增益
PID_TAU_FILTER = 5.0    # 微分滤波时间常数 [s]

# ══════════════════════════════════════════════════════════════════════
# 数值求解参数
# ══════════════════════════════════════════════════════════════════════
N_X      = 50           # 空间节点数
DX       = L_TANK / N_X # 空间步长 [m]
# CFL 时间步长由运行时计算（取决于流速）

# ══════════════════════════════════════════════════════════════════════
# 仿真场景参数
# ══════════════════════════════════════════════════════════════════════
# Phase 3A — Zhang2020 对标（功率降 5%）
ZH_POWER_STEP  = 0.95   # 功率降至 95%
ZH_T_STEP      = 20.0   # 阶跃时刻 [s]
ZH_T_TOTAL     = 800.0  # 总仿真时长 [s]

# Phase 3B — 50% 功率阶跃（库存控制展示）
STEP_50_RATIO  = 0.50   # 功率降至 50%
STEP_50_T_STEP = 300.0  # 阶跃时刻 [s]
STEP_50_T_TOTAL= 1200.0 # 总仿真时长 [s]
T_RAMP_S       = 60.0   # 软启动时间 [s]

if __name__ == '__main__':
    print("=== 系统参数验证 ===")
    print(f"  PI_DESIGN  = {PI_DESIGN:.1f}")
    print(f"  PSI_DESIGN = {PSI_DESIGN:.4f}（热压缩温比）")
    print(f"  κ = {KAPPA:.4f}")
    print(f"  τ = T_MAX/T_MIN = {T_MAX_K/T_MIN_K:.3f}")
    print(f"  V_TANK = {V_TANK:.3f} m³")
    print(f"  A_CROSS = {A_CROSS:.4f} m²")
    print(f"  OMEGA_DESIGN = {OMEGA_DESIGN:.2f} rad/s")
    print(f"  TAU_RATED = {TAU_RATED:.2f} N·m")
    print(f"  TAU_MAX   = {TAU_MAX:.2f} N·m")
    print(f"  TAU_FRIC  = {TAU_FRIC:.2f} N·m")

    # 量纲检验预览
    rho0 = P_LOW_PA / (R_GAS * T_ENV_K)
    u_s0 = M_DOT_DESIGN / (rho0 * A_CROSS)
    u_int= u_s0 / EPS_BED
    Re_p = rho0 * u_s0 * D_PART / MU_GAS
    Nu_p = 2 + 1.1 * Re_p**0.6 * PR_GAS**(1/3)
    h_p  = Nu_p * LAM_GAS / D_PART
    h_v  = 6*(1-EPS_BED)/D_PART * h_p
    Bi   = h_p * (D_PART/6) / LAM_SOLID
    dt_cfl = 0.8 * DX / u_int

    print(f"\n  ρ₀ = {rho0:.3f} kg/m³")
    print(f"  u_s = {u_s0:.4f} m/s")
    print(f"  u_int = {u_int:.4f} m/s")
    print(f"  Re_p = {Re_p:.1f}")
    print(f"  Nu_p = {Nu_p:.2f}")
    print(f"  h_v  = {h_v:.1f} W/(m³·K)")
    print(f"  Bi   = {Bi:.4f}  {'✓ < 0.1' if Bi<0.1 else '✗ > 0.1 !'}")
    print(f"  dt_CFL = {dt_cfl:.4f} s")
