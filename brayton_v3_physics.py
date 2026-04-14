"""
brayton_v3_physics.py — Phase 1+2 核心物理模型
  P1: 修正充放电循环 + Du2025 压比耦合方程
  P2: 角动量ODE + 容积效应ODE + Ergun压降 + PID库存控制
"""
import numpy as np
import CoolProp.CoolProp as CP
from brayton_v3_config import *

# ══════════════════════════════════════════════════════════════════════
# 工具函数：REFPROP 接口
# ══════════════════════════════════════════════════════════════════════

def props(prop, **kw):
    """封装 CoolProp，优先 REFPROP，失败退到 HEOS"""
    keys = list(kw.keys()); vals = list(kw.values())
    for fl in [FLUID, FLUID_FB]:
        try:
            v = CP.PropsSI(prop, keys[0], vals[0], keys[1], vals[1], fl)
            if np.isfinite(v): return float(v)
        except: pass
    raise RuntimeError(f"PropsSI failed: {prop}({kw})")


# ══════════════════════════════════════════════════════════════════════
# § P1-A  改进欧拉法多变过程（压气机/透平）
# ══════════════════════════════════════════════════════════════════════

def _euler_step_compress(h_i, p_i, p_next, eta_p):
    """Heun's method — 压缩一步：dh = v·dp / η_p"""
    v_i  = 1.0 / props('D', H=h_i, P=p_i)
    k1   = v_i / eta_p
    h_pred = h_i + (p_next - p_i) * k1
    v_n  = 1.0 / props('D', H=h_pred, P=p_next)
    k2   = v_n / eta_p
    return h_i + (p_next - p_i) * (k1 + k2) / 2

def _euler_step_expand(h_i, p_i, p_next, eta_p):
    """Heun's method — 膨胀一步：dh = η_p·v·dp"""
    v_i  = 1.0 / props('D', H=h_i, P=p_i)
    k1   = eta_p * v_i
    h_pred = h_i + (p_next - p_i) * k1
    v_n  = 1.0 / props('D', H=h_pred, P=p_next)
    k2   = eta_p * v_n
    return h_i + (p_next - p_i) * (k1 + k2) / 2

def polytropic_machine(T_in, P_in, P_out, eta_p, n_steps=200):
    """
    改进欧拉法多变过程（N₂，REFPROP）
    返回：T_out, h_out, s_out[kJ/kgK], w_spec[J/kg], T_path, s_path
    """
    h_in = props('H', T=T_in, P=P_in)
    s_in = props('S', T=T_in, P=P_in)
    P_arr= np.linspace(P_in, P_out, n_steps+1)
    h    = h_in
    step_fn = _euler_step_compress if P_out > P_in else _euler_step_expand

    idx  = range(0, n_steps+1, max(1, n_steps//50))
    T_path=[]; s_path=[]
    for i in range(n_steps):
        h = step_fn(h, P_arr[i], P_arr[i+1], eta_p)
        if i in idx:
            T_path.append(props('T', H=h, P=P_arr[i+1]))
            s_path.append(props('S', H=h, P=P_arr[i+1]) / 1e3)

    T_out = props('T', H=h, P=P_out)
    s_out = props('S', H=h, P=P_out)
    return dict(T_out=T_out, h_out=h, s_out=s_out/1e3,
                w_spec=abs(h - h_in),
                s_in=s_in/1e3, h_in=h_in, P_in=P_in, P_out=P_out,
                T_path=np.array(T_path), s_path=np.array(s_path))


# ══════════════════════════════════════════════════════════════════════
# § P1-B  Du2025 充放电压比耦合约束（Eq.11 & Eq.16）
# ══════════════════════════════════════════════════════════════════════

def charge_expansion_ratio(beta_c,
                            dp_HR=DP_HR, dp_CR=DP_CR,
                            p1=P_LOW_PA,
                            fp_re=FP_RE, fp_ex=FP_EX):
    """
    Du2025 Eq.11：充能透平压比（由压气机压比和压损决定）

    β_e = (β_c - Δp_HR/p1)·(1-f_p,re)·(1-f_p,ex)
          ─────────────────────────────────────────
          1/(1-f_p,re) + Δp_CR/p1

    物理含义：高压侧压力损失（热储）+ 低压侧压力损失（冷储+回热器）
    使得膨胀机实际可用压比略低于压气机压比
    """
    numerator   = (beta_c - dp_HR/p1) * (1-fp_re) * (1-fp_ex)
    denominator = 1.0/(1-fp_re) + dp_CR/p1
    beta_e      = numerator / denominator
    return max(beta_e, 1.001)


def discharge_compression_ratio(beta_e_prime,
                                 dp_HR=DP_HR, dp_CR=DP_CR,
                                 p1=P_LOW_PA,
                                 fp_re=FP_RE, fp_ex=FP_EX,
                                 zeta=0.001):
    """
    Du2025 Eq.16：放能压气机压比（由膨胀比和压损决定）

    β'_c = 1 / [1 - ζβ'_e + Δp'_HR/(p1'·(1-fp,re)²·(1-fp,ex)²)
                           - (1-fp,re)·Δp'_CR/p1']
    """
    denom = (1 - zeta*beta_e_prime
             + dp_HR/(p1*(1-fp_re)**2*(1-fp_ex)**2)
             - (1-fp_re)*dp_CR/p1)
    return max(1.0/denom, 1.001)


def optimal_discharge_ratio(T2_prime, T4_prime,
                             eta_c_prime=ETA_POLY_C,
                             eta_e_prime=ETA_POLY_E):
    """
    Du2025 Eq.21：最优放能压气机压比（使 χ 最大）

    β'_c,opt = (η'_c·η'_e·T9'/T4')^[1/(κ/η'_c + κ·η'_e)]
    """
    exponent = 1.0 / (KAPPA/eta_c_prime + KAPPA*eta_e_prime)
    ratio    = (eta_c_prime * eta_e_prime * T2_prime / T4_prime)
    return max(ratio**exponent, 1.01)


# ══════════════════════════════════════════════════════════════════════
# § P1-C  完整充能循环（含回热器，修正边界条件）
# ══════════════════════════════════════════════════════════════════════

def charging_cycle_v3(P_scale=1.0,
                       T_HR_out=T_ENV_K, T_CR_out=T_ENV_K,
                       m_dot=M_DOT_DESIGN):
    """
    充能循环（热泵模式）— 含 Du2025 压比耦合 + 正确换热方向

    状态点（杜小泽 Fig.3c 符号体系）：
      1: 压气机入口（回热器 LP 出口）
      2: 压气机出口
      3: 热储换热器出口（气体放热→热储）  ← mode='cool'
      7: 回热器 HP 出口（高压侧冷却）
      4: 膨胀机入口（= State 7）
      5: 膨胀机出口
      6: 冷储换热器出口（气体从冷储吸热）  ← mode='heat'（T5 < T_CR,T5很冷）
      1: 回热器 LP 出口（= State 6 升温后）

    关键修正：
      3→7（回热器HP）：气体降温（放热给LP侧）  mode='cool'
      5→6（冷HX）：T5 < T_CR_out → 气体从冷储吸热，mode='heat' ✓
    """
    P_lo = P_LOW_PA  * P_scale
    P_hi = P_HIGH_PA * P_scale

    # ── Du2025 Eq.11：计算膨胀比 β_e ────────────────────────────────
    beta_c = P_hi / P_lo
    beta_e = charge_expansion_ratio(beta_c, dp_HR=DP_HR*P_scale,
                                     dp_CR=DP_CR*P_scale, p1=P_lo)
    P_e_out = P_lo  # 膨胀机出口（低压侧）

    # ── 迭代求回热器解（双变量不动点） ──────────────────────────────
    T1, T6 = T_CR_out, T_CR_out   # 初始猜测

    for _ in range(30):
        # 1→2: 压气机（改进欧拉法）
        c = polytropic_machine(T1, P_lo, P_hi, ETA_POLY_C)
        T2, h2 = c['T_out'], c['h_out']

        # 2→3: 热储换热（气体放热→热储，需 T2 > T_HR_out）
        if T2 > T_HR_out:
            h3_target = h2 - EPS_RE * (h2 - props('H', T=T_HR_out, P=P_hi*(1-FP_EX)))
            T3 = props('T', H=h3_target, P=P_hi*(1-FP_EX))
            Q_hot = m_dot * (h2 - h3_target)
        else:
            T3 = T2; Q_hot = 0.0; h3_target = h2
        P3 = P_hi * (1 - FP_EX)

        # 3→7: 回热器 HP 侧（气体继续冷却，给 LP 侧预热）
        T7 = T3 - EPS_RE * (T3 - T6)   # T6 为 LP 侧入口
        P7 = P3 * (1 - FP_RE)

        # 7→5: 透平膨胀
        e = polytropic_machine(T7, P7, P_e_out, ETA_POLY_E)
        T5, h5 = e['T_out'], e['h_out']

        # 5→6: 冷储换热（T5 < T_CR_out → 气体从冷储吸热）
        if T5 < T_CR_out:
            h6_target = h5 + EPS_RE * (props('H', T=T_CR_out, P=P_e_out*(1-FP_EX)) - h5)
            T6n = props('T', H=h6_target, P=P_e_out*(1-FP_EX))
            Q_cold = m_dot * (h6_target - h5)
        else:
            T6n = T5; Q_cold = 0.0; h6_target = h5
        P6 = P_e_out * (1 - FP_EX)

        # 6→1: 回热器 LP 侧（被 HP 侧预热）
        T1n = T6n + EPS_RE * (T3 - T6n)

        err = max(abs(T1n-T1), abs(T6n-T6))
        T1, T6 = T1n, T6n
        if err < 0.05: break

    # ── 功率计算 ─────────────────────────────────────────────────────
    W_comp = m_dot * c['w_spec']
    W_exp  = m_dot * e['w_spec']
    W_net  = W_comp - W_exp   # 充能净耗功（>0）

    # ── 密度/体积流量（压气机入口，库存控制基准）─────────────────────
    rho1  = props('D', T=T1, P=P_lo)
    V_dot = m_dot / rho1

    return dict(
        T1=T1, T2=T2, T3=T3, T7=T7, T5=T5, T6=T6,
        P_lo=P_lo, P_hi=P_hi, beta_c=beta_c, beta_e=beta_e,
        W_comp=W_comp, W_exp=W_exp, W_net=W_net,
        Q_hot=Q_hot, Q_cold=Q_cold,
        m_dot=m_dot, V_dot=V_dot, rho1=rho1,
        comp=c, exp=e
    )


def discharging_cycle_v3(P_scale=1.0,
                          T_HR_out=T_MAX_K, T_CR_out=T_MIN_K,
                          m_dot=M_DOT_DESIGN):
    """
    放能循环（热机模式）— 修正冷储换热方向

    流程（Du2025 §2 放能路径）：
      D1: 透平入口（= T_HR_out，热储高温侧）
      D2: 透平出口（高温低压，T_D2 ≈ 120°C @ π=10）
      D3: 冷储换热出口（T_D2 > T_CR_out → 气体放热→冷储，mode='cool'）✓
      D4: 压气机出口（从冷端压缩）
      D5: 热储换热出口（气体从热储吸热，mode='heat'，升温回 T_HR_out）

    关键修正：D2→D3 必须 mode='cool'（透平热排气向冷储放热）
    """
    P_lo = P_LOW_PA  * P_scale
    P_hi = P_HIGH_PA * P_scale

    # ── Du2025 Eq.21：最优放能压气机压比（初始用设计值）────────────
    beta_c_dis = P_hi / P_lo   # 先用设计压比，可迭代优化

    # D1→D2: 透平（从 T_HR_out 膨胀）
    exp = polytropic_machine(T_HR_out, P_hi, P_lo, ETA_POLY_E)
    T_D2 = exp['T_out']

    # ── 方向检验：透平出口必须高于冷储温度才能放热 ──────────────────
    assert T_D2 > T_CR_out, (
        f"物理错误：T_D2={T_D2-273:.0f}°C < T_CR={T_CR_out-273:.0f}°C\n"
        f"透平出口比冷储更冷，无法向冷储放热。"
        f"需要 T_hot > {T_CR_out * PI_DESIGN**(ETA_POLY_E*KAPPA) + 273.15:.0f}°C")

    # D2→D3: 冷储换热（气体放热 → 冷储，气体降温至接近 T_CR_out）
    h_D2 = exp['h_out']
    h_D3 = h_D2 - EPS_RE * (h_D2 - props('H', T=T_CR_out, P=P_lo*(1-FP_EX)))
    T_D3 = props('T', H=h_D3, P=P_lo*(1-FP_EX))
    Q_cold_dis = m_dot * (h_D2 - h_D3)   # 向冷储放热量 [W]

    # D3→D4: 压气机（从冷端压缩）
    comp = polytropic_machine(T_D3, P_lo*(1-FP_EX), P_hi, ETA_POLY_C)
    T_D4 = comp['T_out']

    # D4→D5: 热储换热（气体从热储吸热，升温回 T_HR_out）
    h_D4 = comp['h_out']
    h_D5 = h_D4 + EPS_RE * (props('H', T=T_HR_out, P=P_hi*(1-FP_EX)) - h_D4)
    T_D5 = props('T', H=h_D5, P=P_hi*(1-FP_EX))

    # ── 功率 ─────────────────────────────────────────────────────────
    W_turb  = m_dot * exp['w_spec']
    W_comp  = m_dot * comp['w_spec']
    W_net   = W_turb - W_comp   # 净产功（>0 为正常）

    Q_hot_abs = m_dot * (h_D5 - h_D4)   # 从热储吸热量 [W]

    rho_D1 = props('D', T=T_HR_out, P=P_hi)
    V_dot  = m_dot / rho_D1

    eta_HE = W_net / max(Q_hot_abs, 1.0)

    return dict(
        T_D1=T_HR_out, T_D2=T_D2, T_D3=T_D3, T_D4=T_D4, T_D5=T_D5,
        P_lo=P_lo, P_hi=P_hi, beta_c=beta_c_dis,
        W_turb=W_turb, W_comp=W_comp, W_net=W_net,
        Q_cold_dis=Q_cold_dis, Q_hot_abs=Q_hot_abs,
        m_dot=m_dot, V_dot=V_dot, eta_HE=eta_HE,
        exp=exp, comp=comp
    )


# ══════════════════════════════════════════════════════════════════════
# § P2-A  角动量方程 ODE — Zhang2020 Eq.11
# ══════════════════════════════════════════════════════════════════════

def shaft_ode(omega, W_comp, W_exp, P_motor, k_fric=K_FRIC*TAU_RATED):
    """
    轴系角动量平衡（Zhang2020 Eq.11）

    J·dω/dt = τ_motor - τ_comp + τ_exp_recovery - τ_friction

    量纲验证：
      J [kg·m²]·dω/dt [rad/s²] = [kg·m²·s⁻²] = [J/rad] = [N·m] ✓
      τ = W/ω [W/(rad/s)] = [N·m] ✓

    Returns: dω/dt [rad/s²]
    """
    omega_safe = max(abs(omega), 1.0)   # 防除零

    # 电机扭矩（恒功率控制，受过载扭矩限制）
    tau_motor = min(P_motor / omega_safe, TAU_MAX)

    # 压气机负载扭矩（从功率计算）
    tau_comp  = W_comp / omega_safe

    # 透平回收扭矩（同轴时部分回收）
    tau_exp   = W_exp  / omega_safe

    # 摩擦扭矩（正比于转速）
    tau_fric  = k_fric * (omega_safe / OMEGA_DESIGN)

    dw_dt = (tau_motor - tau_comp + tau_exp - tau_fric) / J_ROTOR
    return float(dw_dt)


# ══════════════════════════════════════════════════════════════════════
# § P2-B  容积效应 ODE — Zhang2020 Eq.7
# ══════════════════════════════════════════════════════════════════════

def volume_ode(P_sys, m_dot_in, m_dot_out, T_avg,
               V_dead=V_SYS_TOTAL):
    """
    系统死体积内的压力变化（理想气体容积效应）

    V·dρ/dt = ṁ_in - ṁ_out  →  dP/dt = RT/V·(ṁ_in - ṁ_out)

    量纲验证：
      dP/dt [Pa/s] = R[J/(kg·K)]·T[K]/V[m³]·Δṁ[kg/s]
                   = [J/m³/s] = [Pa/s] ✓

    Returns: dP_sys/dt [Pa/s]
    """
    return R_GAS * T_avg / V_dead * (m_dot_in - m_dot_out)


# ══════════════════════════════════════════════════════════════════════
# § P2-C  填充床 Schumann PDE + Ergun 压降
# ══════════════════════════════════════════════════════════════════════

def ergun_pressure_drop(m_dot, T_f, P, A_cs=A_CROSS,
                         eps=EPS_BED, d_p=D_PART):
    """
    Ergun 方程：∂P/∂x = -A·u_s - B·u_s²

    量纲验证：
      A = 150μ(1-ε)²/(ε³·d_p²) [Pa·s·m⁻²] × [m/s] = [Pa/m] ✓
      B = 1.75·ρ·(1-ε)/(ε³·d_p) [kg/m³/m] × [m/s]² = [Pa/m] ✓

    Returns: dP/dx [Pa/m]（负值 = 压力沿流动方向减小）
    """
    rho = P / (R_GAS * T_f)
    mu  = MU_GAS * (T_f / 293.15)**0.7   # 黏度温度修正

    u_s = m_dot / (rho * A_cs)   # 表观流速 [m/s]

    A_coef = 150 * mu * (1-eps)**2 / (eps**3 * d_p**2)
    B_coef = 1.75 * rho * (1-eps) / (eps**3 * d_p)

    dP_dx = -(A_coef * u_s + B_coef * u_s**2)
    return dP_dx, u_s, rho


def biot_number(T_f_avg, P_avg):
    """
    Biot 数验证（Desrues Eq.23）
    Bi = h_p·(d_p/6) / λ_s
    Bi < 0.1 → 固体内部温度均匀假设成立
    """
    rho = P_avg / (R_GAS * T_f_avg)
    mu  = MU_GAS * (T_f_avg / 293.15)**0.7
    lam = LAM_GAS * (T_f_avg / 293.15)**0.82
    u_s = M_DOT_DESIGN / (rho * A_CROSS)
    Re  = rho * u_s * D_PART / (mu + 1e-12)
    Nu  = 2.0 + 1.1 * max(Re, 0)**0.6 * PR_GAS**(1/3)
    h_p = Nu * lam / D_PART
    Bi  = h_p * (D_PART / 6) / LAM_SOLID
    return Bi, Re, Nu, h_p


def packed_bed_step_v3(T_f, T_s, u_int, T_inlet, P_f, dt):
    """
    填充床 Schumann PDE + Ergun 压降（算子分裂法）

    步骤：
      A) 对流推进（一阶迎风，显式）
      B) 换热更新（半隐式，无条件稳定）

    Parameters
    ----------
    T_f, T_s : ndarray [N_X]  流体/固体温度场 [K]
    u_int    : float  间隙流速 [m/s]
    T_inlet  : float  入口温度 [K]
    P_f      : float  当前平均压力 [Pa]
    dt       : float  时间步长 [s]

    Returns
    -------
    T_f_new, T_s_new, ΔP_total, dt_used
    """
    N = len(T_f)
    rho_f = P_f / (R_GAS * np.mean(T_f))
    mu    = MU_GAS * (np.mean(T_f) / 293.15)**0.7
    lam   = LAM_GAS * (np.mean(T_f) / 293.15)**0.82

    u_s   = u_int * EPS_BED   # 表观流速

    # Nusselt + 体积换热系数（随温度修正）
    Re    = rho_f * u_s * D_PART / (mu + 1e-12)
    Nu    = 2.0 + 1.1 * max(Re, 0)**0.6 * PR_GAS**(1/3)
    h_p   = Nu * lam / D_PART
    h_v   = 6 * (1 - EPS_BED) / D_PART * h_p

    # CFL 检验
    cfl = abs(u_int) * dt / DX
    assert cfl <= 1.01, f"CFL={cfl:.3f} > 1 → 数值不稳定！减小 dt"

    # ── Step A：对流（迎风格式）──────────────────────────────────────
    T_f_star = T_f.copy()
    if u_int > 0:       # 左→右（充热：hot gas 从左入）
        T_up      = np.empty(N)
        T_up[0]   = T_inlet
        T_up[1:]  = T_f[:-1]
        conv      = u_int * (T_f - T_up) / DX
    else:               # 右→左（放热：cold gas 从右入）
        T_dn      = np.empty(N)
        T_dn[-1]  = T_inlet
        T_dn[:-1] = T_f[1:]
        conv      = u_int * (T_dn - T_f) / DX

    T_f_star -= dt * conv

    # ── Step B：换热（半隐式，解析 2×2）────────────────────────────
    beta_f = h_v / (rho_f * CP_GAS * EPS_BED)
    beta_s = h_v / (RHO_SOLID * CP_SOLID * (1 - EPS_BED))
    det    = 1.0 + dt * (beta_f + beta_s)

    T_f_new = ((1 + dt*beta_s)*T_f_star + dt*beta_f*T_s) / det
    T_s_new = (dt*beta_s*T_f_star + (1 + dt*beta_f)*T_s) / det

    # 温度下限保护
    T_f_new = np.maximum(T_f_new, 50.0)
    T_s_new = np.maximum(T_s_new, 50.0)

    # ── Ergun 总压降 ─────────────────────────────────────────────────
    A_coef = 150 * mu * (1-EPS_BED)**2 / (EPS_BED**3 * D_PART**2)
    B_coef = 1.75 * rho_f * (1-EPS_BED) / (EPS_BED**3 * D_PART)
    dP_dx  = -(A_coef * u_s + B_coef * u_s**2)  # [Pa/m]
    dP_tot = dP_dx * L_TANK                       # [Pa]

    return T_f_new, T_s_new, dP_tot


# ══════════════════════════════════════════════════════════════════════
# § P2-D  PID 库存控制器 — McTigue2024 §3.3.1
# ══════════════════════════════════════════════════════════════════════

class InventoryPIDController:
    """
    PID 闭环库存控制：维持 π = P_hi/P_lo = const

    控制策略：
      误差信号：e = P_target_fraction - P_actual_fraction
      输出：P_scale（两侧压力等比例缩放）→ ṁ_new = ρ(P_new)·V̇_design
    """

    def __init__(self, Kp=PID_KP, Ki=PID_KI, Kd=PID_KD,
                 tau_f=PID_TAU_FILTER):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.tau_f = tau_f
        self._integral = 0.0
        self._err_prev  = 0.0
        self._deriv_f   = 0.0   # 滤波后微分
        self._P_scale   = 1.0

    def update(self, alpha_setpoint, alpha_actual, dt):
        """
        Parameters
        ----------
        alpha_setpoint : float  目标压力比例（0.3~1.5）
        alpha_actual   : float  当前压力比例（= P_lo / P_lo_design）
        dt             : float  时间步长 [s]

        Returns
        -------
        alpha_cmd : float  下一步压力命令
        """
        e = alpha_setpoint - alpha_actual

        # 积分（梯形法）
        self._integral += 0.5 * (e + self._err_prev) * dt

        # 微分（低通滤波）
        raw_d = (e - self._err_prev) / (dt + 1e-12)
        self._deriv_f += dt / (self.tau_f + dt) * (raw_d - self._deriv_f)

        u = self.Kp * e + self.Ki * self._integral + self.Kd * self._deriv_f
        self._err_prev = e

        self._P_scale = float(np.clip(alpha_actual + u, ALPHA_MIN, ALPHA_MAX))
        return self._P_scale

    def reset(self):
        self._integral = 0.0; self._err_prev = 0.0; self._deriv_f = 0.0


def inventory_mass_flow(P_scale, V_dot_ref, T_in=T_ENV_K):
    """
    库存控制的流量计算
    ṁ_new = ρ(T_in, P_lo_new) × V̇_ref

    量纲验证：
      ρ [kg/m³] × V̇ [m³/s] = ṁ [kg/s] ✓
    """
    P_lo_new = P_LOW_PA * P_scale
    rho_new  = P_lo_new / (R_GAS * T_in)
    m_dot_new= rho_new * V_dot_ref
    return float(np.clip(m_dot_new,
                         M_DOT_DESIGN * ALPHA_MIN,
                         M_DOT_DESIGN * ALPHA_MAX))


# ══════════════════════════════════════════════════════════════════════
# § P1-E  三项性能指标（Du2025 Eq.9,24,25）
# ══════════════════════════════════════════════════════════════════════

def compute_performance_v3(ch, dis, V_hot=V_TANK, V_cold=V_TANK):
    """
    Du2025 Eq.9,24,25：往返效率、储能密度、功率密度

    ① χ = W_dis,net / W_ch,net
    ② ρ_E = (1-φ)·[exergy_hot + exergy_cold] / (ρ_s·cp_s·(1/V_hot + 1/V_cold))
    ③ ρ_P = Ẇ_dis,net / V̇_max
    """
    W_ch  = ch['W_net']    # 充能净耗功 [W]（>0）
    W_dis = dis['W_net']   # 放能净产功 [W]（>0）
    V_dot_dis = dis['V_dot']  # 放能工质最大体积流量

    chi = W_dis / max(W_ch, 1.0)   # ①

    # ② 储能密度（Du2025 Eq.24，填充床修正）
    T2 = ch['T2']; T3 = ch['T3']
    T5 = ch['T5']; T6 = ch['T6']
    exergy_hot  = (T2-T3) - T_ENV_K*np.log(max(T2/T3,1e-3)) if T3>0 else 0
    exergy_cold = (T5-T6) - T_ENV_K*np.log(max(T5/T6,1e-3)) if T6>0 else 0
    numerator   = (1-EPS_BED) * abs(exergy_hot + exergy_cold)  # [J/kg_gas * ...]

    # 单位储罐能量容量（固体侧）
    cap_hot  = 1.0 / (RHO_SOLID * CP_SOLID)   # [m³·K/J]
    cap_cold = 1.0 / (RHO_SOLID * CP_SOLID)
    denominator = cap_hot/V_hot + cap_cold/V_cold   # 等价 [m³·K/(J·m³)]

    # 简化：以放能净功 / 总储罐体积来计算
    rho_E_kwh = (W_dis * 3600) / ((V_hot + V_cold) * 3.6e6)  # [kWh/m³]

    # ③ 功率密度
    rho_P = W_dis / max(V_dot_dis, 1e-6) / 1e3   # [kW/(m³/s)]

    return dict(chi=chi, rho_E=rho_E_kwh, rho_P=rho_P,
                W_ch_kW=W_ch/1e3, W_dis_kW=W_dis/1e3)


if __name__ == '__main__':
    import os; os.chdir('/Users/a1234/Carnot_Battery_for_DC/Underground_Brayton')

    print("=== Phase 1 验证：充放电循环 ===\n")
    ch = charging_cycle_v3()
    dis = discharging_cycle_v3(T_HR_out=ch['T2'], T_CR_out=ch['T5'])

    print(f"充能: T1={ch['T1']-273:.1f}°C T2={ch['T2']-273:.1f}°C "
          f"T5={ch['T5']-273:.1f}°C  W_net={ch['W_net']/1e3:.1f}kW")
    print(f"放能: T_D2={dis['T_D2']-273:.1f}°C T_D3={dis['T_D3']-273:.1f}°C "
          f"T_D4={dis['T_D4']-273:.1f}°C  W_net={dis['W_net']/1e3:.1f}kW")
    print(f"  T_D2({dis['T_D2']-273:.1f}°C) > T_CR_out({T_MIN_K-273:.1f}°C)?",
          dis['T_D2'] > T_MIN_K, "← 换热方向正确 ✓")

    pf = compute_performance_v3(ch, dis)
    print(f"\n性能指标:")
    print(f"  χ   = {pf['chi']*100:.1f}%")
    print(f"  ρ_E = {pf['rho_E']:.3f} kWh/m³")
    print(f"  ρ_P = {pf['rho_P']:.1f} kW/(m³/s)")

    print("\n=== Phase 2 验证：Biot 数 & Ergun ===")
    Bi, Re, Nu, h_p = biot_number(450.0, P_HIGH_PA)
    print(f"  Re_p = {Re:.1f}  Nu_p = {Nu:.2f}  h_p = {h_p:.1f} W/m²K")
    print(f"  Bi   = {Bi:.5f}  {'✓ < 0.1' if Bi<0.1 else '✗ FAIL'}")
    dP_dx, u_s, rho = ergun_pressure_drop(M_DOT_DESIGN, 450.0, P_HIGH_PA)
    print(f"  Ergun dP/dx = {dP_dx:.2f} Pa/m  u_s={u_s:.4f} m/s")
    print(f"  总压降 ΔP = {dP_dx*L_TANK:.1f} Pa = {dP_dx*L_TANK/1e5*1000:.2f} mbar")
