"""
dynamic_perturbation.py — Main Script B: ODE Time-Stepping Perturbation Analysis
══════════════════════════════════════════════════════════════════════════════════
容积控制物理机制（修订版）
────────────────────────
转速是被动量，由轴系角动量ODE决定。功率调节通过容积控制储罐实现：

  功率下降指令 → 储罐阀门抽气 → P_lo↓ → ρ_in↓ → ṁ = ρ·V̇↓
               → W_net = ṁ·w_net↓ → 满足电网需求
  控制不变量：V̇ = const，π = P_hi/P_lo = const

求解顺序（§2.5 因果链）：
  1. 容积控制储罐 → alpha_P(t+dt)，m_dot_valve
  2. 流量代数     → P_lo, P_hi, ṁ, V̇
  3. 电机功率命令 → P_motor（前馈 + 比例调速器）
  4. 压缩机特性图 → β_c, T2
  5. 热储滞后ODE  → δT4 → T4
  6. 冷储滞后ODE  → δT1 → T1
  7. 容积效应ODE  → β_t
  8. 膨胀代数     → T5
  9. 功率计算     → W_comp, W_exp
  10. 轴系ODE     → dω/dt → ω(t+dt)

Usage
─────
  python dynamic_perturbation.py                        # uses default.json
  python dynamic_perturbation.py configs/inv_pid.json   # uses specified config

Outputs → /results/<run_id>_*.npz  /results/fig_<run_id>_*.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['PingFang SC', 'Heiti TC',
                                       'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_RES  = os.path.join(_ROOT, 'results')
os.makedirs(_RES, exist_ok=True)

sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

# ── Load config from CLI arg or default ──────────────────────────
import config as C
if len(sys.argv) > 1:
    _cfg_arg = sys.argv[1]
    # Resolve relative to CWD first, then relative to _HERE
    if not os.path.isabs(_cfg_arg):
        _cwd_path  = os.path.join(os.getcwd(), _cfg_arg)
        _here_path = os.path.join(_HERE, _cfg_arg)
        _cfg_arg   = _cwd_path if os.path.exists(_cwd_path) else _here_path
    C.load(_cfg_arg)
    print(f"Config: {C.cfg_path()}")
else:
    print(f"Config: {C.cfg_path()}  (default)")
from _compressor_turbine    import CompressorTurbine
from _hot_storage           import PackedBedTank
from _cold_storage          import make_cold_tank
from _motor_generator       import MotorGenerator
from _storage_tank_control  import StorageTankControl
from brayton_v3_physics     import charging_cycle_v3

# ═════════════════════════════════════════════════════════════════════════
# 设计点标定（CoolProp 仅调用一次）
# ═════════════════════════════════════════════════════════════════════════
print("Initialising: computing design-point via CoolProp/REFPROP...")
_ch0 = charging_cycle_v3(P_scale=1.0, m_dot=C.M_DOT_DESIGN)

T1_0   = _ch0['T1']
T2_0   = _ch0['T2']
T4_0   = _ch0.get('T7', C.T_ENV_K)
T5_0   = _ch0['T5']
P_lo0  = _ch0['P_lo']
P_hi0  = _ch0['P_hi']
W_c0   = _ch0['W_comp']
W_e0   = _ch0['W_exp']
V_dot0 = _ch0['V_dot']

# REFPROP vs 理想气体密度修正系数
_rho1_ideal   = P_lo0 / (C.R_GAS * T1_0)
_rho1_refprop = C.M_DOT_DESIGN / V_dot0
_RHO_CORR     = _rho1_refprop / _rho1_ideal

print(f"  T1={T1_0-273.15:.1f}°C  T2={T2_0-273.15:.1f}°C  "
      f"T4={T4_0-273.15:.1f}°C  T5={T5_0-273.15:.1f}°C")
print(f"  W_comp={W_c0/1e3:.1f}kW  W_exp={W_e0/1e3:.1f}kW  "
      f"V_dot={V_dot0*1e3:.1f}L/s")

ct = CompressorTurbine()
ct.calibrate(T1_0, T4_0, P_lo0, P_hi0, C.M_DOT_DESIGN)
ct._beta_c0 = P_hi0 / P_lo0

mg = MotorGenerator(mode='charge')
mg.calibrate(W_c0, W_e0, V_dot0, P_lo0, P_hi0)

print(f"  F_cal_C={ct._F_cal_C:.4f}  F_cal_E={ct._F_cal_E:.4f}")
print(f"  P_cmd0={mg._P_cmd0/1e3:.2f}kW  J={C.J_ROTOR:.1f}kg·m²")


# ═════════════════════════════════════════════════════════════════════════
# 仿真主函数
# ═════════════════════════════════════════════════════════════════════════
def run_perturbation(delta_pct:    float = 5.0,
                     t_total:      float = None,
                     t_step:       float = None,
                     dt:           float = None) -> dict:
    """
    容积控制驱动的时域扰动仿真。
    所有控制参数从当前加载的 config (C) 读取。

    Parameters
    ──────────
    delta_pct : 功率阶跃幅度 [%]（正 = 降低）
    t_total   : 仿真总时长 [s]（None = 使用 config）
    t_step    : 阶跃时刻 [s]（None = 使用 config）
    dt        : 时步 [s]（None = 使用 config）
    """
    t_total = t_total if t_total is not None else C.T_TOTAL
    t_step  = t_step  if t_step  is not None else C.T_STEP
    dt      = dt      if dt      is not None else C.DT
    alpha_step = 1.0 - delta_pct / 100.0

    # ── 状态初始化 ────────────────────────────────────────────────
    omega   = C.OMEGA_DESIGN
    P_lo    = P_lo0
    P_hi    = P_hi0
    alpha_P = 1.0
    P_motor = mg._P_cmd0
    beta_t  = ct._beta_c0
    dT_hot  = 0.0
    dT_cold = 0.0
    P_sys   = P_lo0
    # Previous-step W_comp/W_exp for motor feedforward (avoids algebraic loop)
    _W_comp_prev = W_c0
    _W_exp_prev  = W_e0

    # 容积控制储罐（连接在低压管道，压缩机入口侧）
    # 容积控制储罐（参数来自配置文件）
    inv_tank = StorageTankControl(
        V_inv      = C.V_INV,
        P_inv_init = P_lo0,
        T_inv      = T1_0,
        control    = C.INV_MODE,
        Kp         = C.INV_KP,
        Ki         = C.INV_KI,
        Kd         = C.INV_KD,
        tau_f      = C.INV_TAU_F,
    )
    inv_tank.calibrate(V_dot0, P_lo0)

    hot_tank  = PackedBedTank(tank_id='hot',  tau_lag=C.TAU_HOT_STORAGE)
    cold_tank = make_cold_tank()
    T4_ref = T4_0
    T1_ref = T1_0

    # 记录容器
    keys = ['t', 'n', 'omega', 'alpha_P', 'P_lo', 'P_hi', 'P2', 'P5',
            'beta_c', 'beta_t', 'T1', 'T2', 'T4', 'T5',
            'dT_hot', 'dT_cold', 'delta_n', 'delta_beta_c',
            'delta_T2', 'delta_T5', 'm_dot', 'V_dot',
            'W_comp', 'W_exp', 'W_net', 'P_motor',
            'tau_net_kNm', 'alpha_cmd',
            'm_dot_valve', 'u_v', 'P_inv', 'fill_frac']
    rec = {k: [] for k in keys}

    n_steps = int(t_total / dt)

    for step in range(n_steps):
        t = step * dt

        # ── 1. 操作员指令 ─────────────────────────────────────────
        alpha_cmd = mg.step_schedule(t, t_step, alpha_step)

        # ── 2. 容积控制：显式储罐 or 隐式一阶滞后 ───────────────
        T1_cur   = T1_ref + dT_cold
        rho_in   = P_lo / (C.R_GAS * T1_cur) * _RHO_CORR
        m_dot    = float(np.clip(rho_in * V_dot0,
                                  C.M_DOT_DESIGN * C.ALPHA_MIN,
                                  C.M_DOT_DESIGN * C.ALPHA_MAX))
        V_dot    = m_dot / rho_in

        if C.INV_MODE == 'implicit':
            # ── 隐式模式：alpha_P 直接以 τ_inv 跟踪 alpha_cmd ──────
            # 等效于切除物理储罐，用一阶线性滤波器代替
            # dα_P/dt = (α_cmd - α_P) / τ_inv
            alpha_P += (alpha_cmd - alpha_P) * dt / C.TAU_INV
            alpha_P  = float(np.clip(alpha_P, C.ALPHA_MIN, C.ALPHA_MAX))
            P_lo     = P_lo0 * alpha_P
            P_hi     = P_hi0 * alpha_P
            m_dot_valve = 0.0
            inv_out  = dict(m_dot_valve=0.0, dP_lo=0.0, u_v=0.0,
                            P_inv=P_lo0, m_inv=0.0, alpha_P_new=alpha_P)
        else:
            # ── 显式模式：物理储罐 + 阀门 ODE ───────────────────────
            inv_out  = inv_tank.step(alpha_cmd, alpha_P, P_lo, V_dot, T1_cur, dt)
            dP_lo_valve = inv_out['dP_lo']
            m_dot_valve = inv_out['m_dot_valve']
            P_lo += dP_lo_valve * dt
            P_lo  = float(np.clip(P_lo, P_lo0 * C.ALPHA_MIN, P_lo0 * C.ALPHA_MAX))
            P_hi  = P_lo * C.PI_DESIGN
            alpha_P = P_lo / P_lo0
            m_dot_valve = inv_out['m_dot_valve']

        # 重新计算 m_dot（P_lo 已更新）
        rho_in = P_lo / (C.R_GAS * T1_cur) * _RHO_CORR
        m_dot  = float(np.clip(rho_in * V_dot0,
                                C.M_DOT_DESIGN * C.ALPHA_MIN,
                                C.M_DOT_DESIGN * C.ALPHA_MAX))
        V_dot  = m_dot / rho_in

        # ── 3. 电机功率命令（前馈 + 比例调速器）────────────────
        # 物理机制（Zhang2020 §3.2）：
        #   ① 前馈：P_ff = alpha_cmd × P_cmd0（操作员指令，立即响应）
        #      → 阶跃时刻 P_motor 立即降低 → tau_net<0 → n↓（速度扰动）
        #   ② 比例调速器：P_gov = K_GOV × (n_design - n)
        #      → 速度偏差时调整电机功率，维持 n = n_design
        #      → K_GOV 足够大以克服压缩机特性图正反馈（稳定性条件）
        #      → K_GOV 足够小以允许瞬态速度扰动（不过度抑制）
        #   稳定性条件：K_GOV > d(W_comp-W_exp)/dn ≈ 1200 W/rpm
        #   选取 K_GOV = 2000 W/rpm（1.7× 稳定裕度）
        W_fric_design = C.K_FRIC * C.TAU_RATED * C.OMEGA_DESIGN
        P_ff       = alpha_cmd * mg._P_cmd0
        n_rpm_now  = omega * 60.0 / (2.0 * np.pi)
        speed_err  = C.N_DESIGN_RPM - n_rpm_now
        K_GOV      = C.K_GOV   # W/rpm，来自配置文件
        P_motor_cmd = float(np.clip(
            P_ff + K_GOV * speed_err,
            0.0, C.P_MOTOR_RATED))
        P_motor += (P_motor_cmd - P_motor) * dt / C.TAU_GOV
        P_motor  = float(np.clip(P_motor, 0.0, C.P_MOTOR_RATED))

        # ── 4. 压缩机特性图 → β_c, T2 ────────────────────────────
        n_rpm  = omega * 60.0 / (2.0 * np.pi)
        T1     = T1_ref + dT_cold
        beta_c = ct.beta_c_from_speed(n_rpm)
        T2     = T1 * beta_c**(C.KAPPA / ct.eta_p_c)
        dT2    = T2 - T2_0

        # ── 5. 热储滞后ODE → T4 ──────────────────────────────────
        dT_hot += (dT2 - dT_hot) * dt / C.TAU_HOT_STORAGE
        T4      = T4_ref + dT_hot

        # ── 6. 冷储滞后ODE → T1（下一步用）─────────────────────
        dT5_prev = T4_ref * beta_t**(-ct.eta_p_e * C.KAPPA) - T5_0
        dT_cold += (dT5_prev - dT_cold) * dt / C.TAU_COLD_STORAGE

        # ── 7. 容积效应ODE → β_t ─────────────────────────────────
        beta_t += (beta_c - beta_t) * dt / C.TAU_VOL
        beta_t  = float(np.clip(beta_t, 1.5, ct._beta_c0 * 1.15))

        # ── 8. 膨胀代数 → T5 ─────────────────────────────────────
        T5 = T4 * beta_t**(-ct.eta_p_e * C.KAPPA)

        # ── 9. 功率计算 ───────────────────────────────────────────
        W_comp = ct._W_comp_fast(beta_c, m_dot, T1)
        W_exp  = ct._W_exp_fast(beta_t, m_dot, T4)
        W_net  = W_comp - W_exp
        _W_comp_prev = W_comp   # update for next step's motor feedforward
        _W_exp_prev  = W_exp

        # ── 10. 轴系ODE ───────────────────────────────────────────
        ct_state  = {'omega': omega, 'P_sys': P_sys, 'beta_t': beta_t}
        ct_inputs = {'P_motor': P_motor, 'm_dot': m_dot,
                     'T1': T1, 'T4': T4, 'P_lo': P_lo, 'P_hi': P_hi}
        ct_deriv, ct_alg = ct.ode_rhs(ct_state, ct_inputs)

        omega += ct_deriv['d_omega'] * dt
        omega  = float(np.clip(omega,
                                C.OMEGA_DESIGN * 0.40,
                                C.OMEGA_DESIGN * 1.05))
        P_sys += ct_deriv['d_P_sys'] * dt

        # ── 记录 ──────────────────────────────────────────────────
        if step % 4 == 0:
            rec['t'].append(t)
            rec['n'].append(omega * 60.0 / (2.0 * np.pi))
            rec['omega'].append(omega)
            rec['alpha_P'].append(alpha_P)
            rec['P_lo'].append(P_lo / 1e5)
            rec['P_hi'].append(P_hi / 1e5)
            rec['P2'].append(P_hi / 1e5)
            rec['P5'].append(P_lo / 1e5)
            rec['beta_c'].append(beta_c)
            rec['beta_t'].append(beta_t)
            rec['T1'].append(T1 - 273.15)
            rec['T2'].append(T2 - 273.15)
            rec['T4'].append(T4 - 273.15)
            rec['T5'].append(T5 - 273.15)
            rec['dT_hot'].append(dT_hot)
            rec['dT_cold'].append(dT_cold)
            rec['delta_n'].append(n_rpm - C.N_DESIGN_RPM)
            rec['delta_beta_c'].append(beta_c - ct._beta_c0)
            rec['delta_T2'].append(T2 - T2_0)
            rec['delta_T5'].append(T5 - T5_0)
            rec['m_dot'].append(m_dot)
            rec['V_dot'].append(V_dot * 1e3)
            rec['W_comp'].append(W_comp / 1e3)
            rec['W_exp'].append(W_exp / 1e3)
            rec['W_net'].append(W_net / 1e3)
            rec['P_motor'].append(P_motor / 1e3)
            rec['tau_net_kNm'].append(ct_alg['tau_net'] / 1e3)
            rec['alpha_cmd'].append(alpha_cmd)
            rec['m_dot_valve'].append(m_dot_valve)
            rec['u_v'].append(inv_out['u_v'])
            rec['P_inv'].append(inv_out['P_inv'] / 1e5)
            rec['fill_frac'].append(inv_tank.fill_fraction)

    for k in rec:
        rec[k] = np.array(rec[k])
    return rec


# ═════════════════════════════════════════════════════════════════════════
# 运行两个场景
# ═════════════════════════════════════════════════════════════════════════
print(f"\n[1/{len(C.DELTA_PCT_LIST)}] Running {C.DELTA_PCT_LIST[0]}% power step "
      f"(inv_mode={C.INV_MODE})...")
r5  = run_perturbation(delta_pct=C.DELTA_PCT_LIST[0])

print(f"[2/{len(C.DELTA_PCT_LIST)}] Running {C.DELTA_PCT_LIST[1]}% power step "
      f"(inv_mode={C.INV_MODE})...")
r50 = run_perturbation(delta_pct=C.DELTA_PCT_LIST[1])


def metrics(r):
    si     = np.searchsorted(r['t'], C.T_STEP)
    n_post = r['n'][si:]
    n_min  = n_post.min()
    t_min  = r['t'][si + n_post.argmin()]
    ovs    = (C.N_DESIGN_RPM - n_min) / C.N_DESIGN_RPM * 100.0
    n_ss   = n_post[-30:].mean()
    tol    = 0.005 * C.N_DESIGN_RPM
    try:
        idx_s = next(i for i, v in enumerate(n_post)
                     if abs(v - n_ss) < tol
                     and all(abs(n_post[i:i+10] - n_ss) < tol))
        t_s = r['t'][si + idx_s]
    except StopIteration:
        t_s = r['t'][-1]
    return n_min, t_min, ovs, n_ss, t_s


m5  = metrics(r5)
m50 = metrics(r50)

print(f"\n  5%  step: n_min={m5[0]:.1f}rpm @t={m5[1]:.0f}s  "
      f"overshoot={m5[2]:.3f}%  n_final={m5[3]:.2f}rpm  t_settle≈{m5[4]:.0f}s")
print(f"  50% step: n_min={m50[0]:.1f}rpm @t={m50[1]:.0f}s  "
      f"overshoot={m50[2]:.2f}%  n_final={m50[3]:.2f}rpm  t_settle≈{m50[4]:.0f}s")
print(f"  Zhang2020 ref: n_min≈2991rpm @t≈62s  overshoot<0.3%  n_final=3000rpm")

# 保存数据（文件名含 run_id）
_d5_pct  = int(C.DELTA_PCT_LIST[0])
_d50_pct = int(C.DELTA_PCT_LIST[1])
npz5_path  = os.path.join(_RES, f'{C.RUN_ID}_{_d5_pct}pct.npz')
npz50_path = os.path.join(_RES, f'{C.RUN_ID}_{_d50_pct}pct.npz')
np.savez(npz5_path,  **r5)
np.savez(npz50_path, **r50)
print(f"\n  Data saved → {npz5_path}")
print(f"  Data saved → {npz50_path}")

# ═════════════════════════════════════════════════════════════════════════
# 可视化（7面板：原6面板 + 容积控制储罐状态）
# ═════════════════════════════════════════════════════════════════════════
CLR5   = '#1A3E6F'
CLR50  = '#CC3300'
CLRANA = 'seagreen'
CLRREF = '#888888'
CLRT2  = '#8B0000'
CLRT5  = '#00509E'
CLRT4  = '#B8860B'

tau_fast = 20.0;  tau_slow = 120.0
t_arr  = r5['t']
t_rel  = np.where(t_arr >= C.T_STEP, t_arr - C.T_STEP, 0.0)
A5     = -15.4
n_ana5 = np.where(t_arr < C.T_STEP, C.N_DESIGN_RPM,
                  C.N_DESIGN_RPM + A5*(np.exp(-t_rel/tau_slow)
                                        - np.exp(-t_rel/tau_fast)))

fig = plt.figure(figsize=(14, 22))
fig.patch.set_facecolor('#FAFAFA')
gs  = gridspec.GridSpec(5, 2, figure=fig,
                         height_ratios=[1.5, 1.2, 1.2, 1.0, 1.0],
                         hspace=0.52, wspace=0.32,
                         left=0.08, right=0.97, top=0.95, bottom=0.04)

fig.suptitle(
    f'§2.5  容积控制驱动的系统关键参数时域响应  [{C.RUN_ID}]\n'
    f'inv_mode={C.INV_MODE}  Kp={C.INV_KP}  Ki={C.INV_KI}  Kd={C.INV_KD}  |  '
    r'N$_2$ PTES  |  $n_0$=3000 rpm  |  $\pi$=10  |  '
    r'$J$=' + f'{C.J_ROTOR:.0f}' + r' kg·m²',
    fontsize=10.5, fontweight='bold', y=0.97)


def fmt(ax, ylabel, title, xlim=(0, C.T_TOTAL), xlabel=''):
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    ax.axvline(C.T_STEP, color='tomato', lw=1.3, ls=':', alpha=0.85)
    ax.axvspan(0, C.T_STEP, alpha=0.05, color='seagreen')
    ax.axvspan(C.T_STEP, C.T_TOTAL, alpha=0.04, color='tomato')
    ax.grid(lw=0.35, alpha=0.45, ls='--')
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(labelsize=9)


si5  = np.searchsorted(r5['t'],  C.T_STEP)
si50 = np.searchsorted(r50['t'], C.T_STEP)

# Panel 1 (full width): n(t)
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(r5['t'],  r5['n'],  color=CLR5,   lw=2.4, zorder=5,
         label=f'Sim 5%  (n_final={m5[3]:.1f}rpm)')
ax0.plot(r50['t'], r50['n'], color=CLR50,  lw=2.4, zorder=5,
         label=f'Sim 50% (n_final={m50[3]:.1f}rpm)')
ax0.plot(t_arr, n_ana5, color=CLRANA, lw=1.8, ls='--', alpha=0.85,
         label=r'Analytic 5% (Zhang2020 双指数)')
ax0.axhline(C.N_DESIGN_RPM, color=CLRREF, lw=0.9, ls=':', alpha=0.7)
ax0.annotate(
    f'5%: n_min={m5[0]:.0f}rpm\novershoot={m5[2]:.3f}%\n@t={m5[1]:.0f}s',
    xy=(m5[1], m5[0]), xytext=(m5[1]+80, m5[0]-30),
    fontsize=8.5, color=CLR5, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=CLR5, lw=1.1))
ax0.annotate(
    f'50%: n_min={m50[0]:.0f}rpm\novershoot={m50[2]:.1f}%',
    xy=(m50[1], m50[0]), xytext=(m50[1]+60, m50[0]-80),
    fontsize=8.5, color=CLR50, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=CLR50, lw=1.1))
ax0.set_xlim(0, C.T_TOTAL)
ax0.set_ylim(min(r50['n'].min()-100, C.N_DESIGN_RPM-300), C.N_DESIGN_RPM+20)
ax0.set_ylabel('Shaft speed $n$ [rpm]', fontsize=10.5)
ax0.set_title(
    u'(1) 转轴转速 n(t)  —  被动响应量，由轴系角动量ODE决定\n'
    u'功率调节通过容积控制储罐改变 ṁ 实现，转速自然恢复至 n₀',
    fontsize=9.5, fontweight='bold', pad=4)
ax0.axvline(C.T_STEP, color='tomato', lw=1.3, ls=':', alpha=0.85)
ax0.axvspan(0, C.T_STEP, alpha=0.05, color='seagreen')
ax0.axvspan(C.T_STEP, C.T_TOTAL, alpha=0.04, color='tomato')
ax0.grid(lw=0.35, alpha=0.45, ls='--')
ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
ax0.tick_params(labelsize=9)
ax0.legend(fontsize=8.5, loc='lower right', framealpha=0.93)

# Panel 2: β_c & β_t
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(r5['t'],  r5['beta_c'], color=CLR5,  lw=2.2, label=r'$\beta_c$ 5%')
ax1.plot(r5['t'],  r5['beta_t'], color=CLR5,  lw=1.8, ls='--',
         label=r'$\beta_t$ 5%')
ax1.plot(r50['t'], r50['beta_c'], color=CLR50, lw=2.2, label=r'$\beta_c$ 50%')
ax1.plot(r50['t'], r50['beta_t'], color=CLR50, lw=1.8, ls='--', alpha=0.7)
ax1.axhline(ct._beta_c0, color=CLRREF, lw=0.9, ls=':', alpha=0.7)
fmt(ax1, r'$\beta$ [-]', u'(2) 压气机压比 & 膨胀比')
ax1.legend(fontsize=8, loc='lower right')

# Panel 3: T2 & P2
ax2 = fig.add_subplot(gs[1, 1])
ax2t = ax2.twinx()
ax2.plot(r5['t'],  r5['T2'], color=CLRT2,  lw=2.2, label='$T_2$ 5%')
ax2.plot(r50['t'], r50['T2'], color=CLR50, lw=2.0, label='$T_2$ 50%')
ax2.axhline(T2_0-273.15, color=CLRREF, lw=0.8, ls=':', alpha=0.6)
ax2t.plot(r5['t'],  r5['P2'], color=CLRT2,  lw=1.3, ls='--', alpha=0.7)
ax2t.plot(r50['t'], r50['P2'], color=CLR50, lw=1.3, ls='--', alpha=0.7)
ax2.set_ylabel('$T_2$ [°C]', fontsize=9.5, color=CLRT2)
ax2t.set_ylabel('$P_2$ [bar]', fontsize=9.5, color='gray')
ax2t.tick_params(colors='gray', labelsize=9)
fmt(ax2, '', u'(3) 压缩机出口温度 & 压力')
ax2.legend(fontsize=8, loc='lower right')

# Panel 4: T5 & P5
ax3 = fig.add_subplot(gs[2, 0])
ax3t = ax3.twinx()
ax3.plot(r5['t'],  r5['T5'], color=CLRT5,  lw=2.2, label='$T_5$ 5%')
ax3.plot(r50['t'], r50['T5'], color=CLR50, lw=2.0, label='$T_5$ 50%')
ax3.axhline(T5_0-273.15, color=CLRREF, lw=0.8, ls=':', alpha=0.6)
ax3t.plot(r5['t'],  r5['P5'], color=CLRT5,  lw=1.3, ls='--', alpha=0.7)
ax3t.plot(r50['t'], r50['P5'], color=CLR50, lw=1.3, ls='--', alpha=0.7)
ax3.set_ylabel('$T_5$ [°C]', fontsize=9.5, color=CLRT5)
ax3t.set_ylabel('$P_5$ [bar]', fontsize=9.5, color='gray')
ax3t.tick_params(colors='gray', labelsize=9)
fmt(ax3, '', u'(4) 膨胀机出口温度 & 压力')
ax3.legend(fontsize=8, loc='upper right')

# Panel 5: T4 & m_dot
ax4 = fig.add_subplot(gs[2, 1])
ax4t = ax4.twinx()
ax4.plot(r5['t'],  r5['T4'], color=CLRT4,  lw=2.2, label='$T_4$ 5%')
ax4.plot(r50['t'], r50['T4'], color=CLR50, lw=2.0, label='$T_4$ 50%')
ax4.axhline(T4_0-273.15, color=CLRREF, lw=0.8, ls=':', alpha=0.6)
ax4t.plot(r5['t'],  r5['m_dot'], color=CLR5,  lw=1.5, ls='--', alpha=0.8,
          label=r'$\dot{m}$ 5%')
ax4t.plot(r50['t'], r50['m_dot'], color=CLR50, lw=1.5, ls='--', alpha=0.8,
          label=r'$\dot{m}$ 50%')
ax4.set_ylabel('$T_4$ [°C]', fontsize=9.5, color=CLRT4)
ax4t.set_ylabel(r'$\dot{m}$ [kg/s]', fontsize=9.5, color='gray')
ax4t.tick_params(colors='gray', labelsize=9)
fmt(ax4, '', u'(5) 热储出口温度 & 质量流量')
ax4.legend(fontsize=8, loc='upper right')

# Panel 6 (full width): 净扭矩
ax5 = fig.add_subplot(gs[3, :])
ax5.plot(r5['t'],  r5['tau_net_kNm'],  color=CLR5,  lw=2.4, zorder=5,
         label=r'$\tau_{net}$ sim 5%')
ax5.plot(r50['t'], r50['tau_net_kNm'], color=CLR50, lw=2.0, zorder=4,
         label=r'$\tau_{net}$ sim 50%')
ax5.axhline(0, color='k', lw=0.9)
fmt(ax5, r'$\tau_{net}$ [kN·m]',
    u'(6) 净扭矩  —  轴系ODE驱动力  |  τ_net→0 表示转速恢复稳态',
    xlabel='')
ax5.legend(fontsize=8.5, loc='upper right', framealpha=0.93)

# Panel 7 (full width): 容积控制储罐状态
ax6 = fig.add_subplot(gs[4, :])
ax6t = ax6.twinx()
ax6.plot(r5['t'],  r5['m_dot_valve']*1e3, color=CLR5,  lw=2.2,
         label='阀门流量 5% (正=抽气)')
ax6.plot(r50['t'], r50['m_dot_valve']*1e3, color=CLR50, lw=2.0,
         label='阀门流量 50%')
ax6.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
ax6t.plot(r5['t'],  r5['P_inv'], color=CLR5,  lw=1.5, ls='--', alpha=0.8,
          label='储罐压力 5% [bar]')
ax6t.plot(r50['t'], r50['P_inv'], color=CLR50, lw=1.5, ls='--', alpha=0.8,
          label='储罐压力 50%')
ax6.set_ylabel(r'$\dot{m}_{valve}$ [g/s]', fontsize=9.5, color=CLR5)
ax6t.set_ylabel('$P_{inv}$ [bar]', fontsize=9.5, color='gray')
ax6t.tick_params(colors='gray', labelsize=9)
ax6.text(0.97, 0.92,
         u'抽气(+): P_lo↓ → ρ↓ → ṁ↓ → W_net↓\n'
         u'补气(−): P_lo↑ → ρ↑ → ṁ↑ → W_net↑\n'
         u'控制目标: V̇ = const，π = const',
         transform=ax6.transAxes, ha='right', va='top', fontsize=8.5,
         bbox=dict(boxstyle='round', fc='#EAF0FB', ec=CLR5, alpha=0.92))
fmt(ax6, '', u'(7) 容积控制储罐  —  阀门流量 & 储罐压力\n'
    u'物理机制：储罐↔低压管道 动态抽/补工质，改变 ṁ 实现功率调节',
    xlabel='Time  $t$  [s]')
h1, l1 = ax6.get_legend_handles_labels()
h2, l2 = ax6t.get_legend_handles_labels()
ax6.legend(h1+h2, l1+l2, fontsize=8, loc='lower right', framealpha=0.93)

fig_path = os.path.join(_RES, f'fig_{C.RUN_ID}_dynamic_response.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved → {fig_path}")
print("\n✓ Dynamic perturbation analysis complete.")
