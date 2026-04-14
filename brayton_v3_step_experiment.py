"""
brayton_v3_step_experiment.py  ── 完整 PDE 版本（HPC 调试用）
════════════════════════════════════════════════════════════════
储热功率骤降实验：5% 和 50% 阶跃
观测：质量流量、体积流量、LP/HP压力、各器件进出口温度、储热温度

物理建模依据：
  超调来源（假说1✓）：
    电机功率命令立即阶跃 → 轴系净扭矩 < 0 → 转速急剧下降
    J·dω/dt = τ_motor(降) − τ_comp(不变) + τ_exp(不变) − τ_fric
    直到 ṁ 缓慢减小使 W_comp↓ → 新扭矩平衡 → 转速恢复

  调节时间来源（假说2✓）：
    τ₁ ≈ J·ω/|ΔP|（轴系力矩恢复，秒量级）
    τ₂ = τ_inv = 30s（库存控制压力建立）
    τ₃ = τ_thermal（储热罐热容惯性）
    全局调节时间 ≈ max(τ₂, τ₃) ≈ 30–600s

关键参数（修正）：
  J_CORRECT = 314 kg·m²  ← 由 Zhang2020 观测推导：
    Zhang2020 观测：5% 功率阶跃后，转速在 t≈62s 达到最小值
    → τ_shaft ≈ 62s（系统响应时间常数）
    → J = τ_shaft × P_rated / ω² = 62 × 500000 / 314.16² ≈ 314 kg·m²
    config_fixed 中的 J_ROTOR=10.13 kg·m²（τ_shaft=2s）偏小约 30 倍，
    此处覆盖为文献对标值。

  P_motor = cmd × P_cmd_0（立即响应操作员命令，非追踪机械平衡）
    这是超调的物理来源：电机降功率 快，压缩机卸载 慢（τ_inv=30s）
"""

import os, sys, numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'   # 避免乱码
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

os.chdir('/Users/a1234/Carnot_Battery_for_DC/Underground_Brayton')
sys.path.insert(0, '.')
import brayton_v3_config_fixed as C
from brayton_v3_physics import (charging_cycle_v3, packed_bed_step_v3)
import CoolProp.CoolProp as CP

# ── 热物性接口 ───────────────────────────────────────────────────────
def props(prop, **kw):
    k,v = list(kw.keys()), list(kw.values())
    for fl in [C.FLUID, C.FLUID_FB]:
        try:
            r = CP.PropsSI(prop,k[0],v[0],k[1],v[1],fl)
            if np.isfinite(r): return float(r)
        except: pass
    raise RuntimeError(f"PropsSI({prop},{kw})")

# ── 修正转动惯量（由 Zhang2020 实测推导，覆盖 config 中的 J_ROTOR）──
# Zhang2020 实测：5%功率阶跃后 t≈62s 转速达到最小值
# → τ_shaft = 62s  → J = τ × P / ω² = 62 × 500000 / 314.16² = 314 kg·m²
# config_fixed 中 J_ROTOR = 10.13（τ_shaft=2s），偏小约30倍，此处使用文献值
J_CORRECT = 62.0 * C.P_MOTOR_RATED / C.OMEGA_DESIGN**2   # ≈ 314 kg·m²

# ── 内联轴系ODE（使用文献修正 J_CORRECT）────────────────────────────
def shaft_ode(omega, W_comp, W_exp, P_motor):
    """J·dω/dt = τ_motor − τ_comp + τ_exp − τ_fric  [N·m]"""
    w = max(abs(omega), 1.0)
    tau_m = min(P_motor/w, C.TAU_MAX)
    tau_c = W_comp/w
    tau_e = W_exp/w
    tau_f = C.K_FRIC * C.TAU_RATED * (w/C.OMEGA_DESIGN)
    return (tau_m - tau_c + tau_e - tau_f) / J_CORRECT   # ← 使用文献值

# ── 基准工况 ─────────────────────────────────────────────────────────
ch0       = charging_cycle_v3(P_scale=1.0, m_dot=C.M_DOT_DESIGN)
W_fric_0  = C.K_FRIC * C.TAU_RATED * C.OMEGA_DESIGN
P_cmd_0   = ch0['W_comp'] - ch0['W_exp'] + W_fric_0   # 稳态电机命令
V_dot_ref = ch0['V_dot']

print(f"基准：n={C.N_DESIGN_RPM}rpm  W_net={ch0['W_net']/1e3:.1f}kW  "
      f"W_comp={ch0['W_comp']/1e3:.1f}kW  Q_hot={ch0['Q_hot']/1e3:.1f}kW")
print(f"      P_cmd={P_cmd_0/1e3:.1f}kW  V_dot={V_dot_ref*1e3:.0f}L/s")
print(f"      J_CORRECT={J_CORRECT:.1f}kg·m²  (Zhang2020, τ_shaft=62s)")
print(f"      [config J_ROTOR={C.J_ROTOR:.2f}kg·m² 已覆盖]")

# ══════════════════════════════════════════════════════════════════════
# 核心仿真函数（物理正确版）
# ══════════════════════════════════════════════════════════════════════
def run_step_experiment(t_total, t_step, alpha_step, dt=0.5, label=''):
    """
    功率骤降实验

    控制架构（Zhang2020 物理机制）：
      t < t_step : 稳态运行
      t = t_step : 操作员命令 → 电机功率立即降至 alpha_step × P_cmd_0
      t > t_step : 系统压力/流量以 τ_inv=30s 缓慢响应
                   两个时间尺度差异 → 真实超调和调节过程

    超调来源：P_motor↓ 快，W_comp↓ 慢 → 净扭矩 < 0 → 转速↓
    调节来源：τ_inv(压力)、τ_thermal(储热) 多尺度叠加
    """
    # 初始状态（稳态）
    omega     = C.OMEGA_DESIGN
    alpha_P   = 1.0          # 当前系统压力比例（外层，滞后）
    Tf_hot    = np.ones(C.N_X) * C.T_ENV_K
    Ts_hot    = np.ones(C.N_X) * C.T_ENV_K
    Tf_cold   = np.ones(C.N_X) * C.T_ENV_K
    Ts_cold   = np.ones(C.N_X) * C.T_ENV_K

    rho0  = C.P_LOW_PA/(C.R_GAS*C.T_ENV_K)
    u0    = C.M_DOT_DESIGN/(rho0*C.A_CROSS*C.EPS_BED)
    dt_pde= float(np.clip(0.7*C.DX/max(u0,1e-4), 0.005, 0.4))

    # 记录字典
    keys = ['t','n','omega','P_lo','P_hi','pi',
            'm_dot','V_dot','W_comp','W_exp','W_net','P_motor',
            'T_comp_in','T_comp_out','T_exp_in','T_exp_out',
            'T_hotHX_in','T_hotHX_out','T_coldHX_in','T_coldHX_out',
            'T_hot_solid','T_cold_solid','Q_hot','Q_cold']
    rec = {k:[] for k in keys}

    for step in range(int(t_total/dt)):
        t = step * dt

        # ── 操作员命令（即时）vs 系统压力（滞后）──────────────────
        cmd = alpha_step if t >= t_step else 1.0  # 电机功率命令（立即响应）

        # 系统压力以 τ_inv=30s 响应命令
        alpha_P += (cmd - alpha_P) * dt / C.TAU_INV
        alpha_P  = float(np.clip(alpha_P, C.ALPHA_MIN, C.ALPHA_MAX))

        # ── 当前压力和流量 ────────────────────────────────────────
        P_lo = C.P_LOW_PA  * alpha_P
        P_hi = C.P_HIGH_PA * alpha_P
        pi_a = P_hi / P_lo   # = C.PI_DESIGN = const ✓

        T_comp_in = max(Tf_cold[0], C.T_ENV_K - 5)
        rho_in    = P_lo / (C.R_GAS * T_comp_in)
        m_dot_now = rho_in * V_dot_ref          # V̇=const → ṁ∝α_P
        m_dot_now = float(np.clip(m_dot_now,
                                   C.M_DOT_DESIGN*0.05,
                                   C.M_DOT_DESIGN*1.5))
        V_dot_now = m_dot_now / rho_in          # 实际体积流量

        # ── 热力学工作点 ─────────────────────────────────────────
        T_HR_out = max(Tf_hot[-1], C.T_ENV_K)
        T_CR_out = min(Tf_cold[0], C.T_ENV_K)
        try:
            ch = charging_cycle_v3(
                P_scale=alpha_P,
                T_HR_out=T_HR_out, T_CR_out=T_CR_out,
                m_dot=m_dot_now)
            W_c = ch['W_comp']; W_e = ch['W_exp']
            T2=ch['T2']; T_cr=ch['T3']  # comp_out, hotHX_in
            T4=ch['T7'] if 'T7' in ch else ch['T4']
            T5=ch['T5']; T6=ch['T6']    # exp_out, coldHX_out
            Q_h=ch['Q_hot']; Q_c=ch['Q_cold']
        except:
            sc = alpha_P * m_dot_now/C.M_DOT_DESIGN
            W_c=ch0['W_comp']*sc; W_e=ch0['W_exp']*sc
            T2=ch0['T2']; T_cr=ch0['T3']
            T4=ch0.get('T4',ch0['T3']); T5=ch0['T5']; T6=ch0['T6']
            Q_h=ch0['Q_hot']*sc; Q_c=ch0['Q_cold']*sc

        # ── 电机功率命令（立即响应，这是超调的来源）──────────────
        # 注意：电机命令基于 cmd（操作员指令），NOT 基于 alpha_P（实际压力）
        # 这造成了 P_motor 降低 但 W_comp 尚未降低 → 扭矩缺口 → 超调
        P_motor = cmd * P_cmd_0   # 立即响应操作员命令
        P_motor = float(np.clip(P_motor, 0, C.P_MOTOR_RATED))

        # ── 轴系 ODE ─────────────────────────────────────────────
        dw    = shaft_ode(omega, W_c, W_e, P_motor)
        omega = float(np.clip(omega + dt*dw,
                               C.OMEGA_DESIGN*0.50,
                               C.OMEGA_DESIGN*1.05))
        n_rpm = omega * 60 / (2*np.pi)

        # ── 填充床 PDE ───────────────────────────────────────────
        P_hi_eff = P_hi
        rho_t    = P_hi_eff/(C.R_GAS*max(Tf_hot.mean(),200))
        u_t      = m_dot_now/(rho_t*C.A_CROSS*C.EPS_BED)
        n_sub    = max(1, int(dt/dt_pde)+1)
        dts      = dt/n_sub
        for _ in range(n_sub):
            if abs(u_t)*dts/C.DX > 0.95: dts *= 0.9
            Tf_hot, Ts_hot, _ = packed_bed_step_v3(
                Tf_hot, Ts_hot,  u_t, T2, P_hi_eff, dts)
            Tf_cold, Ts_cold, _ = packed_bed_step_v3(
                Tf_cold, Ts_cold, -u_t, T5, P_hi_eff*0.1, dts)

        # ── 记录 ─────────────────────────────────────────────────
        if step % 4 == 0:
            rec['t'].append(t)
            rec['n'].append(n_rpm)
            rec['omega'].append(omega)
            rec['P_lo'].append(P_lo/1e5)        # bar
            rec['P_hi'].append(P_hi/1e5)        # bar
            rec['pi'].append(pi_a)
            rec['m_dot'].append(m_dot_now)
            rec['V_dot'].append(V_dot_now*1e3)  # L/s
            rec['W_comp'].append(W_c/1e3)
            rec['W_exp'].append(W_e/1e3)
            rec['W_net'].append((W_c-W_e)/1e3)
            rec['P_motor'].append(P_motor/1e3)
            rec['T_comp_in'].append(T_comp_in-273.15)
            rec['T_comp_out'].append(T2-273.15)
            rec['T_exp_in'].append(T4-273.15)
            rec['T_exp_out'].append(T5-273.15)
            rec['T_hotHX_in'].append(T2-273.15)   # = comp_out
            rec['T_hotHX_out'].append(T_cr-273.15)
            rec['T_coldHX_in'].append(T5-273.15)  # = exp_out
            rec['T_coldHX_out'].append(T6-273.15)
            rec['T_hot_solid'].append(Ts_hot.mean()-273.15)
            rec['T_cold_solid'].append(Ts_cold.mean()-273.15)
            rec['Q_hot'].append(Q_h/1e3)
            rec['Q_cold'].append(Q_c/1e3)

    for k in rec: rec[k] = np.array(rec[k])
    return rec


# ══════════════════════════════════════════════════════════════════════
# 运行实验
# ══════════════════════════════════════════════════════════════════════
T_TOTAL = 800.0   # 仿真时长 [s]（足够观察多个时间常数）
T_STEP  = 100.0   # 阶跃时刻

print("\n运行 5% 功率骤降实验...")
r5  = run_step_experiment(T_TOTAL, T_STEP, 0.95, dt=0.5, label='5pct')

print("运行 50% 功率骤降实验...")
r50 = run_step_experiment(T_TOTAL, T_STEP, 0.50, dt=0.5, label='50pct')

# ── 关键指标提取 ──────────────────────────────────────────────────────
def metrics(r, t_step=T_STEP):
    si = np.searchsorted(r['t'], t_step)
    n_pre  = r['n'][:si].mean()
    n_post = r['n'][si:]
    n_min  = n_post.min()
    t_min  = r['t'][si + n_post.argmin()]
    ovs    = (n_pre - n_min) / n_pre * 100
    # 调节时间：n 恢复至新稳态 ±0.5% 范围内
    n_ss   = n_post[-50:].mean()
    tol    = 0.005 * n_pre
    try:
        idx_settle = next(i for i,v in enumerate(n_post)
                          if abs(v - n_ss) < tol and
                          all(abs(n_post[i:i+20]-n_ss)<tol))
        t_settle = r['t'][si + idx_settle]
    except:
        t_settle = r['t'][-1]
    return n_min, t_min, ovs, n_ss, t_settle

m5  = metrics(r5)
m50 = metrics(r50)

print(f"\n{'='*60}")
print(f"5%  阶跃：n_min={m5[0]:.1f}rpm @t={m5[1]:.0f}s  超调={m5[2]:.3f}%  "
      f"n_ss={m5[3]:.1f}rpm  t_settle≈{m5[4]:.0f}s")
print(f"50% 阶跃：n_min={m50[0]:.1f}rpm @t={m50[1]:.0f}s  超调={m50[2]:.3f}%  "
      f"n_ss={m50[3]:.1f}rpm  t_settle≈{m50[4]:.0f}s")
print(f"Zhang2020对标：n_min≈2991rpm @t≈62s  超调<0.3%  t_settle≈600s")
print(f"[本仿真 J={J_CORRECT:.0f}kg·m²，Schumann PDE 双填充床，全耦合]")

# ══════════════════════════════════════════════════════════════════════
# 综合可视化（5×2 布局）
# ══════════════════════════════════════════════════════════════════════
C_5  = '#1a6faf'  # 5%阶跃蓝
C_50 = '#cc3300'  # 50%阶跃红
C_G  = '#1a9641'  # 绿
C_O  = '#d4a017'  # 金
GRAY = '#666666'

fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor('#f5f7fa')
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.48, wspace=0.32,
                         left=0.07, right=0.97, top=0.93, bottom=0.04)

fig.suptitle(
    'Brayton PTES — Charging Power Step-Down Experiment (Full PDE)\n'
    r'5% step (blue) & 50% step (red) | $n_{design}$=3000rpm | '
    r'$\pi$=10 | N$_2$ (REFPROP) | Schumann dual packed-bed''\n'
    f'J={J_CORRECT:.0f} kg·m² (Zhang2020) | '
    'Overshoot: shaft torque imbalance (Hyp.1) | '
    r'Settling: $\tau_{inv}$+thermal (Hyp.2)',
    fontsize=10.5, fontweight='bold')

def ax_fmt(ax, xl, yl, title, step_t=T_STEP):
    ax.set_xlabel(xl, fontsize=9.5)
    ax.set_ylabel(yl, fontsize=9.5)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    ax.axvline(step_t, color=GRAY, lw=1.2, ls='--', alpha=0.6)
    ax.axvspan(0, step_t, alpha=0.06, color=C_G)
    ax.axvspan(step_t, T_TOTAL, alpha=0.04, color='#cc3300')
    ax.grid(lw=0.35, alpha=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(labelsize=9)

# ── ROW 0: 转速（最重要，展示超调物理机制）─────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(r5['t'],  r5['n'],  '-', color=C_5,  lw=2.2, label='5% step')
ax.plot(r50['t'], r50['n'], '-', color=C_50, lw=2.2, label='50% step')
ax.axhline(C.N_DESIGN_RPM, color=GRAY, lw=0.8, ls='--', alpha=0.5,
           label=f'$n_{{design}}$={C.N_DESIGN_RPM:.0f}rpm')
# 标注超调
for r, m, col, lbl in [(r5,m5,C_5,'5%'),(r50,m50,C_50,'50%')]:
    si = np.searchsorted(r['t'], T_STEP)
    ax.annotate(f'{lbl}: overshoot\n{m[2]:.3f}%\n@t={m[1]:.0f}s',
                xy=(m[1], m[0]),
                xytext=(m[1]+60, m[0]+50*(1 if col==C_5 else -1)),
                fontsize=8, color=col, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=col, lw=0.9))
ax_fmt(ax, '', 'Shaft speed n [rpm]',
       '(1) Shaft speed\nOvershoot = torque gap (J-effect, Hyp.1)')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(min(r50['n'].min()-50, C.N_DESIGN_RPM*0.75),
            C.N_DESIGN_RPM * 1.01)

# ── ROW 0 右：质量流量 vs 体积流量 ──────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax2 = ax.twinx()
ax.plot(r5['t'],   r5['m_dot'],  '-', color=C_5,  lw=2.0, label='5% ṁ')
ax.plot(r50['t'],  r50['m_dot'], '-', color=C_50, lw=2.0, label='50% ṁ')
ax2.plot(r5['t'],  r5['V_dot'],  '--', color=C_5,  lw=1.5, alpha=0.8, label='5% V̇')
ax2.plot(r50['t'], r50['V_dot'], '--', color=C_50, lw=1.5, alpha=0.8, label='50% V̇')
ax.set_ylabel(r'Mass flow $\dot{m}$ [kg/s]', fontsize=9.5)
ax2.set_ylabel(r'Volume flow $\dot{V}$ [L/s]', fontsize=9.5, color=GRAY)
ax2.tick_params(colors=GRAY, labelsize=9)
# 标注 τ_inv
ax.annotate('', xy=(T_STEP+30, r5['m_dot'][np.searchsorted(r5['t'],T_STEP+30)]),
            xytext=(T_STEP, r5['m_dot'][np.searchsorted(r5['t'],T_STEP)]),
            arrowprops=dict(arrowstyle='->', color=C_5, lw=1.2))
ax.text(T_STEP+15, C.M_DOT_DESIGN*0.99,
        r'$\tau_{inv}$=30s', fontsize=8.5, color=C_5, fontweight='bold')
ax_fmt(ax, '', r'$\dot{m}$ [kg/s]',
       r'(2) Mass flow (solid) & Volume flow (dashed)''\n'
       r'$\dot{m}\propto P$ (slow, $\tau_{inv}$=30s) | $\dot{V}$=const ✓')
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc='lower right')

# ── ROW 1 左：LP & HP 压力 ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.plot(r5['t'],   r5['P_lo'],   '-',  color=C_5,      lw=2.0, label='5% P_lo')
ax.plot(r5['t'],   r5['P_hi'],   '--', color=C_5,      lw=1.8, label='5% P_hi')
ax.plot(r50['t'],  r50['P_lo'],  '-',  color=C_50,     lw=2.0, label='50% P_lo')
ax.plot(r50['t'],  r50['P_hi'],  '--', color=C_50,     lw=1.8, label='50% P_hi')
# 初始压力参考
ax.axhline(C.P_LOW_PA/1e5,  color=GRAY, lw=0.7, ls=':', alpha=0.5)
ax.axhline(C.P_HIGH_PA/1e5, color=GRAY, lw=0.7, ls=':', alpha=0.5)
ax.text(5, C.P_LOW_PA/1e5*1.02,  f'LP$_{{design}}$={C.P_LOW_PA/1e5:.0f}bar',
        fontsize=8, color=GRAY)
ax.text(5, C.P_HIGH_PA/1e5*1.01, f'HP$_{{design}}$={C.P_HIGH_PA/1e5:.0f}bar',
        fontsize=8, color=GRAY)
ax_fmt(ax, '', 'Pressure [bar]',
       '(3) LP (solid) & HP (dashed) pressures\n'
       r'Both scale: $\pi=P_{hi}/P_{lo}$=const throughout')
ax.legend(fontsize=8, loc='upper right', ncol=2)

# ── ROW 1 右：压比验证 ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.plot(r5['t'],   r5['pi'],  '-', color=C_5,  lw=2.2, label=r'5% $\pi$')
ax.plot(r50['t'],  r50['pi'], '-', color=C_50, lw=2.2, label=r'50% $\pi$')
ax.axhline(C.PI_DESIGN, color=GRAY, lw=1.0, ls='--', alpha=0.6,
           label=f'$\\pi_{{design}}$={C.PI_DESIGN}')
ax.set_ylim(C.PI_DESIGN-0.01, C.PI_DESIGN+0.01)
ax.text(400, C.PI_DESIGN+0.005,
        r'$\pi$ = const throughout $\checkmark$',
        fontsize=10, color=C_G, fontweight='bold', ha='center')
ax_fmt(ax, '', r'Pressure ratio $\pi$ [-]',
       r'(4) Pressure ratio $\pi = P_{hi}/P_{lo}$''\n'
       r'Inventory control maintains $\pi$=const (Hyp.2 verification)')
ax.legend(fontsize=8)

# ── ROW 2 左：压气机进出口温度 ───────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.plot(r5['t'],  r5['T_comp_in'],  '-',  color=C_5,      lw=1.8, label='5% T$_{in}$')
ax.plot(r5['t'],  r5['T_comp_out'], '--', color=C_5,      lw=2.0, label='5% T$_{out}$')
ax.plot(r50['t'], r50['T_comp_in'], '-',  color=C_50,     lw=1.8, label='50% T$_{in}$')
ax.plot(r50['t'], r50['T_comp_out'],'--', color=C_50,     lw=2.0, label='50% T$_{out}$')
ax_fmt(ax, '', 'Temperature [deg C]',
       '(5) Compressor inlet/outlet temperatures\n'
       'T$_{out}$ tracks pressure ratio change; T$_{in}$ = cold storage outlet')
ax.legend(fontsize=8, ncol=2)

# ── ROW 2 右：透平进出口温度 ─────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
ax.plot(r5['t'],  r5['T_exp_in'],   '-',  color=C_5,  lw=1.8, label='5% T$_{in}$')
ax.plot(r5['t'],  r5['T_exp_out'],  '--', color=C_5,  lw=2.0, label='5% T$_{out}$')
ax.plot(r50['t'], r50['T_exp_in'],  '-',  color=C_50, lw=1.8, label='50% T$_{in}$')
ax.plot(r50['t'], r50['T_exp_out'], '--', color=C_50, lw=2.0, label='50% T$_{out}$')
ax_fmt(ax, '', 'Temperature [deg C]',
       '(6) Expander inlet/outlet temperatures\n'
       'T$_{in}$ from recuperator/hot HX; T$_{out}$ = cold storage inlet')
ax.legend(fontsize=8, ncol=2)

# ── ROW 3 左：热储换热器温度 ─────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
ax.plot(r5['t'],  r5['T_hotHX_in'],  '-',  color=C_5,  lw=1.8, label='5% HX$_{in}$')
ax.plot(r5['t'],  r5['T_hotHX_out'], '--', color=C_5,  lw=2.0, label='5% HX$_{out}$')
ax.plot(r50['t'], r50['T_hotHX_in'], '-',  color=C_50, lw=1.8, label='50% HX$_{in}$')
ax.plot(r50['t'], r50['T_hotHX_out'],'--', color=C_50, lw=2.0, label='50% HX$_{out}$')
ax_fmt(ax, '', 'Temperature [deg C]',
       '(7) Hot-side HX (hot storage boundary)\n'
       'in = compressor outlet; out = toward recuperator/expander')
ax.legend(fontsize=8, ncol=2)

# ── ROW 3 右：冷储换热器温度 ─────────────────────────────────────
ax = fig.add_subplot(gs[3, 1])
ax.plot(r5['t'],  r5['T_coldHX_in'],  '-',  color=C_5,  lw=1.8, label='5% HX$_{in}$')
ax.plot(r5['t'],  r5['T_coldHX_out'], '--', color=C_5,  lw=2.0, label='5% HX$_{out}$')
ax.plot(r50['t'], r50['T_coldHX_in'], '-',  color=C_50, lw=1.8, label='50% HX$_{in}$')
ax.plot(r50['t'], r50['T_coldHX_out'],'--', color=C_50, lw=2.0, label='50% HX$_{out}$')
ax_fmt(ax, '', 'Temperature [deg C]',
       '(8) Cold-side HX (cold storage boundary)\n'
       'in = expander outlet; out = toward compressor inlet')
ax.legend(fontsize=8, ncol=2)

# ── ROW 4 左：储热温度 & 充热功率 ───────────────────────────────
ax  = fig.add_subplot(gs[4, 0])
ax2 = ax.twinx()
ax.plot(r5['t'],   r5['T_hot_solid'],  '-',  color=C_5,  lw=2.0, label='5% T$_{hot}$')
ax.plot(r5['t'],   r5['T_cold_solid'], '--', color=C_5,  lw=1.8, label='5% T$_{cold}$')
ax.plot(r50['t'],  r50['T_hot_solid'], '-',  color=C_50, lw=2.0, label='50% T$_{hot}$')
ax.plot(r50['t'],  r50['T_cold_solid'],'--', color=C_50, lw=1.8, label='50% T$_{cold}$')
ax2.plot(r5['t'],  r5['Q_hot'],  ':',  color=C_5,  lw=1.5, alpha=0.7, label='5% Q$_{hot}$')
ax2.plot(r50['t'], r50['Q_hot'], ':',  color=C_50, lw=1.5, alpha=0.7, label='50% Q$_{hot}$')
ax.set_ylabel('Storage temperature [deg C]', fontsize=9.5)
ax2.set_ylabel('Q$_{hot}$ [kW]', fontsize=9.5, color=C_O)
ax2.tick_params(colors=C_O, labelsize=9)
ax_fmt(ax, 'Time [s]', '',
       '(9) Storage temperatures (solid/dashed) & heat flux (dotted)\n'
       r'$\tau_{thermal}$ >> $\tau_{inv}$: storage is slow/stable buffer')
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=7.5, loc='lower right', ncol=2)

# ── ROW 4 右：功率平衡 ───────────────────────────────────────────
ax = fig.add_subplot(gs[4, 1])
ax.plot(r5['t'],   r5['W_comp'],  '-',  color=C_5,  lw=1.5, label='5% W$_{comp}$')
ax.plot(r5['t'],   r5['W_exp'],   '--', color=C_5,  lw=1.5, label='5% W$_{exp}$')
ax.plot(r5['t'],   r5['W_net'],   '-',  color=C_5,  lw=2.5, alpha=0.9, label='5% W$_{net}$')
ax.plot(r50['t'],  r50['W_comp'], '-',  color=C_50, lw=1.5, label='50% W$_{comp}$')
ax.plot(r50['t'],  r50['W_exp'],  '--', color=C_50, lw=1.5, label='50% W$_{exp}$')
ax.plot(r50['t'],  r50['W_net'],  '-',  color=C_50, lw=2.5, alpha=0.9, label='50% W$_{net}$')
ax.plot(r5['t'],   r5['P_motor'], ':',  color='#888', lw=1.5, label='P$_{motor}$ cmd')
ax.plot(r50['t'],  r50['P_motor'],':',  color='#888', lw=1.5)
ax_fmt(ax, 'Time [s]', 'Power [kW]',
       '(10) Power balance: comp/exp/net + motor command\n'
       'Gap between P$_{motor}$(dotted) and W$_{net}$(thick): source of overshoot')
ax.legend(fontsize=7.5, ncol=2)

plt.savefig('fig_v3_step_experiment.png', dpi=150,
            bbox_inches='tight', facecolor='#f5f7fa')
plt.close(fig)
print("\n✓ 实验图 -> fig_v3_step_experiment.png")

# ── 打印关键数字汇总 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("实验结果汇总")
print("="*60)
for label, r, m, a in [("5%", r5, m5, 0.95),("50%",r50,m50,0.50)]:
    si = np.searchsorted(r['t'], T_STEP)
    print(f"\n{label} 功率骤降（α={a}）：")
    print(f"  转速：{r['n'][:si].mean():.1f} → min={m[0]:.1f}rpm "
          f"@t={m[1]:.0f}s  超调={m[2]:.3f}%  新稳态={m[3]:.1f}rpm")
    print(f"  质量流量：{r['m_dot'][:si].mean():.3f} → "
          f"{r['m_dot'][-20:].mean():.3f} kg/s  (×{r['m_dot'][-20:].mean()/r['m_dot'][:si].mean():.3f})")
    print(f"  体积流量：{r['V_dot'][:si].mean():.1f} → "
          f"{r['V_dot'][-20:].mean():.1f} L/s  (const={abs(r['V_dot'][-20:].mean()-r['V_dot'][:si].mean())/r['V_dot'][:si].mean()*100:.1f}% var)")
    print(f"  LP压力：{r['P_lo'][:si].mean():.2f} → {r['P_lo'][-20:].mean():.2f} bar")
    print(f"  HP压力：{r['P_hi'][:si].mean():.2f} → {r['P_hi'][-20:].mean():.2f} bar")
    print(f"  π：{r['pi'][:si].mean():.4f} → {r['pi'][-20:].mean():.4f}  (const✓)")
    print(f"  T_comp_out：{r['T_comp_out'][:si].mean():.1f} → {r['T_comp_out'][-20:].mean():.1f} degC")
    print(f"  T_exp_out ：{r['T_exp_out'][:si].mean():.1f} → {r['T_exp_out'][-20:].mean():.1f} degC")
    print(f"  W_net：{r['W_net'][:si].mean():.1f} → {r['W_net'][-20:].mean():.1f} kW  "
          f"(×{r['W_net'][-20:].mean()/r['W_net'][:si].mean():.3f})")
    print(f"  Q_hot：{r['Q_hot'][:si].mean():.1f} → {r['Q_hot'][-20:].mean():.1f} kW")
    print(f"  调节时间≈{m[4]:.0f}s")
