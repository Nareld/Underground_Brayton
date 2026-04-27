"""
_Ts_plot.py — T-s 热力学循环图核心绘图模块
═════════════════════════════════════════════════════════════════
参考：brayton_du2023_thermo.py / brayton_du2023_main.py 的可视化风格

功能：
  给定充能循环和放能循环的状态点字典，生成精确的 T-s 图：
  ① 多变压缩/膨胀过程：用 polytropic_machine() 路径点（逐点 CoolProp）
  ② 等压换热过程：沿等压线用 CoolProp 逐点计算真实熵变
  ③ 回热器过程（HP/LP 两侧）：沿各自等压线计算

暴露接口：
  make_charging_paths(ch)   → dict of (T_arr_C, s_arr_kJkgK) per process
  make_discharging_paths(dis) → dict of (T_arr_C, s_arr_kJkgK) per process
  plot_Ts(ch, dis, save_path) → 保存图像到 save_path
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import CoolProp.CoolProp as CP

sys.path.insert(0, os.path.dirname(__file__))
import config as C

from brayton_v3_physics import polytropic_machine


# ─────────────────────────────────────────────────────────────────
# CoolProp 接口
# ─────────────────────────────────────────────────────────────────
def _props(output, in1, v1, in2, v2):
    for fl in [C.FLUID, C.FLUID_FB]:
        try:
            r = CP.PropsSI(output, in1, v1, in2, v2, fl)
            if np.isfinite(r):
                return float(r)
        except Exception:
            pass
    raise RuntimeError(f"PropsSI({output}, {in1}={v1}, {in2}={v2}) failed")


def _isobar_path(T_start: float, T_end: float, P: float,
                 n: int = 60) -> tuple:
    """
    沿等压线从 T_start 到 T_end 计算 T-s 路径（CoolProp 逐点）。
    返回 (T_arr [K], s_arr [kJ/kgK])
    """
    T_arr = np.linspace(T_start, T_end, n)
    s_arr = np.array([_props('S', 'T', T, 'P', P) / 1e3 for T in T_arr])
    return T_arr, s_arr


# ─────────────────────────────────────────────────────────────────
# 充能循环路径
# ─────────────────────────────────────────────────────────────────
def make_charging_paths(ch: dict) -> dict:
    """
    从 charging_cycle_v3() 的返回值构建所有 T-s 路径。

    充能循环状态点（brayton_v3_physics.charging_cycle_v3 符号）：
      1  : 压缩机入口（LP 侧，回热器 LP 出口）
      2  : 压缩机出口（HP 侧）
      3  : 热储出口（HP，气体降温后）
      7  : 回热器 HP 出口（= 膨胀机入口）
      5  : 膨胀机出口（LP 侧）
      6  : 冷储出口（LP，气体吸热后）
      1  : 回热器 LP 出口（= 压缩机入口，闭合）

    过程：
      1→2  : 多变压缩  (polytropic_machine 路径)
      2→3  : 等压放热到热储 (isobar HP)
      3→7  : 回热器 HP 侧 等压冷却 (isobar HP)
      7→5  : 多变膨胀  (polytropic_machine 路径)
      5→6  : 等压吸热自冷储 (isobar LP)
      6→1  : 回热器 LP 侧 等压加热 (isobar LP)
    """
    P_lo = ch['P_lo']
    P_hi = ch['P_hi']

    # 1→2 多变压缩
    comp = polytropic_machine(ch['T1'], P_lo, P_hi, C.ETA_POLY_C, n_steps=200)
    T12  = comp['T_path']            # K
    s12  = comp['s_path']            # kJ/kgK  (already /1e3 in polytropic_machine)

    # 2→3 等压放热（HP 侧，气体降温）
    T23, s23 = _isobar_path(ch['T2'], ch['T3'], P_hi * (1 - C.FP_EX))

    # 3→7 回热器 HP 侧（HP，继续降温）
    T37, s37 = _isobar_path(ch['T3'], ch['T7'], P_hi * (1 - C.FP_EX) * (1 - C.FP_RE))

    # 7→5 多变膨胀
    exp = polytropic_machine(ch['T7'], P_hi * (1 - C.FP_EX) * (1 - C.FP_RE),
                              P_lo, C.ETA_POLY_E, n_steps=200)
    T75 = exp['T_path']
    s75 = exp['s_path']

    # 5→6 等压吸热（LP 侧，气体升温）
    T56, s56 = _isobar_path(ch['T5'], ch['T6'], P_lo * (1 - C.FP_EX))

    # 6→1 回热器 LP 侧（LP，继续升温）
    T61, s61 = _isobar_path(ch['T6'], ch['T1'], P_lo * (1 - C.FP_EX))

    return {
        '1→2 压缩': (T12 - 273.15, s12),
        '2→3 热储放热': (T23 - 273.15, s23),
        '3→7 回热HP': (T37 - 273.15, s37),
        '7→5 膨胀': (T75 - 273.15, s75),
        '5→6 冷储吸热': (T56 - 273.15, s56),
        '6→1 回热LP': (T61 - 273.15, s61),
    }


# ─────────────────────────────────────────────────────────────────
# 放能循环路径
# ─────────────────────────────────────────────────────────────────
def make_discharging_paths(dis: dict) -> dict:
    """
    从 discharging_cycle_v3() 的返回值构建放能循环 T-s 路径。

    放能循环状态点：
      D1 : 透平入口（HP，T_MAX）
      D2 : 透平出口（LP）
      D3 : 冷储出口（LP，气体放热降温后）
      D4 : 压缩机出口（HP）
      D5 : 热储出口（HP，气体吸热升温后）
      →D1: 闭合

    过程：
      D1→D2 : 多变膨胀
      D2→D3 : 等压放热到冷储 (isobar LP)
      D3→D4 : 多变压缩
      D4→D5 : 等压吸热自热储 (isobar HP)
      D5→D1 : (已回到入口，如有差异则小段等压)
    """
    P_lo = C.P_LOW_PA
    P_hi = C.P_HIGH_PA

    # D1→D2 多变膨胀
    exp = polytropic_machine(dis['T_D1'], P_hi, P_lo, C.ETA_POLY_E, n_steps=200)
    T_D1_D2 = exp['T_path']
    s_D1_D2 = exp['s_path']

    # D2→D3 等压放热（LP，气体降温至 T_D3）
    T_D2_D3, s_D2_D3 = _isobar_path(dis['T_D2'], dis['T_D3'],
                                      P_lo * (1 - C.FP_EX))

    # D3→D4 多变压缩
    comp = polytropic_machine(dis['T_D3'], P_lo * (1 - C.FP_EX),
                               P_hi, C.ETA_POLY_C, n_steps=200)
    T_D3_D4 = comp['T_path']
    s_D3_D4 = comp['s_path']

    # D4→D5 等压吸热（HP，气体升温）
    T_D4_D5, s_D4_D5 = _isobar_path(dis['T_D4'], dis['T_D5'],
                                      P_hi * (1 - C.FP_EX))

    return {
        'D1→D2 透平膨胀': (T_D1_D2 - 273.15, s_D1_D2),
        'D2→D3 冷储放热': (T_D2_D3 - 273.15, s_D2_D3),
        'D3→D4 压缩': (T_D3_D4 - 273.15, s_D3_D4),
        'D4→D5 热储吸热': (T_D4_D5 - 273.15, s_D4_D5),
    }


# ─────────────────────────────────────────────────────────────────
# 主绘图函数
# ─────────────────────────────────────────────────────────────────
def plot_Ts(ch: dict, dis: dict, save_path: str,
            title_extra: str = '') -> None:
    """
    绘制充能 + 放能 T-s 循环图，保存到 save_path。

    Parameters
    ──────────
    ch        : charging_cycle_v3() 返回值
    dis       : discharging_cycle_v3() 返回值
    save_path : 输出 PNG 路径
    title_extra: 额外标题文字（可选）
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ch_paths  = make_charging_paths(ch)
    dis_paths = make_discharging_paths(dis)

    # ── 颜色方案 ────────────────────────────────────────────────────
    CH_COLORS = {
        '1→2 压缩':    ('#CC3300', '-',  2.5),
        '2→3 热储放热': ('#228B22', '--', 2.0),
        '3→7 回热HP':  ('#4A7B9D', '--', 1.5),
        '7→5 膨胀':    ('#1A3E6F', '-',  2.5),
        '5→6 冷储吸热': ('#6A0DAD', '--', 2.0),
        '6→1 回热LP':  ('#4A7B9D', ':',  1.5),
    }
    DIS_COLORS = {
        'D1→D2 透平膨胀': ('#1A3E6F', '-',  2.5),
        'D2→D3 冷储放热': ('#6A0DAD', '--', 2.0),
        'D3→D4 压缩':    ('#CC3300', '-',  2.5),
        'D4→D5 热储吸热': ('#228B22', '--', 2.0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('#F8F9FA')

    # ── 充能循环 ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#FAFAFA')
    for label, (T_arr, s_arr) in ch_paths.items():
        color, ls, lw = CH_COLORS[label]
        ax.plot(s_arr, T_arr, color=color, ls=ls, lw=lw, label=label)

    # 状态点标记
    state_pts_ch = {
        '1': (ch['T1'], C.P_LOW_PA),
        '2': (ch['T2'], C.P_HIGH_PA),
        '3': (ch['T3'], C.P_HIGH_PA * (1 - C.FP_EX)),
        '7': (ch['T7'], C.P_HIGH_PA * (1 - C.FP_EX) * (1 - C.FP_RE)),
        '5': (ch['T5'], C.P_LOW_PA),
        '6': (ch['T6'], C.P_LOW_PA * (1 - C.FP_EX)),
    }
    for name, (T, P) in state_pts_ch.items():
        s = _props('S', 'T', T, 'P', P) / 1e3
        ax.scatter(s, T - 273.15, s=70, zorder=8, color='k', marker='o')
        offset_s = 0.003 * (1 if name in ('2','3','7') else -1)
        offset_T = 4
        ax.text(s + offset_s, T - 273.15 + offset_T, name,
                fontsize=10, fontweight='bold', color='#1A3E6F', ha='center')

    ax.set_xlabel('Entropy  $s$  [kJ/(kg·K)]', fontsize=11)
    ax.set_ylabel('Temperature  $T$  [°C]', fontsize=11)
    ax.set_title(
        'Charging Cycle (Heat Pump)\n'
        r'N$_2$, $\pi$=' + f'{C.PI_DESIGN:.0f}'
        + r', $\eta_p$=' + f'{C.ETA_POLY_C:.2f}'
        + r', $\varepsilon_{re}$=' + f'{C.EPS_RE:.2f}',
        fontsize=10.5, fontweight='bold', pad=6)
    ax.legend(fontsize=8.5, framealpha=0.93, loc='upper left')
    ax.grid(lw=0.35, alpha=0.45, ls='--')

    # ── 放能循环 ────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('#FAFAFA')
    for label, (T_arr, s_arr) in dis_paths.items():
        color, ls, lw = DIS_COLORS[label]
        ax.plot(s_arr, T_arr, color=color, ls=ls, lw=lw, label=label)

    # 状态点标记
    state_pts_dis = {
        'D1': (dis['T_D1'], C.P_HIGH_PA),
        'D2': (dis['T_D2'], C.P_LOW_PA),
        'D3': (dis['T_D3'], C.P_LOW_PA * (1 - C.FP_EX)),
        'D4': (dis['T_D4'], C.P_HIGH_PA),
        'D5': (dis['T_D5'], C.P_HIGH_PA * (1 - C.FP_EX)),
    }
    for name, (T, P) in state_pts_dis.items():
        s = _props('S', 'T', T, 'P', P) / 1e3
        ax.scatter(s, T - 273.15, s=70, zorder=8, color='k', marker='o')
        ax.text(s + 0.003, T - 273.15 + 4, name,
                fontsize=9.5, fontweight='bold', color='#CC3300', ha='center')

    ax.set_xlabel('Entropy  $s$  [kJ/(kg·K)]', fontsize=11)
    ax.set_ylabel('Temperature  $T$  [°C]', fontsize=11)
    ax.set_title(
        'Discharging Cycle (Heat Engine)\n'
        r'N$_2$, $\pi$=' + f'{C.PI_DESIGN:.0f}'
        + r', $\eta_p$=' + f'{C.ETA_POLY_E:.2f}',
        fontsize=10.5, fontweight='bold', pad=6)
    ax.legend(fontsize=8.5, framealpha=0.93, loc='upper left')
    ax.grid(lw=0.35, alpha=0.45, ls='--')

    suptitle = (
        f'PTES T-s Diagram  |  N₂  |  '
        f'π={C.PI_DESIGN:.0f}  |  '
        f'{C.P_LOW_PA/1e5:.0f}→{C.P_HIGH_PA/1e5:.0f} bar  |  '
        f'η_p={C.ETA_POLY_C:.2f}'
    )
    if title_extra:
        suptitle += f'  |  {title_extra}'
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"T-s diagram saved -> {save_path}")
