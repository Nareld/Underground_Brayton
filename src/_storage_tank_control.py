"""
_storage_tank_control.py — Module 5: Inventory Control Storage Tank & Valve
═════════════════════════════════════════════════════════════════════════════
物理描述
────────
容积控制储罐（inventory tank）连接在低压管道（压缩机入口侧），通过阀门
动态向循环管道补充或抽出工质，实现质量流量调节。

物理过程
────────
功率下降指令 → 需要 ṁ↓：
  阀门开启（抽气方向）→ 循环工质流入储罐
  → P_lo↓ → ρ_in = P_lo/(R·T1)↓ → ṁ = ρ_in·V̇↓
  → W_net = ṁ·w_net↓ → 满足功率下降需求

功率上升指令 → 需要 ṁ↑：
  阀门开启（补气方向）→ 储罐高压气体注入低压管道
  → P_lo↑ → ρ_in↑ → ṁ↑

控制不变量（容积控制核心）：
  V̇ = ṁ/ρ = const  （体积流量恒定）
  π = P_hi/P_lo = const  （压比恒定，两侧等比例缩放）
  → 折合流量系数 ṁ_c = ṁ√T/P = const → 特性图工作点不变

状态方程（ODE）
────────────────
储罐质量：
  dm_inv/dt = ṁ_valve                          [kg/s]
  ṁ_valve > 0: 气体从循环流入储罐（抽气）
  ṁ_valve < 0: 气体从储罐流入循环（补气）

储罐压力（理想气体，等温近似）：
  dP_inv/dt = R·T_inv/V_inv · (-ṁ_valve)       [Pa/s]
  （流入储罐 → 储罐压力升高）

循环低压侧压力（死体积 + 阀门流量）：
  dP_lo/dt = R·T1/V_sys · ṁ_valve              [Pa/s]
  （抽气 → P_lo 降低；补气 → P_lo 升高）

阀门流量模型（线性节流）：
  ṁ_valve = u_v · K_v · (P_lo - P_inv) / √(R·T_avg)
  u_v ∈ [-1, 1]：阀门开度（正=抽气，负=补气）
  K_v：阀门流量系数 [kg/(s·Pa^0.5·K^0.5)]

控制逻辑（可选）
────────────────
  'open_loop' : u_v = alpha_cmd - 1  （直接映射功率指令到阀门开度）
  'pi'        : PI 控制，误差信号 e = V̇_target - V̇_actual
  'pid'       : PID 控制，同上加微分项

参考：McTigue2024 §3.3.1, Zhang2020 §2.3, PTES_modeling_design_manual_v2.md §5.5
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C


class StorageTankControl:
    """
    容积控制储罐 + 阀门模块。

    Parameters
    ──────────
    V_inv      : float  储罐容积 [m³]  (default: 0.5 m³)
    P_inv_init : float  储罐初始压力 [Pa]  (default: P_LOW_PA，与循环平衡)
    T_inv      : float  储罐温度 [K]  (等温假设，default: T_ENV_K)
    K_v        : float  阀门流量系数 [kg/(s·Pa)]  (default: 自动标定)
    control    : str    控制模式 'open_loop' | 'pi' | 'pid'
    Kp, Ki, Kd : float  PID 增益（仅 pi/pid 模式使用）
    tau_f      : float  PID 微分滤波时间常数 [s]
    """

    def __init__(self,
                 V_inv:      float = 0.5,
                 P_inv_init: float = C.P_LOW_PA,
                 T_inv:      float = C.T_ENV_K,
                 K_v:        float = None,
                 control:    str   = 'pi',
                 Kp:         float = 2.0,
                 Ki:         float = 0.05,
                 Kd:         float = 0.0,
                 tau_f:      float = 5.0):

        self.V_inv   = V_inv
        self.T_inv   = T_inv
        self.control = control
        self.Kp = Kp;  self.Ki = Ki;  self.Kd = Kd
        self.tau_f = tau_f

        # 阀门流量系数：默认标定为在 ΔP=0.1 bar 时产生 0.1×ṁ_design 的流量
        if K_v is None:
            dP_ref = 0.1 * C.P_LOW_PA          # 0.1 bar 压差
            m_ref  = 0.1 * C.M_DOT_DESIGN       # 10% 设计流量
            T_ref  = C.T_ENV_K
            self.K_v = m_ref / (dP_ref / np.sqrt(C.R_GAS * T_ref))
        else:
            self.K_v = K_v

        # 状态变量
        self.P_inv   = P_inv_init              # 储罐压力 [Pa]
        self.m_inv   = P_inv_init * V_inv / (C.R_GAS * T_inv)  # 储罐质量 [kg]

        # PID 内部状态
        self._integral  = 0.0
        self._err_prev  = 0.0
        self._deriv_f   = 0.0
        self._u_v       = 0.0   # 当前阀门开度

        # 设计点参考（由 calibrate() 设置）
        self._V_dot_ref  = None   # m³/s
        self._P_lo_ref   = C.P_LOW_PA
        self._alpha_prev = 1.0

    # ──────────────────────────────────────────────────────────────
    # 标定
    # ──────────────────────────────────────────────────────────────

    def calibrate(self, V_dot_ref: float, P_lo_ref: float = C.P_LOW_PA) -> None:
        """
        设置设计点体积流量参考值（控制目标）。

        Parameters
        ──────────
        V_dot_ref : 设计点体积流量 [m³/s]
        P_lo_ref  : 设计点低压侧压力 [Pa]
        """
        self._V_dot_ref = V_dot_ref
        self._P_lo_ref  = P_lo_ref

    # ──────────────────────────────────────────────────────────────
    # 阀门流量模型
    # ──────────────────────────────────────────────────────────────

    def _valve_flow(self, P_lo: float, u_v: float) -> float:
        """
        阀门质量流量 [kg/s]。

        直接驱动模型（质量流量执行器）：
          ṁ_valve = u_v × ṁ_max
          u_v > 0: 抽气（气体从循环流入储罐，P_lo↓）
          u_v < 0: 补气（气体从储罐流入循环，P_lo↑）
          ṁ_max = 0.2 × ṁ_design（最大阀门流量）

        物理约束：
          - 储罐不能抽空（m_inv > m_min）
          - 储罐不能超压（P_inv < P_inv_max）

        注：实际系统中阀门流量由压差驱动，此处简化为线性执行器，
        等效于假设储罐压力始终由辅助压缩机维持在参考值。
        """
        m_max   = 0.2 * C.M_DOT_DESIGN
        m_dot_v = u_v * m_max

        # 安全约束
        m_min = self.m_inv * 0.05
        P_max = C.P_HIGH_PA * 1.5
        if m_dot_v > 0 and self.m_inv <= m_min:
            m_dot_v = 0.0
        if m_dot_v < 0 and self.P_inv >= P_max:
            m_dot_v = 0.0

        return float(m_dot_v)

    # ──────────────────────────────────────────────────────────────
    # 控制律
    # ──────────────────────────────────────────────────────────────

    def _control_open_loop(self, alpha_cmd: float, alpha_actual: float) -> float:
        """
        开环控制：阀门开度直接映射压力指令偏差。
          u_v = (alpha_actual - alpha_cmd) × U_MAX
          alpha_cmd < alpha_actual → u_v > 0 → 抽气（降低 P_lo）
        """
        return float(np.clip((alpha_actual - alpha_cmd), -1.0, 1.0))

    def _control_pi(self, alpha_cmd: float, alpha_actual: float,
                    dt: float) -> float:
        """
        PI 控制：误差信号 e = alpha_cmd - alpha_actual（压力偏差）。

        控制目标：alpha_actual → alpha_cmd
        e > 0 (压力偏高，需要抽气降压) → u_v > 0 (抽气)
        e < 0 (压力偏低，需要补气升压) → u_v < 0 (补气)

        物理含义：
          alpha_cmd < 1 → 需要降低 P_lo → 抽气 → u_v > 0
          阀门抽气 → P_lo↓ → ρ↓ → ṁ = ρ·V̇↓ → W_net↓
        """
        e = alpha_actual - alpha_cmd   # 正 = 压力偏高，需要抽气
        self._integral += 0.5 * (e + self._err_prev) * dt
        self._err_prev  = e
        u = self.Kp * e + self.Ki * self._integral
        return float(np.clip(u, -1.0, 1.0))

    def _control_pid(self, alpha_cmd: float, alpha_actual: float,
                     dt: float) -> float:
        """PID 控制：在 PI 基础上加滤波微分项。"""
        e = alpha_actual - alpha_cmd
        self._integral += 0.5 * (e + self._err_prev) * dt
        raw_d = (e - self._err_prev) / max(dt, 1e-9)
        self._deriv_f += dt / (self.tau_f + dt) * (raw_d - self._deriv_f)
        self._err_prev = e
        u = self.Kp * e + self.Ki * self._integral + self.Kd * self._deriv_f
        return float(np.clip(u, -1.0, 1.0))

    # ──────────────────────────────────────────────────────────────
    # 主接口：每时步调用
    # ──────────────────────────────────────────────────────────────

    def step(self,
             alpha_cmd:    float,
             alpha_actual: float,
             P_lo:         float,
             V_dot_actual: float,
             T1:           float,
             dt:           float) -> dict:
        """
        推进容积控制一个时步。

        Parameters
        ──────────
        alpha_cmd    : 功率调节指令（0~1.5，1=设计点）
        alpha_actual : 当前系统压力比例 P_lo/P_lo0
        P_lo         : 当前低压侧压力 [Pa]
        V_dot_actual : 当前实际体积流量 [m³/s]（用于记录，不作控制信号）
        T1           : 压缩机入口温度 [K]
        dt           : 时步 [s]

        Returns dict
        ────────────
        m_dot_valve  : 阀门质量流量 [kg/s]（正=抽气，负=补气）
        dP_lo        : 低压侧压力变化率 [Pa/s]（由阀门引起）
        u_v          : 阀门开度 [-1, 1]
        P_inv        : 储罐当前压力 [Pa]
        m_inv        : 储罐当前质量 [kg]
        alpha_P_new  : 更新后的系统压力比例（含阀门效果）
        """
        # ── 1. 计算阀门开度 ──────────────────────────────────────
        if self.control == 'open_loop':
            u_v = self._control_open_loop(alpha_cmd, alpha_actual)
        elif self.control == 'pi':
            u_v = self._control_pi(alpha_cmd, alpha_actual, dt)
        elif self.control == 'pid':
            u_v = self._control_pid(alpha_cmd, alpha_actual, dt)
        else:
            raise ValueError(f"Unknown control mode: {self.control!r}")
        self._u_v = u_v

        # ── 2. 阀门流量 ──────────────────────────────────────────
        m_dot_valve = self._valve_flow(P_lo, u_v)

        # ── 3. 储罐状态 ODE（Euler 步进）────────────────────────
        # dm_inv/dt = m_dot_valve  （正=流入储罐）
        # dP_inv/dt = R·T_inv/V_inv · m_dot_valve
        self.m_inv += m_dot_valve * dt
        self.m_inv  = max(self.m_inv, 1e-6)
        self.P_inv  = self.m_inv * C.R_GAS * self.T_inv / self.V_inv

        # ── 4. 低压侧压力变化（阀门引起）────────────────────────
        # dP_lo/dt = -R·T1/V_sys · m_dot_valve
        # （抽气 m_dot_valve>0 → P_lo 降低）
        dP_lo = -C.R_GAS * T1 / C.V_SYS_TOTAL * m_dot_valve

        # ── 5. 等效 alpha_P 更新 ─────────────────────────────────
        # 阀门引起的 P_lo 变化转换为 alpha_P 修正量
        alpha_P_new = (P_lo + dP_lo * dt) / self._P_lo_ref
        alpha_P_new = float(np.clip(alpha_P_new, C.ALPHA_MIN, C.ALPHA_MAX))

        return dict(
            m_dot_valve  = m_dot_valve,
            dP_lo        = dP_lo,
            u_v          = u_v,
            P_inv        = self.P_inv,
            m_inv        = self.m_inv,
            alpha_P_new  = alpha_P_new,
        )

    # ──────────────────────────────────────────────────────────────
    # 状态查询
    # ──────────────────────────────────────────────────────────────

    @property
    def fill_fraction(self) -> float:
        """储罐充满度（相对于设计低压满罐）。"""
        m_full = C.P_LOW_PA * self.V_inv / (C.R_GAS * self.T_inv)
        return float(self.m_inv / m_full)

    def reset(self) -> None:
        """重置控制器内部状态（积分项等）。"""
        self._integral = 0.0
        self._err_prev = 0.0
        self._deriv_f  = 0.0
        self._u_v      = 0.0
