"""
fdtd_em.utils.cfl

CFL (Courant–Friedrichs–Lewy) stability condition checker
for 3D FDTD simulations.

本模块职责：
- 在仿真开始前检查时间步长 dt 是否满足 CFL 稳定条件
- 防止数值发散
- 不依赖 torch / fdtd / CUDA
"""

import math

# 真空光速（m/s）
C0 = 299_792_458.0


def max_stable_dt(dx, dy, dz):
    """
    计算 3D FDTD 的最大稳定时间步长

    dt_max = 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    """
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)
    return 1.0 / (C0 * math.sqrt(inv_dx2 + inv_dy2 + inv_dz2))


def check_cfl(dx, dy, dz, dt, safety=0.99):
    """
    CFL 稳定性检查：若 dt 超过稳定上限则抛 ValueError
    """
    dt_max = max_stable_dt(dx, dy, dz) * safety
    if dt > dt_max:
        raise ValueError(
            "CFL condition violated:\n"
            f"  dt = {dt:.3e} s\n"
            f"  dt_max = {dt_max:.3e} s (safety={safety})\n"
            "Reduce dt or increase spatial resolution."
        )
    return dt_max
