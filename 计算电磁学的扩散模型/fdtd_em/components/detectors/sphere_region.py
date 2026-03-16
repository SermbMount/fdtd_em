"""
=================================================================================
 球体几何模块 (Sphere Region)
 ---------------------------------------------------------------------------------
 本模块用于在 3D 空间中生成一个标准的完美球体。
 它是整个项目中最基础、最常用的几何图元。无论是单球散射实验、
 双球耦合（花生结构），还是用来做差集（挖空心环），都离不开它。
=================================================================================
"""

import torch
from typing import Tuple
from fdtd_em.components.detectors.base import Region

class SphereRegion(Region):
    """
    【球形区域】
    数学定义：(x-cx)^2 + (y-cy)^2 + (z-cz)^2 <= r^2

    使用范例 (在空间正中心生成一个半径为 15 的介质球):
        # 中心位于 (52, 52, 52)，半径占据 15 个网格
        my_sphere = SphereRegion(center=(52, 52, 52), radius=15.0)
    """

    def __init__(self, center: Tuple[int, int, int], radius: float):
        """
        初始化球体参数。
        :param center: 球心网格坐标 (cx, cy, cz)
        :param radius: 球体半径尺寸 (以网格数为单位)
        """
        self.center = center
        self.radius = radius

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        """
        生成 3D 布尔掩码，计算并判断空间内的点是否在球体内部。
        """
        Nx, Ny, Nz = shape
        cx, cy, cz = self.center
        X, Y, Z = torch.meshgrid(
            torch.arange(Nx, device=device),
            torch.arange(Ny, device=device),
            torch.arange(Nz, device=device),
            indexing="ij"
        )

        # 计算空间中每一点到球心的距离平方
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2

        # 距离平方 <= 半径平方 的点即被激活为 True
        return dist2 <= self.radius ** 2