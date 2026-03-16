"""
=================================================================================
 椭球体几何模块 (Ellipsoid Region)
 ---------------------------------------------------------------------------------
 本模块用于在 3D 空间中生成一个椭球形区域。
 椭球体在计算电磁学中非常实用，常用于模拟被拉伸的细胞、液滴、
 纳米颗粒，或者用于聚焦透镜焦点的能量探测区域。
=================================================================================
"""

import torch
from typing import Tuple
from fdtd_em.components.detectors.base import Region

class EllipsoidRegion(Region):
    """
    【椭球体区域】
    数学定义：(x-cx)^2/rx^2 + (y-cy)^2/ry^2 + (z-cz)^2/rz^2 <= 1

    使用范例 (模拟一个沿 X 轴被拉伸的“药丸/水滴”状纳米颗粒):
        # 中心在 (52, 52, 52)
        # X轴半径为 15，Y轴和Z轴半径为 8 (像一个横放的橄榄球)
        droplet = EllipsoidRegion(center=(52, 52, 52), radii=(15.0, 8.0, 8.0))
    """

    def __init__(self, center: Tuple[int, int, int], radii: Tuple[float, float, float]):
        """
        初始化椭球体参数。
        :param center: 椭球体的中心坐标 (cx, cy, cz)
        :param radii: 椭球体在 x, y, z 三个轴向的半轴长 (rx, ry, rz)
        """
        self.center = center
        self.radii = radii

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        """
        生成 3D 布尔掩码，判断空间内的点是否在椭球体内。
        """
        Nx, Ny, Nz = shape
        cx, cy, cz = self.center
        rx, ry, rz = self.radii
        X, Y, Z = torch.meshgrid(
            torch.arange(Nx, device=device),
            torch.arange(Ny, device=device),
            torch.arange(Nz, device=device),
            indexing="ij"
        )

        # 计算空间中每一点到中心的归一化距离平方和
        norm = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 + ((Z - cz) / rz) ** 2

        # 距离平方和 <= 1 的点即在椭球体内部
        return norm <= 1.0