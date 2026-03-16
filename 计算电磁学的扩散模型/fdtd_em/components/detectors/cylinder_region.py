"""
=================================================================================
 圆柱体几何模块 (Cylinder Region)
 ---------------------------------------------------------------------------------
 本模块用于在 3D 空间中生成一个沿特定轴向延伸的圆柱体。
 常用于模拟光纤、硅纳米线、微柱微腔 (Micropillar) 等一维延伸的超材料结构。
=================================================================================
"""

import torch
from typing import Tuple, Optional
from fdtd_em.components.detectors.base import Region


class CylinderRegion(Region):
    """
    【圆柱体区域】
    定义一个沿 x、y 或 z 轴延伸的圆柱体。既可以是贯穿整个空间的无限长圆柱，
    也可以通过 `range` 参数截断成有限长度的短柱。

    使用范例 (模拟一根沿 Y 轴放置、被截断的硅纳米微柱):
        # 中心在 (52, 52, 52)，横截面半径为 8
        # range=(30, 70) 表示只保留 Y 坐标在 30 到 70 之间的部分
        nanowire = CylinderRegion(
            axis="y",
            center=(52, 52, 52),
            radius=8.0,
            range=(30, 70)
        )
    """

    def __init__(self, axis: str, center: Tuple[int, int, int], radius: float, range: Optional[Tuple[int, int]] = None):
        """
        初始化圆柱体参数。
        :param axis: 圆柱体延伸的主轴，必须为 "x", "y" 或 "z"
        :param center: 圆柱体中心坐标 (cx, cy, cz)
        :param radius: 圆柱体横截面的半径
        :param range: (可选) 沿延伸轴的有效起止范围。如果不填，则默认为无限长。
        """
        assert axis in ("x", "y", "z"), " axis 必须为 'x', 'y' 或 'z'"
        self.axis = axis
        self.center = center
        self.radius = radius
        self.range = range

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        """
        生成 3D 布尔掩码。首先计算二维截面是否在圆内，然后再根据 range 截断长度。
        """
        Nx, Ny, Nz = shape
        cx, cy, cz = self.center
        X, Y, Z = torch.meshgrid(
            torch.arange(Nx, device=device),
            torch.arange(Ny, device=device),
            torch.arange(Nz, device=device),
            indexing="ij"
        )

        if self.axis == "z":
            dist2 = (X - cx) ** 2 + (Y - cy) ** 2
            mask = dist2 <= self.radius ** 2
            if self.range:
                z_start, z_end = self.range
                z_mask = (Z >= z_start) & (Z <= z_end)
                mask &= z_mask

        elif self.axis == "y":
            dist2 = (X - cx) ** 2 + (Z - cz) ** 2
            mask = dist2 <= self.radius ** 2
            if self.range:
                y_start, y_end = self.range
                y_mask = (Y >= y_start) & (Y <= y_end)
                mask &= y_mask

        elif self.axis == "x":
            dist2 = (Y - cy) ** 2 + (Z - cz) ** 2
            mask = dist2 <= self.radius ** 2
            if self.range:
                x_start, x_end = self.range
                x_mask = (X >= x_start) & (X <= x_end)
                mask &= x_mask

        return mask