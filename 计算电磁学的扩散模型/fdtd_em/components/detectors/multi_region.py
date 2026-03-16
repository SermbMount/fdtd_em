"""
=================================================================================
 多区域联合模块 (Multi Region)
 ---------------------------------------------------------------------------------
 本模块支持将多个独立的几何区域组合在一起（取逻辑或 / 并集）。
 它非常适合处理由代码循环动态生成的“区域列表”，一次性将它们全部拼装成总区域。
=================================================================================
"""

import torch
from typing import List, Tuple
from fdtd_em.components.detectors.base import Region

class MultiRegion(Region):
    """
    【多区域联合】
    功能：将传入的一个包含多个 Region 实例的列表，合并成一个整体。

   使用范例 (将一个墙壁平面和一个长方体障碍物组合在一起):
        from fdtd_em.components.detectors.plane_region import PlaneRegion
        from fdtd_em.components.detectors.box_region import BoxRegion

        r1 = PlaneRegion(axis="x", index=30)
        # 注意：这里使用的是咱们刚刚现代化改造过的新版 BoxRegion 语法
        r2 = BoxRegion(x_range=(10, 20), y_range=(10, 20), z_range=(10, 20))

        # 把它们装进列表里，一键合并！
        combo = MultiRegion([r1, r2])
    """

    def __init__(self, regions: List[Region]):
        """
        初始化多区域联合。
        :param regions: 包含多个 Region 实例的列表
        """
        assert all(isinstance(r, Region) for r in regions), "所有子元素必须严格是 Region 的子类"
        self.regions = regions

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        """
        生成 3D 布尔掩码，对列表中所有子区域的掩码进行按位或 (|) 运算。
        """
        combined = torch.zeros(shape, dtype=torch.bool, device=device)
        for region in self.regions:
            combined |= region.mask(shape, device=device)

        return combined