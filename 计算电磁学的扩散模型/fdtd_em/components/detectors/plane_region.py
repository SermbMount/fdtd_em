"""
=================================================================================
 无限大平面几何模块 (Plane Region)
 ---------------------------------------------------------------------------------
 本模块用于在 3D 空间中生成一个无限大的平面（例如墙壁、基底、薄膜）。
 它是构建复杂多层结构（如光子晶体平板、基底上的纳米颗粒）的绝佳底层构件。
=================================================================================
"""

import torch
from typing import Tuple
from fdtd_em.components.detectors.base import Region

class PlaneRegion(Region):
    """
    【平面区域】
    定义一个垂直于指定坐标轴的无限大平面。例如 x=30 处的 Y-Z 平面。

     使用范例 (模拟一个放置在 z=20 处的无限大玻璃基底):
        # 生成一个 Z=20 的单层平面
        surface = PlaneRegion(axis="z", index=20)
        # 可以用 BoxRegion，或者用 PlaneRegion 配合循环生成多层。
    """

    def __init__(self, axis: str, index: int):
        """
        初始化平面参数。
        :param axis: 切平面的法线方向，必须是 "x", "y", 或 "z"
        :param index: 在该法线方向上的具体网格索引坐标
        """
        assert axis in ("x", "y", "z"), " axis 必须是 'x', 'y' 或 'z'"
        self.axis = axis
        self.index = index

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        """
        生成 3D 布尔掩码，在指定平面上的位置设为 True，其余为 False。
        """
        grid = torch.zeros(shape, dtype=torch.bool, device=device)
        # 利用张量切片(Slicing)直接赋值，计算效率极高
        if self.axis == "x":
            grid[self.index, :, :] = True
        elif self.axis == "y":
            grid[:, self.index, :] = True
        elif self.axis == "z":
            grid[:, :, self.index] = True

        return grid