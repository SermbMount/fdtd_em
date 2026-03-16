"""
这个模块能灵活地指定一个立方体区域，例如：
它可以在 Ex、Ey、Ez 等张量上提取子块数据，在 CompositeDetector 里就能组合多个这样的区域
"""

import torch
from fdtd_em.components.detectors.base import Region


class BoxRegion(Region):
    """
    表示一个长方体（立方体）区域 (x1:x2, y1:y2, z1:z2)。
    """

    def __init__(self, x_range: tuple[int, int], y_range: tuple[int, int], z_range: tuple[int, int]):
        self.x1, self.x2 = x_range
        self.y1, self.y2 = y_range
        self.z1, self.z2 = z_range

    def mask(self, shape: tuple[int, int, int], device=None) -> torch.Tensor:
        Nx, Ny, Nz = shape

        # 生成带有 GPU device 标记的 3D 坐标网格
        X, Y, Z = torch.meshgrid(
            torch.arange(Nx, device=device),
            torch.arange(Ny, device=device),
            torch.arange(Nz, device=device),
            indexing="ij"
        )

        # 严格判断是否在长方体的边界范围内
        mask_x = (X >= self.x1) & (X < self.x2)
        mask_y = (Y >= self.y1) & (Y < self.y2)
        mask_z = (Z >= self.z1) & (Z < self.z2)

        # 取交集，同时满足 x, y, z 范围的才是长方体内部
        return mask_x & mask_y & mask_z
