"""
逻辑运算区域

UnionRegion: 并集

IntersectionRegion: 交集

SubtractRegion: 差集

InvertRegion: 取反
"""
import torch
from typing import Tuple
from fdtd_em.components.detectors.base import Region


class UnionRegion(Region):
    """
        【区域并集 (A ∪ B ∪ C...)】
        功能：将传入的所有几何体“融合”在一起。只要空间中的某一点属于其中任意一个几何体，该点就被激活。

        使用范例 (生成一个“花生”或“双体船”形状):
            sphere1 = SphereRegion(center=(40, 52, 52), radius=10)
            sphere2 = SphereRegion(center=(60, 52, 52), radius=10)
            # 将两个球体融合成一个整体
            peanut_shape = UnionRegion(sphere1, sphere2)
        """

    def __init__(self, *regions: Region):
        self.regions = regions

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        # 初始化一个全为 False 的空白掩码
        final_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        for region in self.regions:
            # 严格传递 device，并按位或
            final_mask |= region.mask(shape, device=device)
        return final_mask


class IntersectionRegion(Region):
    """
    【区域交集 (A ∩ B ∩ C...)】
    功能：提取所有几何体“重叠”的部分。只有同时存在于所有几何体内部的空间点，才会被激活。

    使用范例 (切除球体的一半，生成“半球”):
        sphere = SphereRegion(center=(52, 52, 52), radius=15)
        # 假设 BoxRegion 框住了右半边空间
        right_half_box = BoxRegion(x_range=(52, 104), y_range=(0, 104), z_range=(0, 104))
        # 取交集，只保留球体在右半边的部分
        half_sphere = IntersectionRegion(sphere, right_half_box)
    """

    def __init__(self, *regions: Region):
        self.regions = regions

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        if not self.regions:
            return torch.zeros(shape, dtype=torch.bool, device=device)

        # 以第一个区域的掩码作为基准
        final_mask = self.regions[0].mask(shape, device=device)
        for region in self.regions[1:]:
            # 严格传递 device，并按位与
            final_mask &= region.mask(shape, device=device)
        return final_mask


class SubtractRegion(Region):
    """A - B：A 区域中减去 B 区域"""

    def __init__(self, region_a: Region, region_b: Region):
        self.region_a = region_a
        self.region_b = region_b

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        mask_a = self.region_a.mask(shape, device=device)
        mask_b = self.region_b.mask(shape, device=device)
        return mask_a & ~mask_b


class InvertRegion(Region):
    """区域取反（非 ~A）"""

    def __init__(self, region: Region):
        self.region = region

    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        return ~self.region.mask(shape, device=device)

class CompositeRegion(UnionRegion):
    """
    [兼容性保留]
    如果你在 config 里面导了 CompositeRegion，它现在等价于 UnionRegion。
    你可以把任意多个区域传给它，它会自动把它们合并在一起！
    """
    pass
