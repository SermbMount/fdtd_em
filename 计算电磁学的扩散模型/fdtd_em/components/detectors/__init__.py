import torch
from dataclasses import dataclass
from typing import Optional
from fdtd_em.utils.fields import get_field_tensor


@dataclass
class BaseDetector:
    """探测器基类"""
    name: str

    def attach(self, grid) -> None:
        self._grid = grid

    def sample(self, grid) -> None:
        raise NotImplementedError


@dataclass
class FieldPlaneDetector(BaseDetector):
    """高效的切片二维场图探测器"""
    axis: str
    index: int
    stride: int = 1
    to_cpu: bool = True
    dtype_out: torch.dtype = torch.float32
    last_map: Optional[torch.Tensor] = None

    def sample(self, grid) -> None:
        # 获取电场张量
        E = get_field_tensor(grid, "E")

        # 计算电场强度
        intensity = (E ** 2).sum(dim=-1)

        # 利用切片直接提取二维平面
        if self.axis == "x":
            plane_data = intensity[self.index, ::self.stride, ::self.stride]
        elif self.axis == "y":
            plane_data = intensity[::self.stride, self.index, ::self.stride]
        elif self.axis == "z":
            plane_data = intensity[::self.stride, ::self.stride, self.index]
        else:
            raise ValueError(f"Unknown axis: {self.axis}")

        out = plane_data.to(self.dtype_out)
        if self.to_cpu:
            out = out.cpu()
        self.last_map = out