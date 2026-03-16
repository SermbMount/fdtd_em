import torch
from abc import ABC, abstractmethod
from typing import Tuple

class Region(ABC):
    """
    所有探测区域的基类。每个子类都必须实现 mask() 方法。
    """

    @abstractmethod
    def mask(self, shape: Tuple[int, int, int], device=None) -> torch.Tensor:
        """
        返回一个布尔张量，表示在给定网格尺寸下该区域的位置。
        shape: (Nx, Ny, Nz)
        device: 确保张量生成在正确的设备上 (CPU 或 GPU)
        返回: torch.bool 类型的张量，形状为 shape。
        """
        pass