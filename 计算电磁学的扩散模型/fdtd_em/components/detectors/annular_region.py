import torch
from fdtd_em.components.detectors.base import Region

class AnnularRegion(Region):
    """
    空心球壳区域：满足 r_inner <= 距离(center) <= r_outer 的所有点
    """
    def __init__(self, center: tuple[int, int, int], r_inner: float, r_outer: float):
        assert r_outer > r_inner, "外半径必须大于内半径"
        self.center = center
        self.r_inner = r_inner
        self.r_outer = r_outer

    def mask(self, shape: tuple[int, int, int], device=None) -> torch.Tensor:
        Nx, Ny, Nz = shape
        cx, cy, cz = self.center
        X, Y, Z = torch.meshgrid(
            torch.arange(Nx, device=device), 
            torch.arange(Ny, device=device), 
            torch.arange(Nz, device=device), 
            indexing="ij"
        )
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        return (dist2 >= self.r_inner ** 2) & (dist2 <= self.r_outer ** 2)