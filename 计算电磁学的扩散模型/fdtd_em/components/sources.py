"""
激励源组件 (Source Components)
"""
from dataclasses import dataclass
import fdtd
from fdtd_em.config import generation_config as cfg
from fdtd_em.utils.cfl import max_stable_dt

@dataclass
class PlaneWaveSource:
    axis: str = "z"
    index: int = 20
    period: int = 20
    amplitude: float = 1.0
    name: str = "inc_plane"

    def apply(self, grid: fdtd.Grid) -> None:
        # dt 是每个时间步的真实长度
        dt = max_stable_dt(cfg.dx, cfg.dx, cfg.dx) * cfg.dt_factor
        real_period_seconds = self.period * dt
        # 避免把光源建在 PML 吸收层（厚度 10）里面，两边各留 10 个网格的安全距离
        pml = cfg.pml_thickness
        nx, ny, nz = cfg.Nx, cfg.Ny, cfg.Nz

        # 生成物理光源
        src = fdtd.PlaneSource(
            period=real_period_seconds,
            amplitude=self.amplitude,
            name=self.name,
            polarization="x" if self.axis == "z" else "y" # 强制横波偏振
        )
        # 挂载到网格的安全区内
        if self.axis == "z":
            grid[pml:nx-pml, pml:ny-pml, self.index] = src
        elif self.axis == "x":
            grid[self.index, pml:ny-pml, pml:nz-pml] = src
        elif self.axis == "y":
            grid[pml:nx-pml, self.index, pml:nz-pml] = src

        print(f" [激励源注入] 轴向:{self.axis}, 物理周期:{real_period_seconds:.2e}s")