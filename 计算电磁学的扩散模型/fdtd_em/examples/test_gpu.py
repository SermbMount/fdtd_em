"""
运行代码：
python.exe -m fdtd_em.examples.test_gpu
"""
import torch
from fdtd_em.config.simulation import SimulationConfig
from fdtd_em.core.grid import Grid
from fdtd_em.utils.cfl import max_stable_dt   # 用项目自带 CFL

def main():
    dx = 1e-3  # 你可以改大/改小，但 dt 必须跟着 CFL 来
    dt = max_stable_dt(dx, dx, dx) * 0.99      # 保证 < dt_max

    sim_cfg = SimulationConfig(
        Nx=60, Ny=60, Nz=60,
        dx=dx, dt=dt,
        steps=10,
        src_axis="x",
        src_index=5,
        src_period=20,
        src_amplitude=1.0,
        pml_thickness=8,
        execution_mode="local",
        device="cuda",   # 关键
    )

    g = Grid(sim_cfg)

    E = getattr(g.grid, "E", None)
    print("E type:", type(E))
    if isinstance(E, torch.Tensor):
        print("E device:", E.device, "dtype:", E.dtype)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("CUDA mem allocated MB:", torch.cuda.memory_allocated()/1024**2)

    g.step(5)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("CUDA mem allocated MB after step:", torch.cuda.memory_allocated()/1024**2)

if __name__ == "__main__":
    main()

