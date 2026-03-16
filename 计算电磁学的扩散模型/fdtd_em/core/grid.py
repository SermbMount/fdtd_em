import fdtd
import inspect
import torch
from typing import List

from fdtd_em.utils.memory import assert_within_8gb
from fdtd_em.utils.cfl import check_cfl
from fdtd_em.components.boundaries import PMLBoundary
from fdtd_em.utils.fields import get_field_tensor
from fdtd_em.components.sources import PlaneWaveSource
from fdtd_em.config.simulation import SimulationConfig


class Grid:
    """
    ==============================================================
     FDTD 三维网格管理类（兼容 CUDA / torch 加速）
    ==============================================================

    本类设计原则：
    ----------------------------------------------------------------
    1. Grid 是项目中唯一可操作 `fdtd.Grid` 的地方
    2. 所有物理组件（几何体 / 光源 / 边界 / 探测器）必须通过 Grid 注册
    3. 初始化时自动完成：CUDA 检查 + 显存预算 + CFL 条件验证
    4. 外部模块只能通过 Grid 调用，不得直接访问 fdtd 库
    """

    def __init__(self, sim_cfg: SimulationConfig):
        """
        初始化一个完整的 3D FDTD 仿真环境

        参数
        ----------
        sim_cfg : SimulationConfig
            仿真配置对象，包含空间、时间、源、边界、设备等设置
        """
        self.cfg = sim_cfg
        self.device = sim_cfg.device  # GPU/CPU 信息记录

        # ===========================
        #  1: device / CUDA 检查
        # ===========================
        # 支持：None / "cpu" / "cuda" / "cuda:0" / "cuda:1" ...
        if sim_cfg.device is not None:
            dev = str(sim_cfg.device)
            if dev != "cpu" and not dev.startswith("cuda"):
                raise ValueError(f"Unknown device: {sim_cfg.device}")

        # ===========================
        #  2: 选择 fdtd 后端（关键：必须在 fdtd.Grid 创建前）
        # ===========================
        if hasattr(fdtd, "set_backend"):
            if sim_cfg.device is not None and str(sim_cfg.device).startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "sim_cfg.device 以 'cuda' 开头，但 torch.cuda.is_available() = False。"
                        "请确认安装的是 CUDA 版 PyTorch，显卡可用。"
                    )
                fdtd.set_backend("torch.cuda")
            else:
                fdtd.set_backend("torch")

        # ================================
        #  3: 显存预算检查（8GB）
        # ================================
        if sim_cfg.execution_mode == "local":
            assert_within_8gb(
                Nx=sim_cfg.Nx,
                Ny=sim_cfg.Ny,
                Nz=sim_cfg.Nz,
                n_arrays=sim_cfg.n_arrays_estimate,
                alpha=sim_cfg.vram_alpha,
            )

        # ==============================
        #  4: CFL 数值稳定条件
        # ==============================
        check_cfl(
            dx=sim_cfg.dx,
            dy=sim_cfg.dx,
            dz=sim_cfg.dx,
            dt=sim_cfg.dt,
        )

        # ============================
        #  5: 创建底层 fdtd 网格
        # ============================
        grid_kwargs = {
            "shape": (sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz),
            "grid_spacing": sim_cfg.dx,
            "permittivity": 1.0,
            "permeability": 1.0,
        }

        sig = inspect.signature(fdtd.Grid)
        if "device" in sig.parameters:
            grid_kwargs["device"] = sim_cfg.device
        if "dtype" in sig.parameters:
            grid_kwargs["dtype"] = sim_cfg.dtype
        if "grid_spacing" not in sig.parameters and "spacing" in sig.parameters:
            grid_kwargs["spacing"] = grid_kwargs.pop("grid_spacing")

        self.grid = fdtd.Grid(**grid_kwargs)
        if self.cfg.debug:
         # ===== DEBUG: 确认后端 / device（确认完可删）=====
            E = getattr(self.grid, "E", None)
            H = getattr(self.grid, "H", None)
            print("[DEBUG] grid.E type:", type(E))
            print("[DEBUG] grid.H type:", type(H))
            if isinstance(E, torch.Tensor):
                print("[DEBUG] grid.E device:", E.device, "dtype:", E.dtype)
            if isinstance(H, torch.Tensor):
                print("[DEBUG] grid.H device:", H.device, "dtype:", H.dtype)

        # ============================
        #  6: 初始化组件列表
        # ============================
        self.geometries: List = []
        self.sources: List = []
        self.detectors: List = []
        self.boundaries: List = []

        # ============================
        #  7: 默认注册 PML 和光源
        # ============================
        pml = PMLBoundary(thickness=sim_cfg.pml_thickness)
        pml.apply(self.grid)

        src = PlaneWaveSource(
            axis=sim_cfg.src_axis,
            index=sim_cfg.src_index,
            period=sim_cfg.src_period,
            amplitude=sim_cfg.src_amplitude,
        )
        src.apply(self.grid)

    # ===========================
    #  组件注册接口
    # ===========================

    def add_geometry(self, geom):
        """注册几何体（只负责 apply + 记录）"""
        geom.apply(self.grid)
        self.geometries.append(geom)

    def add_detector(self, det):
        det.attach(self.grid)
        self.detectors.append(det)

    def add_boundary(self, boundary):
        boundary.apply(self.grid)
        self.boundaries.append(boundary)

    # ===========================
    #  时间推进控制接口
    # ===========================

    def step(self, n: int = 1):
        for i in range(n):
            self.grid.step()

            # 仅 debug 第 2 步（i==1），并且兼容 GPU tensor
            if self.cfg.debug and i == 1:
                E = getattr(self.grid, "E", None)
                H = getattr(self.grid, "H", None)

                def _max_abs(x):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        # GPU 上计算 max，再取标量，不会把整块张量拉回 CPU
                        return x.abs().max().item()
                    # 兼容 numpy（万一后端不是 torch）
                    import numpy as np
                    return float(np.max(np.abs(x)))

                print("[DEBUG] step=2 | E max:", _max_abs(E))
                print("[DEBUG] step=2 | H max:", _max_abs(H))

            for det in self.detectors:
                det.sample(self.grid)

    def run(self):
        self.step(self.cfg.steps)

    # ===========================
    #  读取场数据接口（安全）
    # ===========================

    def get_field(self, component: str):
        """
        获取某个场分量（如 'Ex', 'Ey', 'Ez', ...）
        返回的是 detached 的 torch.Tensor
        """
        return get_field_tensor(self.grid, component).detach()
