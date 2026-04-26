
from dataclasses import dataclass
import torch
import fdtd

@dataclass
class SimulationConfig:
    """
    Global simulation configuration (single source of truth)
    """

    # ------------------------------------------------------------
    # 1. 网格参数（空间）
    # ------------------------------------------------------------
    Nx: int = 120         # x 方向 Yee cell 数
    Ny: int = 120         # y 方向 Yee cell 数
    Nz: int = 120         # z 方向 Yee cell 数
    dx: float = 1e-8      # 空间步长（m）

    # ------------------------------------------------------------
    # 2. 时间参数
    # ------------------------------------------------------------
    dt: float = 1e-17         # 时间步长（s）
    steps: int = 300        # 总时间步数

    # ------------------------------------------------------------
    # 3. 边界条件
    # ------------------------------------------------------------
    pml_thickness: int = 10

    # ------------------------------------------------------------
    # 4. 激励源条件
    # ------------------------------------------------------------
    src_axis: str = "x"
    src_index: int = 30   # index要 > pml_thickness ,不然会被反吸收
    src_period: int = 20
    src_amplitude: float = 1.0

    # ------------------------------------------------------------
    # 5. 数值后端
    # ------------------------------------------------------------
    device: str = "cuda"  # "cuda" / "cuda:0" / "cuda:1"
    dtype: torch.dtype = torch.float32  # float32
    debug: bool = False  # 是否输出调试信息

    # ------------------------------------------------------------
    # 6. 显存预算控制（分为local与server）
    #    在本地实验与服务器两套代码转换
    # ------------------------------------------------------------
    """
    n_arrays_estimate：
        GPU 上“常驻”的 3D tensor 数量估计
        典型构成：
        6 个场量 Ex Ey Ez Hx Hy Hz
        材料系数（εr / σ / 更新系数）
        PML 辅助变量
    """
    execution_mode: str = "local"# "local" or "server"

    n_arrays_estimate: int = 20

    """
    vram_alpha：
        显存额外开销安全系数
        - tensor 对齐
        - 临时 buffer
        - PyTorch 内部 workspace
    """
    vram_alpha: float = 1.3

