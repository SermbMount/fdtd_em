"""
fdtd_em.utils.memory

显存预算工具（float32, torch.cuda）

本模块必须是“纯工具模块”：
- 不允许从项目其他模块反向 import（避免循环依赖）
- 只提供函数，不依赖 Grid / Config 等高层对象
"""

def estimate_vram_bytes(Nx, Ny, Nz, n_arrays=20, alpha=1.3, bytes_per_float=4):
    """
    估算 GPU 显存需求（字节）

    Mem ≈ Nx*Ny*Nz * n_arrays * bytes_per_float * alpha

    参数：
    - n_arrays: 常驻 3D 数组数量（场量 + 系数 + PML 辅助量）
    - alpha: 额外开销系数（padding/临时buffer/workspace）
    - bytes_per_float: float32=4, float64=8
    """
    N = Nx * Ny * Nz
    return int(N * n_arrays * bytes_per_float * alpha)

def assert_within_vram(Nx, Ny, Nz, vram_gb, usable=0.8, n_arrays=20, alpha=1.3, bytes_per_float=4):
    """
    通用显存检查：如果预计显存超限，抛出 MemoryError
    """
    usable_bytes = int(vram_gb * (1024**3) * usable)
    need = estimate_vram_bytes(Nx, Ny, Nz, n_arrays=n_arrays, alpha=alpha, bytes_per_float=bytes_per_float)
    if need > usable_bytes:
        raise MemoryError(
            f"VRAM overflow: need {need/1024**3:.2f} GB > usable {usable_bytes/1024**3:.2f} GB. "
            f"Reduce Nx/Ny/Nz or lower n_arrays/alpha."
        )
    return need

def assert_within_8gb(Nx, Ny, Nz, n_arrays=20, alpha=1.3, usable=0.8):
    """
    8GB GPU 的快捷检查（float32 默认）
    """
    return assert_within_vram(
        Nx, Ny, Nz,
        vram_gb=8,
        usable=usable,
        n_arrays=n_arrays,
        alpha=alpha,
        bytes_per_float=4,  # float32
    )
