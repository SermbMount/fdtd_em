import torch
import fdtd
import numpy as np


def add_sphere(grid, center, radius, eps_r):
    """
    在 grid 中添加一个球形区域，修改介电常数 grid.inverse_permittivity。
    自动适配 Numpy (CPU) 或 Torch (GPU/CPU)。
    """
    Nx, Ny, Nz = grid.shape
    cx, cy, cz = center  # 解包中心坐标

    # 1. 检查 grid 中的数组是 Tensor 还是 Numpy
    # fdtd 库初始化时通常是 numpy，set_backend 后才变 tensor


    is_torch_tensor = False
    device = "cpu"

    if hasattr(grid.inverse_permittivity, "device"):
        # 说明是 Torch Tensor
        is_torch_tensor = True
        device = grid.inverse_permittivity.device

    # 2. 计算 Mask (球体区域)
    if is_torch_tensor:
        # --- GPU/Torch 路径 ---
        x = torch.arange(Nx, device=device)
        y = torch.arange(Ny, device=device)
        z = torch.arange(Nz, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        sphere_mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius ** 2

        # 修改介电常数
        grid.inverse_permittivity[sphere_mask] = 1.0 / eps_r

    else:
        # --- CPU/Numpy 路径 ---
        x = np.arange(Nx)
        y = np.arange(Ny)
        z = np.arange(Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        sphere_mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius ** 2

        # 修改介电常数
        grid.inverse_permittivity[sphere_mask] = 1.0 / eps_r