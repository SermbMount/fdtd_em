import numpy as np
import torch

def unwrap_to_torch(x, device=None):
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise TypeError("Unsupported input type")
    if device:
        tensor = tensor.to(device)
    return tensor

def get_field_tensor(grid, name: str) -> torch.Tensor:
    # 初始化一次性 debug 标志
    if not hasattr(get_field_tensor, "_printed"):
        get_field_tensor._printed = False

    # 直接分量式
    if hasattr(grid, name):
        f = unwrap_to_torch(getattr(grid, name))
        if isinstance(f, torch.Tensor):
            return f
        raise TypeError(f"{name} exists but not tensor: {type(f)}")

    # 矢量式：grid.E
    if name in ("Ex", "Ey", "Ez") and hasattr(grid, "E"):
        E = unwrap_to_torch(getattr(grid, "E"))
        if not isinstance(E, torch.Tensor):
            raise TypeError(f"grid.E cannot convert to torch.Tensor: {type(E)}")

        # 🔹 只打印一次
        if not get_field_tensor._printed:
            print(
                "DEBUG E (after unwrap) shape:",
                tuple(E.shape),
                "dtype:", E.dtype,
                "device:", E.device,
            )
            get_field_tensor._printed = True

        comp = {"Ex": 0, "Ey": 1, "Ez": 2}[name]

        if E.ndim == 4 and E.shape[-1] == 3:
            return E[..., comp]

        if E.ndim == 4 and E.shape[0] == 3:
            return E[comp, ...]

        raise ValueError(f"Unsupported grid.E shape: {tuple(E.shape)}")

    raise AttributeError(f"Cannot find field '{name}' on grid")

