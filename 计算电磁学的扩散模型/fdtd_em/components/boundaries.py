"""
fdtd_em.components.boundaries

边界条件组件（Boundary Components）

当前实现：
- PMLBoundary：完美匹配层（Perfectly Matched Layer）

适配 fdtd==0.3.x
"""

from dataclasses import dataclass
import fdtd


@dataclass
class PMLBoundary:
    """
    Perfectly Matched Layer (PML)

    参数说明：
    ----------
    thickness : int
        PML 厚度（网格点数）
        一般建议：10~20
    """
    #默认值是保底机制，若在外部完全控制了真实参数，它根本不会生效。
    thickness: int = 10

    def apply(self, grid: fdtd.Grid) -> None:
        """
        将 PML 挂载到 fdtd.Grid 的六个边界上

        fdtd 的写法标准是：（该版本算法0.3.X）
        grid[0:thickness, :, :] = fdtd.PML(...)
        """

        t = self.thickness

        # x 方向
        grid[0:t, :, :] = fdtd.PML(name="pml_x_low")
        grid[-t:, :, :] = fdtd.PML(name="pml_x_high")

        # y 方向
        grid[:, 0:t, :] = fdtd.PML(name="pml_y_low")
        grid[:, -t:, :] = fdtd.PML(name="pml_y_high")

        # z 方向
        grid[:, :, 0:t] = fdtd.PML(name="pml_z_low")
        grid[:, :, -t:] = fdtd.PML(name="pml_z_high")
