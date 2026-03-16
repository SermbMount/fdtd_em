"""
 #Detectors (CSG 几何与探测器构造库)

该文件夹包含 FDTD 物理引擎中基于 **CSG (构造实体几何)** 的核心构建块（`Region` 类型）。
它们不仅能为探测器提供可抽象、可组合的结构组块，更能直接作为物理引擎中的“介电常数掩码（Mask）”，支持像搭乐高积木一样构造出极其复杂的三维电磁超材料与超表面结构。

# 核心基类
* `base.py`：定义了所有几何体的抽象基类 `Region`。所有派生类必须实现并严格遵守 `mask(shape, device)` 接口规范。

# 基础几何图元 (Primitives)
* `sphere_region.py`：完美球体
* `box_region.py`：长方体 / 立方体区域
* `cylinder_region.py`：圆柱体 / 纳米线（支持轴向截断）
* `ellipsoid_region.py`：椭球体 / 液滴状
* `annular_region.py`：空心环 / 环形谐振器
* `plane_region.py`：无限大平面 / 基底切片

#高级逻辑组合器 (CSG Operations)
* `composite_regions.py`：**核心组合大脑！** 实现了极其强大的多图元布尔逻辑运算：
  * `UnionRegion` (并集：A ∪ B ∪ C...)
  * `IntersectionRegion` (交集：A ∩ B ∩ C...)
  * `SubtractRegion` (差集：A - B)
  * `InvertRegion` (取反：~A)
  * `CompositeRegion` (向下兼容别名，等同于 UnionRegion)
* `multi_region.py`：批量组合器。支持直接传入一个 `List[Region]`，将其内部所有几何体执行并集操作，非常适合通过 `for` 循环动态生成的阵列结构。

#组合示例
```python
import torch
from fdtd_em.components.detectors import SphereRegion, PlaneRegion, IntersectionRegion

# 1. 定义一个大球
big_sphere = SphereRegion(center=(60, 60, 60), radius=15)

# 2. 定义一个切半平面 (假设只保留 X 轴 > 60 的部分)
# 注意：若需精确切半，通常结合 BoxRegion 使用，此处仅作概念演示
half_space = PlaneRegion(axis="x", index=60)

# 3. 施加交集魔法：大球 ∩ 半平面 = 半球！
half_sphere = IntersectionRegion(big_sphere, half_space)

# 4. 一键生成 GPU 掩码注入物理引擎
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask = half_sphere.mask(shape=(120, 120, 120), device=device)


"""