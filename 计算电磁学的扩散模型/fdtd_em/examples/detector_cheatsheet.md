"""
 📡 计算电磁学 - 探测器 (Detectors) 终极备忘录大全
=================================================================================
前置条件假设 (基于当前默认config):
- 空间总大小: 120 x 120 x 120
- 绝对坐标区 (避开PML): 11 ~ 108
- 光源位置: Z = 20 ~ 25 (向 +Z 传播)
- 物体中心位置: Z ≈ 60
"""
from fdtd_em.components.detectors.multi_region import MultiRegion
from fdtd_em.components.detectors.plane_region import PlaneRegion
from fdtd_em.components.detectors.box_region import BoxRegion
from fdtd_em.components.detectors.sphere_region import SphereRegion
from fdtd_em.components.detectors.annular_region import AnnularRegion
from fdtd_em.components.detectors.cylinder_region import CylinderRegion

# 1. PlaneRegion (平面探测器) - 最常用，算透射率/反射率必备
#  适用场景：看特定切面的场图，或者计算通过某个面的总能流 (Poynting Vector)。
#  注意事项：index 必须在安全区内。
trans_det = PlaneRegion(axis="z", index=90)  # 透射面 (物体后方)
refl_det  = PlaneRegion(axis="z", index=35)  # 反射面 (夹在光源Z=25和物体Z=60之间)
side_det  = PlaneRegion(axis="y", index=60)  # 侧剖面 (上帝视角，看波怎么绕过物体)

# 2. BoxRegion (3D 矩形包围盒探测器)
#  适用场景：用来计算目标的总吸收截面 (Absorption Cross Section)。
# ️ 注意事项：就是一个长方体，必须完全包裹住你的物体，但不能碰到 PML。
# 参数: x_min, x_max, y_min, y_max, z_min, z_max
box_det = BoxRegion(x_min=30, x_max=90,
                    y_min=30, y_max=90,
                    z_min=30, z_max=90)

# 3. SphereRegion (实心球体探测器) & AnnularRegion (空心球壳探测器)
#  适用场景：空心球壳 (Annular) 是电磁学里的神器，专门用来做“近场到远场外推 (N2F)”，
#    计算雷达散射截面 (RCS) 或者远场辐射方向图 (Radiation Pattern)。
# ️ 注意事项：球心最好和物体中心重合，半径要足够大包住物体，但绝不能切进 PML。

# 实心球探测器 (较少用作独立探测，常用来做布尔组合)
solid_sphere_det = SphereRegion(center=(60, 60, 60), radius=35)

# ★ 空心球壳雷达探测器 (高级货)
# r_inner 是内径，r_outer 是外径。只抓取这一层极薄的球壳里的能量分布。
rcs_radar_det = AnnularRegion(center=(60, 60, 60), r_inner=35, r_outer=37)


# 4. CylinderRegion (圆柱体探测器)
#  适用场景：当你仿真的物体是一根光纤、波导管、或者纳米线时，
#    用圆柱体包裹探测是最贴合物理几何的。
# 参数：axis (圆柱朝向), center (底面圆心), radius (半径), height (高度)
fiber_det = CylinderRegion(axis="z", center=(60, 60, 30), radius=20, height=60)


#  5. MultiRegion (多通道组合模块) - 系统的总路由
#  适用场景：无论你上面定义了什么奇葩形状的探测器，只要你想同时用它们，
#    就必须把它们塞进一个列表里，用 MultiRegion 打包！引擎只认这个。

# 组合范例 A：双平面双目视觉 (AI 训练最爱)
detector_region_A = MultiRegion([trans_det, refl_det])

# 组合范例 B：平面透射 + 全向雷达球壳 (极度硬核的数据采集)
detector_region_B = MultiRegion([PlaneRegion(axis="z", index=90), rcs_radar_det])
