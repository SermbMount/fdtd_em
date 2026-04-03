"""
=================================================================================
 计算电磁学逆向设计 - 全局控制台
# 1. FDTD 物理引擎显存占用 (呈三次方指数级增长)
#    VRAM_FDTD ≈ 1.5GB (系统及CUDA基础占用) + (Nx * Ny * Nz) * 1.5e-6 GB
#    -> 当 Nx=Ny=Nz=120 时: 1.5GB + 2.6GB ≈ 4.1GB (甜点区，极其流畅)
#    -> 当 Nx=Ny=Nz=150 时: 1.5GB + 5.0GB ≈ 6.5GB (极度危险，极易触发内存代偿导致速度慢百倍)
#
# 2. U-Net 深度学习显存占用 (与 Batch_Size 成正比，与网格呈平方关系)
#    VRAM_UNet ≈ 1.5GB + Batch_Size * (N / 120)^2 * 0.3GB
#    -> 当 N=120, Batch=16 时: 1.5 + 16 * 1 * 0.3 = 6.3GB (完美吃满 7G 可用显存，效率最高)
#    -> 当 N=150, Batch=16 时: 1.5 + 16 * 1.56 * 0.3 ≈ 9.0GB (必定 OOM 报错显存溢出)
=================================================================================
"""
import random
from fdtd_em.components.detectors.base import Region
from fdtd_em.components.detectors.sphere_region import SphereRegion
from fdtd_em.components.detectors.composite_regions import UnionRegion, SubtractRegion, IntersectionRegion, CompositeRegion
from fdtd_em.components.detectors.annular_region import AnnularRegion
from fdtd_em.components.detectors.cylinder_region import CylinderRegion
from fdtd_em.components.detectors.box_region import BoxRegion
from fdtd_em.components.detectors.ellipsoid_region import EllipsoidRegion
from fdtd_em.components.detectors.plane_region import PlaneRegion
from fdtd_em.components.detectors.multi_region import MultiRegion


# ===============================================================================
# 1.网格与时间步参数 (Spatial & Temporal Grid Parameters)
# 控制底层分辨率
# ===============================================================================
Nx, Ny, Nz = 120, 120, 120
# 三维仿真区域的网格数量（Yee Cell）采样（2^3 = 8）必须用8的整数倍。显存大户，8G是120×120×120为极限7G
dx = 1e-8
dy = 1e-8
dz = 1e-8
freq_center = 3e14  # 光学频段用 3e14，如果是微波/雷达频段可以填 10e9
# 空间步长 (单位: 米)。即每个网格代表 10nm 的实际物理尺寸。
dt_factor = 0.99
# 库朗数 (Courant factor) 缩放系数。取 0.99 是为了在满足 CFL 稳定性条件的前提下追求最大时间步，防止电磁波发散。
steps = 500
# FDTD仿真的总时间步数。
pml_thickness = 10
# 层 (PML) 的厚度（网格数）。吸收到达边界的电磁波，模拟无限大自由空间，防止波反射回来干扰视线。


# ===============================================================================
# 2.激励源参数 (Source Parameters)
# 控制入射电磁波的物理形态，引入随机性以增强生成模型的泛化能力
# ===============================================================================
# 【源类型选择】
source_type = "point"  # 核心选项: "plane" (面源/平面波), "point" (点源/偶极子), "line" (线源)

# 【通用参数】
src_amplitude = 1.62     # 激励源的电场初始振幅。
src_period_min = 15      # 最小波长对应的周期。波长越短，高频散射特征越明显。
src_period_max = 40      # 最大波长对应的周期。波长越长，绕射现象更明显。

# 【依据 source_type 激活的特定参数】
# 若为 "plane" (平面波):
src_axis = "z"           # 传播方向
src_plane_index = 25     # 面源所在的 Z 轴坐标
src_index_min = 20
src_index_max = 25

# 若为 "point" (点源):
src_point_center = (60, 60, 25) # 点源在 3D 空间的三维坐标

# 若为 "line" (线源):
# src_line_p1 = (20, 60, 25)      # 线源起点
# src_line_p2 = (100, 60, 25)     # 线源终点


# ===============================================================================
# 3. 探测器设置 (Detector Parameters)
# ===============================================================================
"""
探测器空间布局去example的detector_cheatsheet.md瞅瞅，
不同的detector有不同的写法，还有inedx的限制
"""
transmission_detector = PlaneRegion(axis="z", index=90)

reflection_detector = PlaneRegion(axis="z", index=15)

detector_region = MultiRegion([transmission_detector, reflection_detector])

# 使用一个半径为 35 的极薄中空球壳作为探测器  雷达散射截面 (RCS)

#spherical_detector = AnnularRegion(center=(60, 60, 60), r_inner=35, r_outer=37)
#detector_region = MultiRegion([spherical_detector])

# ===============================================================================
# 4.结构几何参数 (Geometry & Material Parameters)
# 控制我们在仿真空间里随机生成的“盲盒”物体（即 U-Net 需要反推的 Ground Truth）
# ===============================================================================
eps_r_min = 2.0             # 介电常数下限 (如: 聚合物)
eps_r_max = 6.0             # 介电常数上限 (如: 半导体材料)

# 【空间位置设定】
# 物体中心允许在空间内随意平移的边界
center_min = 40
center_max = 74
sphere_radius_min = 5.0     # 随机生成的最小半径
sphere_radius_max = 15.0    # 随机生成的最大半径

# 【几何形态池】
# 在 build_custom_structure 函数中，可以随意调用 Sphere, Box, Cylinder 等基础几何体
# 并通过 Union, Subtract 进行无限组合。

# ===============================================================================
# 5.数据生成与执行设置 (Pipeline Execution Settings)
# ===============================================================================
num_samples = 500  # 每次运行 generator.py 生成的样本总数。深度学习中，数据量越大，最终 MAE 误差越低。
execution_mode = "local"  # 运行模式标记（后续若上超算集群可改为 "server"）。
device = "cuda"  # 指定FDTD物理求解器的加速硬件。

# ===============================================================================
# 6.U-Net 扩散模型训练超参数 (Deep Learning Hyperparameters)
# ===============================================================================
batch_size = 8  # 每次喂给U-Net的图像数量。若报OOM（Out of Memory），请果断更改为当前使用的最大显存。
epochs = 500  # 整个样本的数据集将被学习X遍。
learning_rate = 1e-4  # 学习率。决定了模型修改自己参数的步子大小。
diffusion_steps = 1000  # DDPM 扩散与去噪的总步数。步数越多，生成的介电常数图像细节越细腻、边缘越锐利。

def build_custom_structure(is_batch_generation=False) -> Region:
    if not is_batch_generation:
        # 直观图预览 / 物理规划的单次测试 (所见即所得)在preview_setup.py
        # 示例：一个大球被偏心挖掉一个小球 (月牙形)
        '''
        big_sphere = SphereRegion(center=(52, 52, 52), radius=12)
        small_sphere = SphereRegion(center=(52, 52, 58), radius=7)
        my_structure = SubtractRegion(big_sphere, small_sphere)
        '''

        my_structure = AnnularRegion(center=(60, 60, 60), r_inner=6, r_outer=12)
        return my_structure

    else:

        r1 = random.uniform(8, 12)
        r2 = random.uniform(6, 10)
        cx1 = random.randint(40, 64)
        cy1, cz1 = random.randint(40, 64), random.randint(40, 64)

        offset = int((r1 + r2) * 0.6)
        cx2 = cx1 + random.choice([-offset, offset, 0])
        cy2 = cy1 + random.choice([-offset, offset, 0])
        cz2 = cz1 + random.choice([-offset, offset, 0])

        sphere1 = SphereRegion(center=(cx1, cy1, cz1), radius=r1)
        sphere2 = SphereRegion(center=(cx2, cy2, cz2), radius=r2)

        return UnionRegion(sphere1, sphere2)
