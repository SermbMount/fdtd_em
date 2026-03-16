import os
import torch
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

from fdtd_em.config import generation_config as cfg
from fdtd_em.config.simulation import SimulationConfig
from fdtd_em.core.grid import Grid
from fdtd_em.components.detectors import FieldPlaneDetector
from fdtd_em.components.detectors.composite_regions import UnionRegion, SubtractRegion
from fdtd_em.components.detectors.annular_region import AnnularRegion
from fdtd_em.utils.cfl import max_stable_dt


def main():
    print(f"[INFO] 当前设备是否支持 CUDA: {torch.cuda.is_available()} | 使用设备: {cfg.execution_mode}")

    dx = cfg.dx
    dt = max_stable_dt(dx, dx, dx) * cfg.dt_factor

    # 1. 仿真参数（从 Config 读取）
    sim_cfg = SimulationConfig(
        Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz,
        dx=dx, dt=dt,
        steps=cfg.steps,
        src_axis=cfg.src_axis,
        src_index=int((cfg.src_index_min + cfg.src_index_max) / 2),
        src_period=float((cfg.src_period_min + cfg.src_period_max) / 2),
        src_amplitude=cfg.src_amplitude,
        pml_thickness=cfg.pml_thickness,
        execution_mode=cfg.execution_mode,
    )

    g = Grid(sim_cfg)

    # 2. 放置结构 (测试用空心环)
    ring_structure = AnnularRegion(center=(cfg.Nx // 2, cfg.Ny // 2, cfg.Nz // 2), r_inner=6, r_outer=12)
    grid_device = g.grid.inverse_permittivity.device
    mask = ring_structure.mask(shape=(cfg.Nx, cfg.Ny, cfg.Nz), device=grid_device)
    g.grid.inverse_permittivity[mask] = 1.0 / cfg.eps_r_max

    # 3. 【核心修复】动态适配 MultiRegion 探测器
    detectors = []
    # 判断 Config 里是多个平面还是单个
    regions = cfg.detector_region.regions if hasattr(cfg.detector_region, 'regions') else [cfg.detector_region]

    for i, reg in enumerate(regions):

        det = FieldPlaneDetector(
            name=f"test_plane_{i}",
            axis=reg.axis,
            index=reg.index,
            stride=getattr(cfg, "detector_stride", 1),
            to_cpu=True
        )
        det.attach(g.grid)
        g.add_detector(det)
        detectors.append(det)

    print(f"[INFO] 成功挂载 {len(detectors)} 个探测面，准备开始仿真...")

    # 4. 执行
    start_time = time.time()
    g.run()
    print(f"测试完成！耗时: {time.time() - start_time:.2f} 秒")

    # 5. 可视化所有探测面
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("physics_test_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(detectors), figsize=(len(detectors) * 5, 5))
    if len(detectors) == 1: axes = [axes]  # 统一成列表方便遍历

    for i, (det, ax) in enumerate(zip(detectors, axes)):
        det.sample(g.grid)
        field_map = det.last_map.cpu().numpy()

        # 保存原始数据
        np.save(os.path.join(output_dir, f"plane_{i}_data.npy"), field_map)

        im = ax.imshow(field_map, cmap='viridis', origin='lower')
        ax.set_title(f"Plane {i} ({regions[i].axis}={regions[i].index})")
        plt.colorbar(im, ax=ax, label='|E|²')

    plt.tight_layout()
    img_path = os.path.join(output_dir, "step1_physics_test_dual_plane.png")
    plt.savefig(img_path, dpi=150)
    print(f"【物理测试】多通道对比图已保存至: {img_path}")

    # 注意：这里没有写 plt.show()，用 show()，系统会弹出图片并卡住，
    # 必须等你手动关闭图片，才会进行下一步。
if __name__ == "__main__":
    main()


