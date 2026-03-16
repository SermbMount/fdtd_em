import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime

from fdtd_em.config import generation_config as cfg
from fdtd_em.config.simulation import SimulationConfig
from fdtd_em.core.grid import Grid
from fdtd_em.components.detectors import FieldPlaneDetector
from fdtd_em.components.geometry import add_sphere
from fdtd_em.utils.cfl import max_stable_dt

#python -m fdtd_em.generator


def generate_sample(sample_id: int, sample_root: str):
    """
    升级版：支持多探测器（双通道）数据采集
    """
    print(f"[DEBUG] Generating sample {sample_id} with Dual-Detectors...")

    # 1. 参数随机化
    current_src_index = int(np.random.randint(cfg.src_index_min, cfg.src_index_max))
    current_src_period = float(np.random.uniform(cfg.src_period_min, cfg.src_period_max))
    current_cx = float(np.random.uniform(cfg.center_min, cfg.center_max))
    current_cy = float(np.random.uniform(cfg.center_min, cfg.center_max))
    current_cz = float(cfg.Nz // 2)
    center = [current_cx, current_cy, current_cz]
    radius = float(np.random.uniform(cfg.sphere_radius_min, cfg.sphere_radius_max))
    eps_r = float(np.random.uniform(cfg.eps_r_min, cfg.eps_r_max))

    # 2. 仿真环境配置
    dx = cfg.dx
    dt = max_stable_dt(dx, dx, dx) * cfg.dt_factor
    sim_cfg = SimulationConfig(
        Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz,
        dx=dx, dt=dt,
        steps=cfg.steps,
        src_axis=cfg.src_axis,
        src_index=current_src_index,
        src_period=current_src_period,
        src_amplitude=cfg.src_amplitude,
        pml_thickness=cfg.pml_thickness,
        execution_mode=cfg.execution_mode,
    )

    g = Grid(sim_cfg)
    add_sphere(g.grid, center=center, radius=radius, eps_r=eps_r)

    # =======================================================
    # 3. 核心改动：初始化双探测器
    # =======================================================
    detectors = []
    # 遍历 MultiRegion 中的每一个子平面 (PlaneRegion)
    regions = cfg.detector_region.regions if hasattr(cfg.detector_region, 'regions') else [cfg.detector_region]

    for i, reg in enumerate(regions):
        det = FieldPlaneDetector(
            name=f"Detector_{i}",
            axis=reg.axis,
            index=reg.index,
            stride=cfg.detector_stride,
            to_cpu=True
        )
        det.attach(g.grid)
        detectors.append(det)

    # 运行物理仿真
    g.run()

    # 4. 数据提取与合并
    field_planes = []
    for det in detectors:
        det.sample(g.grid)
        # 获取该探测面的场图 (H, W)
        field_planes.append(det.last_map.cpu().numpy())

    # 将多个面合并为一个多通道张量 (C, H, W) -> 比如 (2, 104, 104)
    # Channel 0: 反射面 | Channel 1: 透射面
    combined_field = np.stack(field_planes, axis=0)

    # 5. 保存数据
    sample_dir = os.path.join(sample_root, f"sample_{sample_id:04d}")
    os.makedirs(sample_dir, exist_ok=True)

    # 保存参数
    structure_info = {
        "shape": "sphere",
        "center": center, "radius": radius, "eps_r": eps_r,
        "source_index": current_src_index,
        "source_period": current_src_period,
        "detector_info": [{"axis": r.axis, "index": r.index} for r in regions]
    }
    with open(os.path.join(sample_dir, "structure.json"), "w") as f:
        json.dump(structure_info, f, indent=2)

    # 可视化保存 (对比显示两个通道)
    fig, axes = plt.subplots(1, len(detectors), figsize=(10, 5))
    names = ["Reflection (Z=15)", "Transmission (Z=90)"]
    for i, ax in enumerate(axes):
        im = ax.imshow(combined_field[i], cmap="viridis", origin="lower")
        ax.set_title(names[i] if i < len(names) else f"Plane {i}")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, "field_map_dual.png"), dpi=150)
    plt.close()

    # 保存核心数据
    np.save(os.path.join(sample_dir, "field_map.npy"), combined_field)
    torch.save(torch.from_numpy(combined_field), os.path.join(sample_dir, "field_map.pt"))


if __name__ == "__main__":
    print(f"[INFO] 使用设备: {cfg.execution_mode} | 探测器模式: Dual-Channel")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = os.path.join("dataset", timestamp)
    os.makedirs(batch_dir, exist_ok=True)

    # 写入批次配置
    with open(os.path.join(batch_dir, "config.json"), "w") as f:
        # 这里过滤掉无法序列化的对象，仅记录数值
        json.dump({"Nx": cfg.Nx, "samples": cfg.num_samples, "steps": cfg.steps}, f)

    sample_root = os.path.join(batch_dir, "samples")
    os.makedirs(sample_root, exist_ok=True)

    for i in range(cfg.num_samples):
        generate_sample(i, sample_root=sample_root)