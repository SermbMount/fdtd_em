import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from fdtd_em.config import generation_config as cfg
from fdtd_em.config.simulation import SimulationConfig
from fdtd_em.core.grid import Grid
from fdtd_em.components.detectors import FieldPlaneDetector
from fdtd_em.utils.cfl import max_stable_dt

def run_single_simulation(sample_id, output_dir, device="cuda"):
    # 1. 物理参数随机化 (光源与介电常数)
    current_src_index = int(np.random.randint(cfg.src_index_min, cfg.src_index_max))
    current_src_period = float(np.random.uniform(cfg.src_period_min, cfg.src_period_max))
    eps_r_val = float(np.random.uniform(cfg.eps_r_min, cfg.eps_r_max))

    # 2. 初始化 FDTD 物理网格配置
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
    grid_device = g.grid.inverse_permittivity.device

    # 3. 【核心解耦：动态挂载几何体】
    my_structure = cfg.build_custom_structure(is_batch_generation=True)

    # 提取结构的形状掩码，并注入介电常数
    mask = my_structure.mask(shape=(cfg.Nx, cfg.Ny, cfg.Nz), device=grid_device)
    g.grid.inverse_permittivity[mask] = 1.0 / eps_r_val

    # 4. 挂载探测器
    detectors = []
    regions = cfg.detector_region.regions if hasattr(cfg.detector_region, 'regions') else [cfg.detector_region]

    for i, reg in enumerate(regions):
        det = FieldPlaneDetector(
            name=f"Detector_{i}",
            axis=reg.axis,
            index=reg.index,
            stride=getattr(cfg, "detector_stride", 1),
            to_cpu=True
        )
        det.attach(g.grid)
        g.add_detector(det)
        detectors.append(det)

    # 5. 执行仿真
    g.run()

    # 6. 数据提取与合并
    field_planes = []
    for det in detectors:
        det.sample(g.grid)
        field_planes.append(det.last_map.cpu().numpy())

    # 将多个面合并为一个多通道张量 (C, H, W)
    combined_field = np.stack(field_planes, axis=0)

    sample_path = os.path.join(output_dir, f"sample_{sample_id:05d}")
    os.makedirs(sample_path, exist_ok=True)

    # 7. 落盘保存 (含 Matplotlib 出图)

    # 7.1 保存真实电磁场数据张量 [2, H, W]
    field_map_tensor = torch.tensor(combined_field, dtype=torch.float32)
    torch.save(field_map_tensor, os.path.join(sample_path, "field_map.pt"))

    # 7.2 保存 2D 介质掩码作为条件标签
    structure_slice = mask.any(dim=2).float().cpu().numpy() * eps_r_val
    np.save(os.path.join(sample_path, "structure_mask.npy"), structure_slice)

    # 7.3 生成并保存对比图
    fig, axes = plt.subplots(1, len(detectors), figsize=(10, 5))
    names = ["Reflection", "Transmission"]
    for idx, ax in enumerate(axes):
        im = ax.imshow(combined_field[idx], cmap="viridis", origin="lower")
        ax.set_title(names[idx] if idx < len(names) else f"Plane {idx}")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(sample_path, "field_map_dual.png"), dpi=150)
    plt.close()

    # 7.4 保存物理参数 JSON
    params = {
        "source_index": current_src_index,
        "source_period": current_src_period,
        "eps_r": eps_r_val,
        "shape_type": "dynamic_from_config",
        "detector_info": [{"axis": r.axis, "index": r.index} for r in regions]
    }
    with open(os.path.join(sample_path, "structure.json"), "w") as f:
        json.dump(params, f, indent=4)

def main():
    print(f" 启动 FDTD 物理引擎 | 设备: {cfg.execution_mode}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_dir = os.path.join("dataset", timestamp, "samples")
    os.makedirs(dataset_dir, exist_ok=True)

    batch_dir = os.path.join("dataset", timestamp)
    with open(os.path.join(batch_dir, "config.json"), "w") as f:
        json.dump({
            "Nx": cfg.Nx,
            "samples": cfg.num_samples,
            "steps": cfg.steps,
            "detector_type": type(cfg.detector_region).__name__
        }, f, indent=2)

    print(f" 计划生成 {cfg.num_samples} 个样本...")

    for i in tqdm(range(cfg.num_samples), desc="Simulating FDTD"):
        run_single_simulation(i, dataset_dir)

    print(" 物理数据集生成完毕！")

if __name__ == "__main__":
    main()
