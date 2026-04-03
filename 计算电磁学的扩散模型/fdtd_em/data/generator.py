import os
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from fdtd_em.config import generation_config as cfg
from fdtd_em.config.simulation import SimulationConfig
from fdtd_em.core.grid import Grid
from fdtd_em.components.detectors import FieldPlaneDetector
from fdtd_em.utils.cfl import max_stable_dt


def generate_random_parameters():
    """随机生成物理参数和介质球参数"""
    # 光源位置
    src_idx = int(cfg.Nx * 0.2)
    # 兼容配置中的周期
    src_period = float((getattr(cfg, 'src_period_min', 10.0) + getattr(cfg, 'src_period_max', 30.0)) / 2)

    # 随机生成球体参数
    center_x = int(cfg.Nx * 0.5)
    center_y = int(cfg.Ny * 0.5)
    center_z = int(cfg.Nz * 0.5)
    radius = float(np.random.uniform(5.0, 15.0))
    eps_r = float(np.random.uniform(2.0, cfg.eps_r_max))

    return {
        "source_index": src_idx,
        "source_period": src_period,
        "center": [center_x, center_y, center_z],
        "radius": radius,
        "eps_r": eps_r
    }


def run_single_simulation(sample_id, output_dir, device="cuda"):
    params = generate_random_parameters()

    # 1. 初始化 FDTD 物理网格配置
    dx = cfg.dx
    dt = max_stable_dt(dx, dx, dx) * cfg.dt_factor

    sim_cfg = SimulationConfig(
        Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz,
        dx=dx, dt=dt,
        steps=getattr(cfg, 'time_steps', 300),
        src_axis=getattr(cfg, 'src_axis', 'x'),
        src_index=params["source_index"],
        src_period=params["source_period"],
        src_amplitude=getattr(cfg, 'src_amplitude', 1.0),
        pml_thickness=getattr(cfg, 'pml_thickness', 10),
        execution_mode=getattr(cfg, 'execution_mode', 'cuda'),
    )
    g = Grid(sim_cfg)

    # 2. 注入介质体 (直接在张量层操作，杜绝任何 Geometry 模块找不到的 Bug)
    grid_device = g.grid.inverse_permittivity.device
    Z, Y, X = torch.meshgrid(
        torch.arange(cfg.Nz, device=grid_device),
        torch.arange(cfg.Ny, device=grid_device),
        torch.arange(cfg.Nx, device=grid_device),
        indexing='ij'
    )
    cx, cy, cz = params["center"]
    # 构建球体掩码并赋予介电常数
    mask = ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= params["radius"] ** 2
    g.grid.inverse_permittivity[mask] = 1.0 / params["eps_r"]

    # 3. 部署近场探测器
    reflect_plane = FieldPlaneDetector(name="reflect", axis='x', index=int(cfg.Nx * 0.1), to_cpu=True)
    trans_plane = FieldPlaneDetector(name="trans", axis='x', index=int(cfg.Nx * 0.9), to_cpu=True)

    reflect_plane.attach(g.grid)
    trans_plane.attach(g.grid)
    g.add_detector(reflect_plane)
    g.add_detector(trans_plane)

    # 4. 执行仿真
    g.run()

    # 5. 提取数据与保存
    reflect_plane.sample(g.grid)
    trans_plane.sample(g.grid)

    # 提取场图矩阵
    E_ref = reflect_plane.last_map.cpu().numpy()
    E_trans = trans_plane.last_map.cpu().numpy()

    sample_path = os.path.join(output_dir, f"sample_{sample_id:05d}")
    os.makedirs(sample_path, exist_ok=True)

    # 拼装为双通道张量 [2, H, W] 并保存
    field_map = torch.tensor(np.stack([E_ref, E_trans], axis=0), dtype=torch.float32)
    torch.save(field_map, os.path.join(sample_path, "field_map.pt"))

    # 保存物理条件标签
    with open(os.path.join(sample_path, "structure.json"), "w") as f:
        json.dump(params, f, indent=4)


def main():
    print(f" 启动 FDTD 数据引擎 ")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_dir = os.path.join("dataset", timestamp, "samples")
    os.makedirs(dataset_dir, exist_ok=True)

    num_samples = getattr(cfg, 'num_samples', 10)
    print(f" 计划极速生成 {num_samples} 个样本...")

    for i in tqdm(range(num_samples), desc="Simulating FDTD"):
        run_single_simulation(i, dataset_dir)

    print(" 全部物理数据生成完毕！")


if __name__ == "__main__":
    main()
