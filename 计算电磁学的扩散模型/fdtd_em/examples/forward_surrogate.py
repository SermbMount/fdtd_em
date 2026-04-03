import os
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import fdtd_em.config.generation_config as cfg
from fdtd_em.train.dataset import MyDataset
from fdtd_em.train.model import ConditionalDDPM


def find_latest_checkpoint(root="checkpoints"):
    all_runs = glob.glob(os.path.join(root, "20*"))
    if not all_runs:
        raise FileNotFoundError("未找到任何权重，请先跑 train.py")
    return sorted(all_runs)[-1]


def find_latest_dataset(root="dataset"):
    all_runs = glob.glob(os.path.join(root, "20*/samples"))
    if not all_runs:
        raise FileNotFoundError("未找到数据集")
    return sorted(all_runs)[-1]


def load_field_stats(dataset_root):
    """读取全局场图统计量，用于反归一化"""
    stats_path = os.path.join(dataset_root, "field_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return stats['min'], stats['max']
    print(" 未找到 field_stats.json，使用默认 [-1, 1] 比例")
    return -1.0, 1.0


def denormalize_field(field_tensor, f_min, f_max):
    """将 [-1, 1] 的网络输出还原为真实物理电场强度 (V/m)"""
    return (field_tensor + 1.0) / 2.0 * (f_max - f_min) + f_min


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"启动正向物理预测测试 | 设备: {device}")

    # 1. 加载最新的模型和数据
    latest_ckpt_dir = find_latest_checkpoint()
    model_path = os.path.join(latest_ckpt_dir, "3d_fdtd_model.pth")

    dataset_path = find_latest_dataset()
    data_dir = Path(dataset_path)

    # 获取测试样本
    test_files = [p for p in data_dir.iterdir() if p.is_dir()]
    if not test_files:
        print(" 数据集目录下没有找到样本！")
        return
    test_files = test_files[:4]

    # 提取统计量 (已修复层级找错的 Bug)
    f_min, f_max = load_field_stats(data_dir)

    test_dataset = MyDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_files), shuffle=False)

    # 2. 初始化网络并加载权重
    model = ConditionalDDPM(target_dim=2, cond_dim=1, n_steps=cfg.diffusion_steps, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # ==========================================
    # 【核心修复】：强行将模型及内部模块锁定至 GPU
    # ==========================================
    model.to(device)
    model.eval()

    # 3. 提取数据
    real_fields, structures, phys_conds = next(iter(test_loader))
    structures = structures.to(device)
    real_fields = real_fields.to(device)
    phys_conds = phys_conds.to(device)

    # 4. 正向预测
    print(" 正在推演电磁波的空间散射分布 (可能需要几秒钟)...")
    with torch.no_grad():
        pred_fields = model.sample(img_cond=structures, phys_cond=phys_conds)

    # 5. 反归一化到真实物理量级
    real_fields_phys = denormalize_field(real_fields, f_min, f_max)
    pred_fields_phys = denormalize_field(pred_fields, f_min, f_max)

    # 计算定量指标
    mse = torch.nn.functional.mse_loss(pred_fields_phys, real_fields_phys).item()
    mae = torch.nn.functional.l1_loss(pred_fields_phys, real_fields_phys).item()
    print(f" 定量评估结果 -> 物理场 MSE: {mse:.4e}, MAE: {mae:.4e}")

    # 6. 可视化对比
    structures_np = structures.cpu().numpy()
    real_fields_np = real_fields_phys.cpu().numpy()
    pred_fields_np = pred_fields_phys.cpu().numpy()

    num_samples = len(structures_np)
    fig, axes = plt.subplots(num_samples, 5, figsize=(16, 3 * num_samples))
    fig.suptitle(f"Forward Surrogate Evaluation (MSE: {mse:.2e})", fontsize=16)

    # 兼容单个样本的情况
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # 第一列：输入的结构图 (介电常数分布)
        eps_map = (structures_np[i, 0] + 1.0) / 2.0 * (cfg.eps_r_max - 1.0) + 1.0
        im0 = axes[i, 0].imshow(eps_map, cmap='plasma', origin='lower', vmin=1.0, vmax=cfg.eps_r_max)
        axes[i, 0].set_title("Input: Structure (eps_r)")
        axes[i, 0].axis('off')
        if i == 0: fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # 统一场图的颜色映射范围 (取真实场图的极值)
        vmin, vmax = real_fields_np[i].min(), real_fields_np[i].max()

        axes[i, 1].imshow(real_fields_np[i, 1], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Real: Transmission")

        axes[i, 2].imshow(real_fields_np[i, 0], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Real: Reflection")

        axes[i, 3].imshow(pred_fields_np[i, 1], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 3].set_title("AI Pred: Transmission")

        im4 = axes[i, 4].imshow(pred_fields_np[i, 0], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        axes[i, 4].set_title("AI Pred: Reflection")
        if i == 0: fig.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04, label="E-Field (V/m)")

    plt.tight_layout()
    from datetime import datetime
    output_dir = "predict_output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(output_dir, f"forward_prediction_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" 预测结果与物理误差图已存入: {save_path}")


if __name__ == "__main__":
    main()
