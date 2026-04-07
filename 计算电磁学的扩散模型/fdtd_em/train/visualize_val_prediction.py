import os
import glob
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fdtd_em.config import generation_config as cfg
from fdtd_em.train.dataset import MyDataset
from fdtd_em.train.model import ConditionalDDPM


def get_latest_dataset(root="dataset"):
    runs = glob.glob(os.path.join(root, "20*"))
    if not runs:
        raise FileNotFoundError("未找到数据集，请先运行数据生成。")
    latest_run = sorted(runs)[-1]
    return latest_run, os.path.join(latest_run, "samples")


def get_latest_checkpoint_dir(root="checkpoints"):
    runs = glob.glob(os.path.join(root, "20*"))
    if not runs:
        raise FileNotFoundError("未找到 checkpoint，请先训练模型。")
    return sorted(runs)[-1]


def split_train_val(sample_dirs, train_ratio=0.8, seed=42):
    sample_dirs = sorted(sample_dirs)
    random.seed(seed)
    random.shuffle(sample_dirs)

    split_idx = int(len(sample_dirs) * train_ratio)
    train_files = sample_dirs[:split_idx]
    val_files = sample_dirs[split_idx:]

    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError(
            f"训练/验证切分失败：总样本数={len(sample_dirs)}，"
            f"train={len(train_files)}，val={len(val_files)}"
        )
    return train_files, val_files


def load_stats(stats_path):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"未找到 field_stats.json: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats["min"], stats["max"]


def denorm_field(field_norm, f_min, f_max):
    """
    field_norm: torch.Tensor [C,H,W] in [-1,1]
    return: torch.Tensor [C,H,W] in physical scale
    """
    return (field_norm + 1.0) / 2.0 * (f_max - f_min) + f_min


@torch.no_grad()
def predict_one_sample(model, structure, phys_cond, device):
    """
    structure: [1,H,W]
    phys_cond: [3]
    return pred_norm: [C,H,W]
    """
    model.eval()

    structure = structure.unsqueeze(0).to(device)   # [1,1,H,W]
    phys_cond = phys_cond.unsqueeze(0).to(device)   # [1,3]

    pred_norm = model.sample(img_cond=structure, phys_cond=phys_cond)  # [1,C,H,W]
    return pred_norm.squeeze(0).cpu()


def plot_six_panels(gt_field, pred_field, save_path, title_prefix="val_sample"):
    """
    gt_field, pred_field: numpy arrays [2,H,W], physical scale
    """
    err_field = np.abs(pred_field - gt_field)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    plane_titles = ["Plane 0", "Plane 1"]

    for i in range(2):
        gt = gt_field[i]
        pred = pred_field[i]
        err = err_field[i]

        # 为了公平对比，GT 和 Pred 用同一个颜色范围
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        im0 = axes[i, 0].imshow(gt, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{plane_titles[i]} - GT")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{plane_titles[i]} - Pred")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(err, origin="lower", cmap="magma")
        axes[i, 2].set_title(f"{plane_titles[i]} - Error |Pred-GT|")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"可视化验证样本 | device = {device}")

    # 1. 找最新数据集
    latest_dataset_dir, samples_dir = get_latest_dataset()
    all_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    print(f"最新数据集: {latest_dataset_dir}")
    print(f"总样本数: {len(all_files)}")

    # 2. 和训练时一致的 train/val 切分
    train_files, val_files = split_train_val(all_files, train_ratio=0.8, seed=42)
    print(f"训练样本数: {len(train_files)} | 验证样本数: {len(val_files)}")

    # 3. 构造 val dataset
    val_dataset = MyDataset(val_files)

    # 4. 加载场图归一化统计量
    stats_path = os.path.join(samples_dir, "field_stats.json")
    f_min, f_max = load_stats(stats_path)
    print(f"field stats: min={f_min:.4e}, max={f_max:.4e}")

    # 5. 找最新 checkpoint，并优先加载 best_model.pth
    latest_ckpt_dir = get_latest_checkpoint_dir()
    best_model_path = os.path.join(latest_ckpt_dir, "best_model.pth")
    final_model_path = os.path.join(latest_ckpt_dir, "3d_fdtd_model.pth")

    if os.path.exists(best_model_path):
        ckpt_path = best_model_path
    elif os.path.exists(final_model_path):
        ckpt_path = final_model_path
    else:
        raise FileNotFoundError(
            f"在 {latest_ckpt_dir} 下既没找到 best_model.pth，也没找到 3d_fdtd_model.pth"
        )

    print(f"加载模型权重: {ckpt_path}")

    # 6. 初始化模型
    model = ConditionalDDPM(
        target_dim=2,
        cond_dim=1,
        n_steps=cfg.diffusion_steps,
        device=device
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 7. 从验证集取一个样本
    sample_idx = 0
    gt_field_norm, structure_norm, phys_cond = val_dataset[sample_idx]

    print("gt_field_norm.shape =", gt_field_norm.shape)
    print("structure_norm.shape =", structure_norm.shape)
    print("phys_cond =", phys_cond)

    # 8. 预测
    pred_field_norm = predict_one_sample(model, structure_norm, phys_cond, device)

    # 9. 反归一化到物理量
    gt_field_phys = denorm_field(gt_field_norm, f_min, f_max).numpy()
    pred_field_phys = denorm_field(pred_field_norm, f_min, f_max).numpy()

    # 10. 保存图像
    output_dir = os.path.join(latest_ckpt_dir, "val_visualizations")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"val_sample_{sample_idx:03d}_gt_pred_error.png")
    title_prefix = f"Dataset: {os.path.basename(latest_dataset_dir)} | Checkpoint: {os.path.basename(latest_ckpt_dir)} | Val sample {sample_idx}"
    plot_six_panels(gt_field_phys, pred_field_phys, save_path, title_prefix=title_prefix)

    print(f"6张对比图已保存至: {save_path}")


if __name__ == "__main__":
    main()
