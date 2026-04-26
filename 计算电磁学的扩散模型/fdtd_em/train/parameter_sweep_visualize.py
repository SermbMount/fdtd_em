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
    return (field_norm + 1.0) / 2.0 * (f_max - f_min) + f_min


@torch.no_grad()
def predict_one_sample(model, structure, phys_cond, device):
    model.eval()
    structure = structure.unsqueeze(0).to(device)   # [1,1,H,W]
    phys_cond = phys_cond.unsqueeze(0).to(device)   # [1,3]
    pred_norm = model.sample(img_cond=structure, phys_cond=phys_cond)  # [1,C,H,W]
    return pred_norm.squeeze(0).cpu()


def plot_sweep_grid(
    structure_img,
    gt_field_phys,
    preds_phys,
    labels,
    save_path,
    title_prefix="parameter_sweep"
):
    """
    structure_img: [H,W]
    gt_field_phys: [2,H,W]
    preds_phys: list of [2,H,W]
    labels: list[str]
    """
    n = len(preds_phys)

    fig, axes = plt.subplots(nrows=n, ncols=5, figsize=(18, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        pred = preds_phys[i]

        # 第一列：结构图
        axes[i, 0].imshow(structure_img, origin="lower", cmap="gray")
        axes[i, 0].set_title(f"Input Structure\n{labels[i]}")

        # 第二列：GT Plane 0
        gt0 = gt_field_phys[0]
        pred0 = pred[0]
        vmin0 = min(gt0.min(), pred0.min())
        vmax0 = max(gt0.max(), pred0.max())
        im = axes[i, 1].imshow(gt0, origin="lower", cmap="viridis", vmin=vmin0, vmax=vmax0)
        axes[i, 1].set_title("GT Plane 0")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 第三列：Pred Plane 0
        im = axes[i, 2].imshow(pred0, origin="lower", cmap="viridis", vmin=vmin0, vmax=vmax0)
        axes[i, 2].set_title("Pred Plane 0")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # 第四列：GT Plane 1
        gt1 = gt_field_phys[1]
        pred1 = pred[1]
        vmin1 = min(gt1.min(), pred1.min())
        vmax1 = max(gt1.max(), pred1.max())
        im = axes[i, 3].imshow(gt1, origin="lower", cmap="viridis", vmin=vmin1, vmax=vmax1)
        axes[i, 3].set_title("GT Plane 1")
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # 第五列：Pred Plane 1
        im = axes[i, 4].imshow(pred1, origin="lower", cmap="viridis", vmin=vmin1, vmax=vmax1)
        axes[i, 4].set_title("Pred Plane 1")
        plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)

    fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"参数扫描可视化 | device = {device}")

    # ===== 这里可改 =====
    sample_idx = 2   # 建议先用效果较好的样本，例如 2
    eps_r_sweep = [2.5, 4.0, 5.5]
    period_sweep = [18.0, 27.0, 36.0]

    # 1. 找最新数据集
    latest_dataset_dir, samples_dir = get_latest_dataset()
    all_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    print(f"最新数据集: {latest_dataset_dir}")
    print(f"总样本数: {len(all_files)}")

    # 2. 和训练一致的 train/val 切分
    train_files, val_files = split_train_val(all_files, train_ratio=0.8, seed=42)
    print(f"训练样本数: {len(train_files)} | 验证样本数: {len(val_files)}")

    val_dataset = MyDataset(val_files)

    # 3. 加载 stats
    stats_path = os.path.join(samples_dir, "field_stats.json")
    f_min, f_max = load_stats(stats_path)
    print(f"field stats: min={f_min:.4e}, max={f_max:.4e}")

    # 4. 找 checkpoint
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

    # 5. 初始化模型
    model = ConditionalDDPM(
        target_dim=2,
        cond_dim=1,
        n_steps=cfg.diffusion_steps,
        device=device
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 6. 取一个验证样本
    gt_field_norm, structure_norm, phys_cond = val_dataset[sample_idx]
    gt_field_phys = denorm_field(gt_field_norm, f_min, f_max).numpy()
    structure_img = structure_norm.squeeze(0).numpy()

    print(f"使用验证样本 sample_idx = {sample_idx}")
    print(f"原始 phys_cond = {phys_cond.tolist()}")

    output_dir = os.path.join(latest_ckpt_dir, "parameter_sweeps")
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================
    # 第一组：eps_r 扫描
    # ==========================================================
    preds_eps = []
    labels_eps = []

    print("\n开始 eps_r 扫描:")
    for eps_r in eps_r_sweep:
        test_phys = phys_cond.clone()
        test_phys[2] = float(eps_r / cfg.eps_r_max)

        pred_norm = predict_one_sample(model, structure_norm, test_phys, device)
        pred_phys = denorm_field(pred_norm, f_min, f_max).numpy()

        preds_eps.append(pred_phys)
        labels_eps.append(f"eps_r = {eps_r:.2f}")

        print(f"  eps_r = {eps_r:.2f} -> phys_cond = {test_phys.tolist()}")

    eps_save_path = os.path.join(output_dir, f"sample_{sample_idx:03d}_eps_r_sweep.png")
    plot_sweep_grid(
        structure_img=structure_img,
        gt_field_phys=gt_field_phys,
        preds_phys=preds_eps,
        labels=labels_eps,
        save_path=eps_save_path,
        title_prefix=f"Parameter Sweep: eps_r | Val sample {sample_idx}"
    )

    # ==========================================================
    # 第二组：source_period 扫描
    # ==========================================================
    preds_period = []
    labels_period = []

    print("\n开始 source_period 扫描:")
    for period in period_sweep:
        test_phys = phys_cond.clone()
        test_phys[1] = float(period / cfg.src_period_max)

        pred_norm = predict_one_sample(model, structure_norm, test_phys, device)
        pred_phys = denorm_field(pred_norm, f_min, f_max).numpy()

        preds_period.append(pred_phys)
        labels_period.append(f"period = {period:.2f}")

        print(f"  source_period = {period:.2f} -> phys_cond = {test_phys.tolist()}")

    period_save_path = os.path.join(output_dir, f"sample_{sample_idx:03d}_source_period_sweep.png")
    plot_sweep_grid(
        structure_img=structure_img,
        gt_field_phys=gt_field_phys,
        preds_phys=preds_period,
        labels=labels_period,
        save_path=period_save_path,
        title_prefix=f"Parameter Sweep: source_period | Val sample {sample_idx}"
    )

    print("\n参数扫描完成。")
    print(f"eps_r 扫描图: {eps_save_path}")
    print(f"source_period 扫描图: {period_save_path}")


if __name__ == "__main__":
    main()


#python -m fdtd_em.train.parameter_sweep_visualize
