import os
import torch
import glob
import matplotlib.pyplot as plt
import datetime

from pathlib import Path
from fdtd_em.config import generation_config as cfg
from fdtd_em.train.model import ConditionalDDPM
from fdtd_em.train.dataset import MyDataset


def find_latest_file(folder, extension=""):
    files = glob.glob(os.path.join(folder, f"*{extension}"))
    if not files: return None
    return sorted(files)[-1]


def plot_and_save(field_map, real_map, pred_map):
    print("正在生成并保存【物理约束图生图】可视化对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    im1 = ax1.imshow(field_map.squeeze().cpu().numpy(), cmap='viridis', origin='lower')
    ax1.set_title("Input: 2D Field Map", fontsize=14, fontweight='bold')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(real_map.squeeze().cpu().numpy(), cmap='plasma', origin='lower', vmin=1.0, vmax=cfg.eps_r_max)
    ax2.set_title("Ground Truth: Structure Image", fontsize=14, color='green')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = axes[2]
    im3 = ax3.imshow(pred_map.squeeze().cpu().numpy(), cmap='plasma', origin='lower', vmin=1.0, vmax=cfg.eps_r_max)
    ax3.set_title("Predicted: Physics-Informed AI", fontsize=14, color='darkred')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("predict_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)  # 如果 predict_output 不存在会自动创建

    save_path = os.path.join(output_dir, "ai_prediction_result.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"【物理约束图生图】可视化图像已保存至: {save_path}")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" 物理参数注入：逆向设计推理 (设备: {device})")

    latest_model_path = find_latest_file("checkpoints", "diffusion_model_*.pth")

    model = ConditionalDDPM(n_steps=cfg.diffusion_steps, device=device)
    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path, map_location=device, weights_only=True))
    model.to(device)

    latest_dataset_run = find_latest_file("dataset")
    samples_dir = os.path.join(latest_dataset_run, "samples")
    train_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    dataset = MyDataset(train_files)

    # 解包 3 个返回值
    real_structure_img, field_map, phys_cond = dataset[0]

    field_map_cond = field_map.unsqueeze(0).to(device)
    # 给 phys_cond 加上 batch 维度并推入显存
    phys_cond_input = phys_cond.unsqueeze(0).to(device)

    print(f" 开始U-Net(带物理约束)扩散模型逆向去噪采样 ({cfg.diffusion_steps}步)...")

    # 推理时必须同时提供场图和物理参数
    predicted_structure_img = model.sample(img_cond=field_map_cond, phys_cond=phys_cond_input)

    denorm_pred = (predicted_structure_img[0] + 1.0) / 2.0 * (cfg.eps_r_max - 1.0) + 1.0
    denorm_real = (real_structure_img + 1.0) / 2.0 * (cfg.eps_r_max - 1.0) + 1.0

    plot_and_save(field_map, denorm_real, denorm_pred)


if __name__ == "__main__":
    main()