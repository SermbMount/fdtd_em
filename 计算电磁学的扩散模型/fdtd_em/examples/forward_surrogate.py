import os
import glob
import torch
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
    return sorted(all_runs)[-1]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"启动正向物理预测测试 | 设备: {device}")

    # 1. 加载最新的模型和数据
    latest_ckpt_dir = find_latest_checkpoint()
    model_path = os.path.join(latest_ckpt_dir, "3d_fdtd_model.pth")

    data_dir = Path(find_latest_dataset())
    test_files = [p for p in data_dir.iterdir() if p.is_dir()][:4]  # 取前4个样本测试
    test_dataset = MyDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 2. 初始化网络并加载权重
    model = ConditionalDDPM(n_steps=cfg.diffusion_steps, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. 提取数据
    structures, real_fields, phys_conds = next(iter(test_loader))
    structures = structures.to(device)
    real_fields = real_fields.to(device)
    phys_conds = phys_conds.to(device)

    # 4. 让进行正向预测 (看结构，画场图)
    print("正在导入电磁波的传播规律...")
    pred_fields = model.sample(img_cond=structures, phys_cond=phys_conds)

    # 5. 可视化对比 (Ground Truth vs AI Prediction)
    structures_np = structures.cpu().numpy()
    real_fields_np = real_fields.cpu().numpy()
    pred_fields_np = pred_fields.cpu().numpy()

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(4):
        # 第一列：输入的结构图 (AI 的考题)
        axes[i, 0].imshow(structures_np[i, 0], cmap='gray', origin='lower')
        axes[i, 0].set_title("Input: Structure")
        axes[i, 0].axis('off')

        # 第二列：真实的透射场 (FDTD 算出来的)
        axes[i, 1].imshow(real_fields_np[i, 1], cmap='viridis', origin='lower')
        axes[i, 1].set_title("Real: Transmission")

        # 第三列：真实的反射场
        axes[i, 2].imshow(real_fields_np[i, 0], cmap='viridis', origin='lower')
        axes[i, 2].set_title("Real: Reflection")

        # 第四列：AI 预测的透射场
        axes[i, 3].imshow(pred_fields_np[i, 1], cmap='viridis', origin='lower')
        axes[i, 3].set_title("AI Pred: Transmission")

        # 第五列：AI 预测的反射场
        axes[i, 4].imshow(pred_fields_np[i, 0], cmap='viridis', origin='lower')
        axes[i, 4].set_title("AI Pred: Reflection")

    plt.tight_layout()
    from datetime import datetime
    output_dir = "predict_output"
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在就自动建一个
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(output_dir, f"forward_prediction_{timestamp}.png")
    plt.savefig(save_path, dpi=200)
    print(f"预测结果对比图已存入: {save_path}")


if __name__ == "__main__":
    main()