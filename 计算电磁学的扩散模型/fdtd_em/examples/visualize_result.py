import os
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入我们的核心模型和数据集
from fdtd_em.train.model import ConditionalDDPM
from fdtd_em.train.dataset import MyDataset, BOUNDS


def find_latest_file(folder, extension=""):
    """寻找最新生成的文件或目录"""
    files = glob.glob(os.path.join(folder, f"*{extension}"))
    if not files: return None
    return sorted(files)[-1]


def create_epsilon_map(cx, cy, r, eps, grid_size=120):
    """
    创建一个 2D 介电常数分布切片用于直观可视化
    背景设为真空 (eps=1.0)，圆内设为对应的介电常数。
    """
    eps_map = np.ones((grid_size, grid_size))
    y, x = np.ogrid[0:grid_size, 0:grid_size]
    # 计算圆内区域 (模拟球体截面)
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    eps_map[mask] = eps
    return eps_map


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 启动结果可视化脚本 (设备: {device})")

    # 1. 自动定位并加载最新的扩散模型权重
    latest_model_path = find_latest_file("checkpoints", "diffusion_model_*.pth")
    if not latest_model_path:
        print("❌ 找不到模型权重，请先完成训练！")
        return

    model = ConditionalDDPM(target_dim=5, n_steps=1000, device=device)
    model.load_state_dict(torch.load(latest_model_path, map_location=device, weights_only=True))
    model.to(device)

    # 2. 自动定位最新数据集并拿取测试样本
    latest_dataset_run = find_latest_file("dataset")
    samples_dir = os.path.join(latest_dataset_run, "samples")
    train_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    dataset = MyDataset(train_files)

    # 抽取一个样本
    real_structure, field_map = dataset[0]
    field_map_cond = field_map.unsqueeze(0).to(device)

    # 3. 运行扩散模型逆向采样
    print("✨ 正在运行扩散模型逆向生成，请稍候...")
    predicted_structure = model.sample(condition=field_map_cond)

    # 4. 数据反归一化 (将 [-1, 1] 放大回真实的物理范围)
    p_min = BOUNDS[:, 0].to(device)
    p_max = BOUNDS[:, 1].to(device)
    denorm_pred = (predicted_structure[0] + 1.0) / 2.0 * (p_max - p_min) + p_min
    denorm_real = (real_structure.to(device) + 1.0) / 2.0 * (p_max - p_min) + p_min

    # 转为 numpy 方便画图
    pred_vals = denorm_pred.cpu().numpy()
    real_vals = denorm_real.cpu().numpy()

    # 5. ================= 开始使用 Matplotlib 绘制对比大图 =================
    print("🎨 正在绘制可视化对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # 1行3列的宽图

    # [图 1]：模型输入的条件 (Target Field Map)
    ax1 = axes[0]
    im1 = ax1.imshow(field_map.numpy(), cmap='viridis', origin='lower')
    ax1.set_title("Input Condition\n(Target 2D Field Map)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Z-axis grid")
    ax1.set_ylabel("Y-axis grid")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Electric Field Intensity")

    # [图 2]：真实的物理结构 (Ground Truth)
    r_cx, r_cy, r_cz, r_r, r_eps = real_vals
    ax2 = axes[1]
    real_eps_map = create_epsilon_map(r_cx, r_cy, r_r, r_eps)
    # 统一使用 plasma 颜色条映射介电常数 1 到 6
    im2 = ax2.imshow(real_eps_map, cmap='plasma', origin='lower', vmin=1.0, vmax=6.0)
    title_real = f"Ground Truth Structure\n" \
                 f"eps = {r_eps:.2f}, R = {r_r:.1f}\n" \
                 f"Center = ({r_cx:.0f}, {r_cy:.0f}, {r_cz:.0f})"
    ax2.set_title(title_real, fontsize=12, color='green')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Relative Permittivity (eps_r)")

    # [图 3]：扩散模型逆向预测的物理结构 (Predicted)
    p_cx, p_cy, p_cz, p_r, p_eps = pred_vals
    ax3 = axes[2]
    pred_eps_map = create_epsilon_map(p_cx, p_cy, p_r, p_eps)
    im3 = ax3.imshow(pred_eps_map, cmap='plasma', origin='lower', vmin=1.0, vmax=6.0)
    title_pred = f"Predicted Structure\n" \
                 f"eps = {p_eps:.2f}, R = {p_r:.1f}\n" \
                 f"Center = ({p_cx:.0f}, {p_cy:.0f}, {p_cz:.0f})"
    # 用红色标题突出预测结果
    ax3.set_title(title_pred, fontsize=12, color='darkred')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Relative Permittivity (eps_r)")

    plt.tight_layout()

    # 保存高清图并弹出展示
    save_path = "diffusion_inverse_design_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化图像已保存至项目根目录的: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

    #左侧：是复杂的电磁散射场图（这代表了客户给定的性能指标/目标）。
    #中间：是标准的真实介电常数小球切面。
    #右侧：是你的 AI 模型从零基础乱码中自己“推算”出来的物理结构。