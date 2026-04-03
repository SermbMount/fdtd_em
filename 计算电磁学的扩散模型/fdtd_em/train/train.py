import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from fdtd_em.config import generation_config as cfg
from fdtd_em.train.dataset import MyDataset
from fdtd_em.train.model import ConditionalDDPM


def get_latest_dataset(root="dataset"):
    runs = glob.glob(os.path.join(root, "20*"))
    if not runs:
        raise FileNotFoundError("未找到数据集，请先运行 run_pipeline.py 生成数据。")
    latest_run = sorted(runs)[-1]
    return os.path.join(latest_run, "samples")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" 启动带有 PINN 物理约束的 DDPM 训练 | 设备: {device}")

    # 1. 加载数据集
    samples_dir = get_latest_dataset()
    train_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    print(f" 找到 {len(train_files)} 个训练样本。")

    dataset = MyDataset(train_files)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # 获取数据集的全局场图统计量 (用于物理损失的反归一化)
    f_min, f_max = dataset.field_min, dataset.field_max
    print(f" 物理场量级范围: {f_min:.4e} 到 {f_max:.4e} V/m")

    # 2. 初始化带物理约束的模型
    # target_dim=1 表示我们要预测的是单通道的结构图 (如果逆推结构)
    # 若你的目标是正向预测场图，target_dim 应为 2。此处以正向代理模型训练为例：
    model = ConditionalDDPM(target_dim=2, cond_dim=1, n_steps=cfg.diffusion_steps, device=device)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 3. 创建保存目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 4. 开始训练循环
    print(" 开始训练...")
    for epoch in range(cfg.epochs):
        model.train()
        total_epoch_loss = 0.0
        total_mse_loss = 0.0
        total_phys_loss = 0.0

        for batch_idx, (fields, structures, phys_conds) in enumerate(dataloader):
            fields = fields.to(device)
            structures = structures.to(device)
            phys_conds = phys_conds.to(device)

            optimizer.zero_grad()

            # 核心修改：传入 f_min, f_max 激活 Helmholtz 物理损失
            # 注意：此处假设你的正向任务是预测 fields (x0), 条件是 structures (img_cond)
            loss, mse_loss, phys_loss = model.get_loss(fields, structures, phys_conds, f_min, f_max)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
            optimizer.step()

            total_epoch_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_phys_loss += phys_loss.item()

        scheduler.step()

        # 打印详细损失
        num_batches = len(dataloader)
        avg_loss = total_epoch_loss / num_batches
        avg_mse = total_mse_loss / num_batches
        avg_phys = total_phys_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{cfg.epochs}] | LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Total Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Phys: {avg_phys:.4e})")

        # 定期保存权重
        if (epoch + 1) % 50 == 0 or epoch == cfg.epochs - 1:
            save_path = os.path.join(ckpt_dir, f"diffusion_model_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)

    # 训练结束保存最终版本
    final_path = os.path.join(ckpt_dir, "3d_fdtd_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f" 训练完成！最终权重已保存至: {final_path}")


if __name__ == "__main__":
    main()
