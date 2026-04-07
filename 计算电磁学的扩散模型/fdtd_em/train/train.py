import os
import glob
import random
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
            f"train={len(train_files)}，val={len(val_files)}。"
            f"请确保样本数足够，建议至少 > 10。"
        )

    return train_files, val_files


def evaluate(model, dataloader, device, f_min, f_max):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0

    with torch.no_grad():
        for fields, structures, phys_conds in dataloader:
            fields = fields.to(device)
            structures = structures.to(device)
            phys_conds = phys_conds.to(device)

            loss, mse_loss, phys_loss = model.get_loss(fields, structures, phys_conds, f_min, f_max)

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_phys += phys_loss.item()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("验证集 DataLoader 的 batch 数为 0，请检查 batch_size 或样本数。")

    return (
        total_loss / num_batches,
        total_mse / num_batches,
        total_phys / num_batches,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" 启动带有 PINN 物理约束的 DDPM 训练 | 设备: {device}")

    # 1. 加载最新数据集
    samples_dir = get_latest_dataset()
    all_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    print(f" 找到 {len(all_files)} 个总样本。")

    # 2. 切分训练集 / 验证集
    train_files, val_files = split_train_val(all_files, train_ratio=0.8, seed=42)
    print(f" 训练样本数: {len(train_files)} | 验证样本数: {len(val_files)}")

    train_dataset = MyDataset(train_files)
    val_dataset = MyDataset(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False
    )

    train_num_batches = len(train_loader)
    val_num_batches = len(val_loader)
    print(f" Train batch 数: {train_num_batches} | Val batch 数: {val_num_batches}")

    if train_num_batches == 0:
        raise ValueError(
            f"Train DataLoader produced 0 batches. "
            f"dataset size={len(train_dataset)}, batch_size={cfg.batch_size}"
        )
    if val_num_batches == 0:
        raise ValueError(
            f"Val DataLoader produced 0 batches. "
            f"dataset size={len(val_dataset)}, batch_size={cfg.batch_size}"
        )

    # 3. 获取归一化统计量
    # train_dataset / val_dataset 来自同一批次目录，field_stats.json 应一致
    f_min, f_max = train_dataset.field_min, train_dataset.field_max
    print(f" 物理场量级范围: {f_min:.4e} 到 {f_max:.4e} V/m")

    # 4. 初始化模型
    model = ConditionalDDPM(
        target_dim=2,
        cond_dim=1,
        n_steps=cfg.diffusion_steps,
        device=device
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 5. 创建保存目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_mse = float("inf")

    # 6. 开始训练
    print(" 开始训练...")
    for epoch in range(cfg.epochs):
        model.train()
        total_epoch_loss = 0.0
        total_mse_loss = 0.0
        total_phys_loss = 0.0

        for batch_idx, (fields, structures, phys_conds) in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                print("fields.shape =", fields.shape)
                print("structures.shape =", structures.shape)
                print("phys_conds.shape =", phys_conds.shape)

            fields = fields.to(device)
            structures = structures.to(device)
            phys_conds = phys_conds.to(device)

            optimizer.zero_grad()

            loss, mse_loss, phys_loss = model.get_loss(fields, structures, phys_conds, f_min, f_max)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_phys_loss += phys_loss.item()

        scheduler.step()

        train_avg_loss = total_epoch_loss / train_num_batches
        train_avg_mse = total_mse_loss / train_num_batches
        train_avg_phys = total_phys_loss / train_num_batches

        val_avg_loss, val_avg_mse, val_avg_phys = evaluate(
            model, val_loader, device, f_min, f_max
        )

        if val_avg_mse < best_val_mse:
            best_val_mse = val_avg_mse
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{cfg.epochs}] | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Train Loss: {train_avg_loss:.4f} (MSE: {train_avg_mse:.4f}, Phys: {train_avg_phys:.4e}) | "
                f"Val Loss: {val_avg_loss:.4f} (MSE: {val_avg_mse:.4f}, Phys: {val_avg_phys:.4e})"
            )

        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            save_path = os.path.join(ckpt_dir, f"diffusion_model_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)

    final_path = os.path.join(ckpt_dir, "3d_fdtd_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f" 训练完成！最佳验证模型已保存至: {os.path.join(ckpt_dir, 'best_model.pth')}")
    print(f" 最终权重已保存至: {final_path}")


if __name__ == "__main__":
    main()
