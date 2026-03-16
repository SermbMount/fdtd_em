import os
import glob
import json
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import fdtd_em.config.generation_config as cfg
from fdtd_em.train.dataset import MyDataset
from fdtd_em.train.model import ConditionalDDPM


def find_latest_dataset(root="dataset"):
    #该代码优先找最新的数据进行训练
    all_runs = glob.glob(os.path.join(root, "20*/samples"))
    if not all_runs:
        raise FileNotFoundError("未找到任何样本目录，请先运行 generator.py 生成数据。")
    return sorted(all_runs)[-1]


def main():
    DATASET_ROOT = find_latest_dataset()
    data_dir = Path(DATASET_ROOT)
    train_files = [p for p in data_dir.iterdir() if p.is_dir()]

    print(f"在 {data_dir} 找到训练样本文件数: {len(train_files)}")

    train_dataset = MyDataset(train_files)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============== 开始训练 (U-Net + 物理参数注入模式) ==============")
    if device == "cuda":
        print(f" 正在使用 GPU: {torch.cuda.get_device_name(0)}")

    model = ConditionalDDPM(n_steps=cfg.diffusion_steps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        # 1：接收 3 个变量 (结构图, 场图, 物理条件)
        for batch_idx, (structures, field_maps, phys_conds) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")):

            structures = structures.to(device)
            field_maps = field_maps.to(device)
            phys_conds = phys_conds.to(device)  # 将物理条件推入显存

            optimizer.zero_grad()
            # 2：将物理条件喂给 loss 函数
            loss = model.get_loss(x_0=field_maps, img_cond=structures, phys_cond=phys_conds)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}, Step {batch_idx + 1}: 去噪 Loss = {epoch_loss / (batch_idx + 1):.6f}")

        epoch_loss /= (batch_idx + 1 if batch_idx >= 0 else 1)
        print(f"Epoch {epoch + 1} 结束: 平均 Loss = {epoch_loss:.6f}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存模型权重字典
    model_path = os.path.join(save_dir, "3d_fdtd_model.pth")
    torch.save(model.state_dict(), model_path)

    # 2. 保存训练配置记录 JSON
    config_record = {
        "device": device,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "diffusion_steps": cfg.diffusion_steps,
        "dataset": str(data_dir),
        "model_file": "3d_fdtd_model.pth",
        "model_type": "Physics_Informed_ConditionalUNet"
    }
    with open(os.path.join(save_dir, "training_record.json"), "w") as f:
        json.dump(config_record, f, indent=2)

    shutil.copy("fdtd_em/config/generation_config.py", os.path.join(save_dir, "saved_config.py"))

    print(f"训练完成！模型与配置已归档至文件夹: {save_dir}")

if __name__ == "__main__":
    main()