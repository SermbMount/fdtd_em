import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from fdtd_em.config import generation_config as cfg

class MyDataset(Dataset):
    def __init__(self, sample_dirs):
        self.sample_dirs = sample_dirs
        self.grid_size = cfg.Nx

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # 1. 读取 JSON
        with open(os.path.join(sample_dir, "structure.json"), "r") as f:
            data = json.load(f)

        src_idx = float(data["source_index"]) / cfg.Nx
        src_period = float(data["source_period"]) / cfg.src_period_max
        phys_cond = torch.tensor([src_idx, src_period], dtype=torch.float32)

        # 2. 绘制结构图 (Target)
        structure_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        y, x = np.ogrid[0:self.grid_size, 0:self.grid_size]
        mask = (x - data["center"][0]) ** 2 + (y - data["center"][1]) ** 2 <= data["radius"] ** 2
        structure_map[mask] = data["eps_r"]

        norm_structure = 2.0 * (torch.from_numpy(structure_map) - 1.0) / (cfg.eps_r_max - 1.0) - 1.0
        norm_structure = norm_structure.unsqueeze(0) # [1, H, W]

        # 3. 读取场图 (支持多通道加载)
        field_map_path = os.path.join(sample_dir, "field_map.pt")
        if os.path.exists(field_map_path):
            field_map = torch.load(field_map_path, weights_only=True)
        else:
            field_map = torch.from_numpy(np.load(os.path.join(sample_dir, "field_map.npy")))

        # 确保是 [C, H, W] 格式
        if field_map.dim() == 2:
            field_map = field_map.unsqueeze(0)

        # 插值对齐
        field_map = F.interpolate(
            field_map.unsqueeze(0),
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        for c in range(field_map.shape[0]):
            c_min, c_max = field_map[c].min(), field_map[c].max()
            if c_max > c_min:
                field_map[c] = 2.0 * (field_map[c] - c_min) / (c_max - c_min) - 1.0
            else:
                field_map[c] = field_map[c] * 0.0

        return norm_structure.float(), field_map.float(), phys_cond