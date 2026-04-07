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

        # --- [新增] 全局场图数据归一化统计算法 ---
        # 获取数据集根目录（例如 dataset/2025-12-17_22-47-16/samples 的上一级）
        if len(sample_dirs) > 0:
            self.dataset_root = os.path.dirname(sample_dirs[0])
        else:
            self.dataset_root = ""

        self.stats_file = os.path.join(self.dataset_root, "field_stats.json")
        self.field_min, self.field_max = self._init_or_load_stats()

    def _init_or_load_stats(self):
        """加载或计算全局场图的极值，用于严格的 [-1, 1] 归一化"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, "r") as f:
                stats = json.load(f)
            return stats["min"], stats["max"]

        print("[Dataset] 首次加载，正在计算全局场图统计量用于归一化...")
        global_min = float('inf')
        global_max = float('-inf')

        for s_dir in self.sample_dirs:
            pt_path = os.path.join(s_dir, "field_map.pt")
            npy_path = os.path.join(s_dir, "field_map.npy")

            if os.path.exists(pt_path):
                f_map = torch.load(pt_path, weights_only=True)
            else:
                f_map = torch.from_numpy(np.load(npy_path))

            global_min = min(global_min, f_map.min().item())
            global_max = max(global_max, f_map.max().item())

        # 留出 5% 的裕量，防止测试集中出现极其罕见的异常高峰值导致越界
        margin = (global_max - global_min) * 0.05
        global_min -= margin
        global_max += margin

        stats = {"min": global_min, "max": global_max}
        with open(self.stats_file, "w") as f:
            json.dump(stats, f)

        print(f"[Dataset] 统计完成并保存至 {self.stats_file} | Min: {global_min:.4e}, Max: {global_max:.4e}")
        return global_min, global_max

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # 1. 读取 JSON
        json_path = os.path.join(sample_dir, "structure.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. 读取 structure_mask.npy，而不是自己重建结构图
        structure_path = os.path.join(sample_dir, "structure_mask.npy")
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Missing structure mask: {structure_path}")

        structure_map = np.load(structure_path).astype(np.float32)

        # 如果保存的是 [H, W]，转成 [1, H, W]
        norm_structure = torch.from_numpy(structure_map).float().unsqueeze(0)

        # 结构图归一化到 [-1, 1]
        # 当前生成器里 structure_slice = mask * eps_r_val
        # 背景是 0，目标区域是 eps_r
        norm_structure = 2.0 * norm_structure / cfg.eps_r_max - 1.0

        # 3. 读取场图
        field_map_path = os.path.join(sample_dir, "field_map.pt")
        if not os.path.exists(field_map_path):
            raise FileNotFoundError(f"Missing field map: {field_map_path}")

        field_map = torch.load(field_map_path, weights_only=True)

        # 确保为 [C, H, W]
        if field_map.dim() == 2:
            field_map = field_map.unsqueeze(0)

        field_map = field_map.float()

        # 插值对齐到统一尺寸
        field_map = F.interpolate(
            field_map.unsqueeze(0),
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 4. 物理条件向量
        src_idx = float(data["source_index"]) / cfg.Nx
        src_period = float(data["source_period"]) / cfg.src_period_max
        eps_r = float(data["eps_r"]) / cfg.eps_r_max

        phys_cond = torch.tensor([src_idx, src_period, eps_r], dtype=torch.float32)

        # 5. 场图归一化到 [-1, 1]
        field_map = 2.0 * (field_map - self.field_min) / (self.field_max - self.field_min) - 1.0

        return field_map, norm_structure, phys_cond
