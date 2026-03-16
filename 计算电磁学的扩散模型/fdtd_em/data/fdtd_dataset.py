import fdtd
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class FDTDFieldDataset(Dataset):
    """
    加载每个样本的 field_map.npy 和对应 eps_r 标签。
    假设数据结构为：
    dataset/
        2025-12-17_22-47-16/
            samples/
                0000/
                    field_map.npy
                    structure.json
                0001/
                    ...
    """
    def __init__(self, dataset_root):
        self.samples = []
        # 遍历所有批次
        for batch in os.listdir(dataset_root):
            batch_path = os.path.join(dataset_root, batch, "samples")
            if not os.path.isdir(batch_path):
                continue
            for sample_name in os.listdir(batch_path):
                sample_path = os.path.join(batch_path, sample_name)
                field_path = os.path.join(sample_path, "field_map.npy")
                struct_path = os.path.join(sample_path, "structure.json")
                if os.path.isfile(field_path) and os.path.isfile(struct_path):
                    self.samples.append((field_path, struct_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        field_path, struct_path = self.samples[idx]
        field = np.load(field_path)  # shape: (H, W)
        with open(struct_path, 'r') as f:
            struct = json.load(f)
        eps_r = struct["eps_r"]

        field_tensor = torch.tensor(field, dtype=torch.float32).unsqueeze(0)  # shape: [1, H, W]
        label_tensor = torch.tensor(eps_r, dtype=torch.float32)  # shape: []
        return field_tensor, label_tensor
