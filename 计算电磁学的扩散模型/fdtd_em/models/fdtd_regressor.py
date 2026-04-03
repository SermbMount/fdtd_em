import os
import json
import glob
import torch
import numpy as np
from pathlib import Path

from fdtd_em.config import generation_config as cfg
from fdtd_em.train.model import ConditionalDDPM
from fdtd_em.train.dataset import MyDataset


class DiffusionInverseSolver:
    """
    基于条件扩散模型的超构表面/散射体逆向设计求解器
    """

    def __init__(self, checkpoints_dir="checkpoints", dataset_dir="dataset", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f" 初始化逆向求解器 | 设备: {self.device}")

        # 加载最新模型
        latest_ckpt_dir = self._find_latest(checkpoints_dir)
        model_path = os.path.join(latest_ckpt_dir, "3d_fdtd_model.pth")

        # 逆向设计目标：生成单通道结构图(target_dim=1)，条件为双通道场图(cond_dim=2)
        self.model = ConditionalDDPM(target_dim=1, cond_dim=2, n_steps=cfg.diffusion_steps, device=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # 加载归一化统计量
        latest_dataset = self._find_latest(dataset_dir)
        self.stats_path = os.path.join(latest_dataset, "field_stats.json")
        self.f_min, self.f_max = self._load_stats()

    def _find_latest(self, root):
        runs = glob.glob(os.path.join(root, "20*"))
        if not runs:
            raise FileNotFoundError(f"未找到目录: {root}")
        return sorted(runs)[-1]

    def _load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                stats = json.load(f)
            return stats['min'], stats['max']
        return -1.0, 1.0

    def normalize_target_field(self, field_tensor):
        """将物理目标场强转换为网络所需的 [-1, 1] 空间"""
        return 2.0 * (field_tensor - self.f_min) / (self.f_max - self.f_min) - 1.0

    @torch.no_grad()
    def solve(self, target_field, phys_cond=None, binarize=True):
        """
        执行逆向设计
        :param target_field: [1, 2, H, W] 物理目标场图张量 (V/m)
        :param phys_cond: [1, 2] 物理条件 (如光源位置、周期)
        :param binarize: 是否将生成的连续介电常数离散化为物理实体 (真空 vs 介质)
        :return: [H, W] 的物理介电常数矩阵
        """
        # 1. 归一化输入条件
        cond_norm = self.normalize_target_field(target_field).to(self.device)
        if phys_cond is not None:
            phys_cond = phys_cond.to(self.device)

        # 2. DDPM 逆向推演
        print(" 正在执行 DDPM 物理推演 (约耗时数秒)...")
        pred_struct_norm = self.model.sample(img_cond=cond_norm, phys_cond=phys_cond)

        # 3. 反归一化到物理介电常数
        pred_eps = (pred_struct_norm.squeeze().cpu().numpy() + 1.0) / 2.0 * (cfg.eps_r_max - 1.0) + 1.0

        # 4. 物理二值化 (制造可加工的超表面)
        if binarize:
            # 设定阈值，例如大于中间值的认为是介质，小于的认为是真空
            threshold = (cfg.eps_r_max + 1.0) / 2.0
            pred_eps[pred_eps >= threshold] = cfg.eps_r_max
            pred_eps[pred_eps < threshold] = 1.0

        return pred_eps


# ================= 测试脚本 =================
if __name__ == "__main__":
    
    solver = DiffusionInverseSolver()

    # 获取一个测试数据
    latest_dataset = solver._find_latest("dataset")
    samples_dir = os.path.join(latest_dataset, "samples")
    test_files = [p for p in Path(samples_dir).iterdir() if p.is_dir()]
    dataset = MyDataset(test_files[:1])  # 取第一个

    # 模拟“提供的目标场图” (从真实的 Dataset 中取出来的原本是 [-1,1] 的，我们要假装它是物理量)
    target_field_norm, gt_structure, phys_cond = dataset[0]

    # 转换为物理电场 V/m 作为输入目标
    target_field_phys = (target_field_norm + 1.0) / 2.0 * (solver.f_max - solver.f_min) + solver.f_min
    target_field_phys = target_field_phys.unsqueeze(0)  # [1, 2, H, W]

    # 执行求解
    predicted_eps_matrix = solver.solve(target_field_phys, phys_cond=phys_cond.unsqueeze(0), binarize=True)

    # 导出为 numpy 数组，这个数组可以直接被 FDTD 的 Grid 对象读取来进行二次验证！
    output_path = "inverse_designed_structure.npy"
    np.save(output_path, predicted_eps_matrix)
    print(f" 逆向设计成功！可加工的介电常数矩阵已保存至: {output_path}")
