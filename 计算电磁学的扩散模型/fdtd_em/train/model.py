import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fdtd_em.config import generation_config as cfg


# ==========================================
# 1. 物理验证：
# ==========================================
class HelmholtzPhysicsLoss(nn.Module):
    def __init__(self, dx=cfg.dx, dy=cfg.dy, freq=cfg.freq_center):
        super().__init__()
        self.dx = dx
        self.dy = dy
        # 真空中光速与角频率
        c = 3e8
        self.omega = 2 * math.pi * freq
        self.mu_0 = 4 * math.pi * 1e-7
        self.eps_0 = 8.854e-12
        self.k0_sq = (self.omega ** 2) * self.mu_0 * self.eps_0

        # 拉普拉斯算子卷积核 (Laplacian Kernel) 提取空间二阶导数
        laplacian_kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # 除以网格步长的平方得到真实的二阶空间偏导
        self.register_buffer('laplacian_kernel', laplacian_kernel / (dx * dy))

    def forward(self, E_field, eps_r):
        """
        计算二维非齐次波动方程的物理残差: \nabla^2 E + k0^2 * eps_r * E = 0
        E_field: 真实的物理电场强度 [B, C, H, W]
        eps_r: 真实的相对介电常数分布 [B, 1, H, W]
        """
        # 对每一个通道（透射/反射）分别求拉普拉斯
        B, C, H, W = E_field.shape
        loss_phys = 0.0

        for i in range(C):
            E_c = E_field[:, i:i + 1, :, :]
            # 计算空间二阶导数 \nabla^2 E
            laplacian_E = F.conv2d(E_c, self.laplacian_kernel, padding=1)

            # 计算物理方程残差: Residual = \nabla^2 E + k0^2 * eps_r * E
            residual = laplacian_E + self.k0_sq * eps_r * E_c

            # 采用 L2 范数作为物理惩罚项
            loss_phys += torch.mean(residual ** 2)

        return loss_phys / C


# ==========================================
# 2. 核心扩散模型架构 (带物理约束)
# ==========================================
class ConditionalDDPM(nn.Module):
    def __init__(self, target_dim=2, cond_dim=1, n_steps=1000, device="cuda"):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.target_dim = target_dim

        # 物理条件向量编码器：phys_cond = [src_idx, src_period, eps_r]
        self.phys_embed_dim = 16
        self.phys_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, self.phys_embed_dim),
        )

        # 输入通道 = 当前噪声图 + 结构条件图 + 物理条件特征图
        self.network = UNet(
            in_channels=target_dim + cond_dim + self.phys_embed_dim,
            out_channels=target_dim
        )

        self.physics_loss_fn = HelmholtzPhysicsLoss()

        # DDPM 的 Beta Schedule 设定
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_loss(self, x0, img_cond, phys_cond, f_min, f_max):
        """
        带物理约束 (PINN) 的前向加噪与损失计算
        x0: [B, C, H, W] 归一化后的场图 (-1 到 1)
        img_cond: [B, 1, H, W] 归一化后的结构图 (-1 到 1)
        """
        B = x0.shape[0]
        # 1. 随机采样时间步 t 和噪声 epsilon
        t = torch.randint(0, self.n_steps, (B,), device=self.device)
        noise = torch.randn_like(x0)

        # 2. 计算 q(x_t | x_0) (前向加噪过程)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # 3. 拼接图像条件，给 U-Net 预测噪声
        # 注意：此处你可以将 phys_cond (1D 向量) 融入 U-Net 时间嵌入，此处简化为只拼图像
        net_input = torch.cat([xt, img_cond], dim=1)
        pred_noise = self.network(net_input, t)

        # 4. 数据拟合损失 (Data Loss - MSE)
        loss_mse = F.mse_loss(pred_noise, noise)

        # 5. ============ 物理约束损失 (PINN Loss) ============
        # 从预测的噪声反推当前对纯净物理场 x0 的预测
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        # 将无量纲的预测张量反归一化为真实的物理量
        E_field_phys = (x0_pred + 1.0) / 2.0 * (f_max - f_min) + f_min
        eps_r_phys = (img_cond + 1.0) / 2.0 * (cfg.eps_r_max - 1.0) + 1.0

        # 计算麦克斯韦/亥姆霍兹残差
        loss_phys = self.physics_loss_fn(E_field_phys, eps_r_phys)

        # 自适应权重：随着网络逐渐收敛，增加物理损失的权重
        lambda_phys = 1e-8  # 物理量级较大，需要极小的系数对齐梯度
        total_loss = loss_mse + lambda_phys * loss_phys

        return total_loss, loss_mse, loss_phys

    @torch.no_grad()
    def sample(self, img_cond, phys_cond=None):
        """
        基于条件的逆向去噪采样过程 (推演物理分布)
        """
        B = img_cond.shape[0]
        x = torch.randn((B, self.target_dim, img_cond.shape[2], img_cond.shape[3]), device=self.device)

        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

            # 拼接结构图作为条件引导
            net_input = torch.cat([x, img_cond], dim=1)
            pred_noise = self.network(net_input, t_tensor)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            # DDPM 采样公式
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        return x


# ==========================================
# 3. 支撑架构：标准的 U-Net 实现 (已修复维度对称性)
# ==========================================
class UNet(nn.Module):
    """
    轻量化的条件 U-Net (包含时间步的 Embedding)
    """

    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 编码器 (Downsampling)
        self.down1 = self.block(in_channels, 64)
        self.down2 = self.block(64, 128)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层 (Bottleneck)
        self.bot1 = self.block(128, 256)
        self.bot2 = self.block(256, 256)

        # 解码器 1 (Upsampling H/4 -> H/2)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv1 = self.block(256, 128)  # 128(来自up) + 128(来自skip) = 256

        # 解码器 2 (Upsampling H/2 -> H)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv2 = self.block(128, 64)  # 64(来自up) + 64(来自skip) = 128

        self.out = nn.Conv2d(64, out_channels, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        # 时间步编码
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.time_mlp(t)[:, :, None, None]

        # 编码
        d1 = self.down1(x)  # [B, 64, 120, 120]
        d2 = self.down2(self.pool(d1))  # [B, 128, 60, 60]

        # 瓶颈层融合时间信息
        b1 = self.bot1(self.pool(d2)) + t_emb  # [B, 256, 30, 30]
        b2 = self.bot2(b1)  # [B, 256, 30, 30]

        # 解码与跳跃连接 1
        u1 = self.up1(b2)  # [B, 128, 60, 60]
        u1 = torch.cat([u1, d2], dim=1)  # 拼接 d2 -> [B, 256, 60, 60]
        u1 = self.up_conv1(u1)  # [B, 128, 60, 60]

        # 解码与跳跃连接 2
        u2 = self.up2(u1)  # [B, 64, 120, 120]
        u2 = torch.cat([u2, d1], dim=1)  # 拼接 d1 -> [B, 128, 120, 120]
        u2 = self.up_conv2(u2)  # [B, 64, 120, 120]

        return self.out(u2)

    #$$L_{total} = ||\epsilon - \epsilon_\theta||^2 + \lambda_{phys} ||\nabla^2 \hat{E}_0 + k_0^2 \epsilon_r \hat{E}_0||^2$$
