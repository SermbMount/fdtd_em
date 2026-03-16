import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.double_conv(x)


class ConditionalUNet(nn.Module):
    # 现在的条件(cond)是结构图，结构图只有 1 个通道(只表示介电常数)，所以默认=1
    def __init__(self, cond_channels=1):
        super().__init__()

        # 第一层卷积的输入通道。
        # 画布现在是双通道的场图(加了噪声的)，所以是 2。加上参考图(1通道的结构图)，总共就是 2 + 1 = 3。
        self.inc = DoubleConv(2 + cond_channels, 64)

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.phys_mlp = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 512))
        self.time_mlp = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        # 网络最终输出预测的“噪声”。
        # 因为我们的目标画布是双通道场图，所以预测出来的去噪图也必须是 2 个通道！(以前这里是 1)
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x_t, t, img_condition, phys_condition):
        # 合并画布和参考图。
        # x_t (带噪声的场图): [Batch, 2, H, W]
        # img_condition (结构图): [Batch, 1, H, W]
        # 拼在一起后送入网络，形状变成 [Batch, 3, H, W]
        x = torch.cat([x_t, img_condition], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        t_emb = self.time_mlp(t).view(-1, 512, 1, 1)
        p_emb = self.phys_mlp(phys_condition).view(-1, 512, 1, 1)
        x4 = x4 + t_emb + p_emb

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        return self.outc(x)


class ConditionalDDPM(nn.Module):
    def __init__(self, n_steps=1000, device="cuda"):
        super().__init__()
        self.device = device
        self.n_steps = n_steps
        self.denoise_net = ConditionalUNet().to(device)
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha_bar = torch.cumprod(1. - self.beta, dim=0)

    def get_loss(self, x_0, img_cond, phys_cond):
        B = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=self.device).long()
        noise = torch.randn_like(x_0).to(self.device)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1. - alpha_bar_t) * noise

        predicted_noise = self.denoise_net(x_t, t.float().unsqueeze(1) / self.n_steps, img_cond, phys_cond)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, img_cond, phys_cond):
        self.eval()
        B, C, H, W = img_cond.shape

        x_t = torch.randn((B, 2, H, W), device=self.device)

        for t_step in reversed(range(self.n_steps)):
            t = torch.full((B,), t_step, device=self.device, dtype=torch.long)
            p_noise = self.denoise_net(x_t, t.float().unsqueeze(1) / self.n_steps, img_cond, phys_cond)

            alpha_t = (1. - self.beta[t]).view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
            beta_t = self.beta[t].view(-1, 1, 1, 1)

            z = torch.randn_like(x_t) if t_step > 0 else 0
            x_t = (1 / torch.sqrt(alpha_t)) * (
                    x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * p_noise) + torch.sqrt(beta_t) * z

        self.train()
        return x_t