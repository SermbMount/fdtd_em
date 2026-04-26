import matplotlib.pyplot as plt
import numpy as np
import torch
import fdtd_em.config.generation_config as cfg

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def preview_simulation_setup():
    """生成仿真环境的双视图预演 (俯视 + 侧视)"""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    pml = cfg.pml_thickness

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # --- 1. 获取 3D 掩码数据 ---
    device = torch.device('cpu')
    struct_mask = cfg.build_custom_structure().mask((Nx, Ny, Nz), device).numpy()
    det_mask = cfg.detector_region.mask((Nx, Ny, Nz), device).numpy()

    # --- 2. 左子图：俯视图 (X-Y Plane at Mid-Z) ---
    ax1.set_title("俯视图 (X-Y Plane, 传播截面)")
    ax1.add_patch(plt.Rectangle((0, 0), Nx, Ny, color='white', ec='black'))
    # 画 PML 阴影
    ax1.axvspan(0, pml, color='gray', alpha=0.2, label='PML')
    ax1.axvspan(Nx - pml, Nx, color='gray', alpha=0.2)

    # 画结构截面 (Z轴中间)
    ax1.contourf(struct_mask[:, :, Nz // 2].T, levels=[0.5, 1.5], colors=['darkgreen'], alpha=0.7)

    # 标注光源位置文字
    ax1.text(Nx // 2, Ny - 5, f"光源位于 Z 轴深处", color='red', ha='center', fontweight='bold')
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")

    # --- 3. 右子图：侧视图 (X-Z Plane at Mid-Y)
    ax2.set_title("侧视图 (X-Z Plane, 传播方向剖面)")
    ax2.add_patch(plt.Rectangle((0, 0), Nx, Nz, color='white', ec='black'))

    # 画 PML (左右是 X 轴的 PML)
    ax2.axvspan(0, pml, color='gray', alpha=0.2)
    ax2.axvspan(Nx - pml, Nx, color='gray', alpha=0.2)

    # 画结构侧面投影
    ax2.contourf(struct_mask[:, Ny // 2, :].T, levels=[0.5, 1.5], colors=['darkgreen'], alpha=0.7)

    # 画探测器，找到探测器所在的 Z 索引

    z_indices = np.where(np.any(det_mask, axis=(0, 1)))[0]
    for zi in z_indices:
        ax2.axhline(y=zi, color='blue', linewidth=2, linestyle='--', label='Detector' if zi == z_indices[0] else "")
        ax2.text(Nx - pml - 20, zi + 2, f"Z={zi}", color='blue', fontsize=9)

    # 画光源 (假设在 Z ~ 17)
    # 这里本来打算在 config 里定义一个 src_z_pos 方便引用，目前手动标一下
    src_z = 17  # 对应你图中的 Z ≈ 17
    ax2.axhline(y=src_z, color='red', linewidth=3, label='Source (激励源)')
    ax2.text(pml + 5, src_z - 5, "电磁波向 +Z 传播 ->", color='red', fontsize=10, fontweight='bold')

    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Z Axis (Propagation)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("simulation_preview_3d.png", dpi=150)
    print(" 双视图预演图已生成：simulation_preview_3d.png")
    plt.show()


if __name__ == "__main__":
    preview_simulation_setup()import matplotlib.pyplot as plt
import numpy as np
import torch
import fdtd_em.config.generation_config as cfg

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def preview_simulation_setup():
    """生成仿真环境的双视图预演 (俯视 + 侧视)"""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    pml = cfg.pml_thickness

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # --- 1. 获取 3D 掩码数据 ---
    device = torch.device('cpu')
    struct_mask = cfg.build_custom_structure().mask((Nx, Ny, Nz), device).numpy()
    det_mask = cfg.detector_region.mask((Nx, Ny, Nz), device).numpy()

    # --- 2. 左子图：俯视图 (X-Y Plane at Mid-Z) ---
    ax1.set_title("俯视图 (X-Y Plane, 传播截面)")
    ax1.add_patch(plt.Rectangle((0, 0), Nx, Ny, color='white', ec='black'))
    # 画 PML 阴影
    ax1.axvspan(0, pml, color='gray', alpha=0.2, label='PML')
    ax1.axvspan(Nx - pml, Nx, color='gray', alpha=0.2)

    # 画结构截面 (Z轴中间)
    ax1.contourf(struct_mask[:, :, Nz // 2].T, levels=[0.5, 1.5], colors=['darkgreen'], alpha=0.7)

    # 标注光源位置文字
    ax1.text(Nx // 2, Ny - 5, f"光源位于 Z 轴深处", color='red', ha='center', fontweight='bold')
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")

    # --- 3. 右子图：侧视图 (X-Z Plane at Mid-Y)
    ax2.set_title("侧视图 (X-Z Plane, 传播方向剖面)")
    ax2.add_patch(plt.Rectangle((0, 0), Nx, Nz, color='white', ec='black'))

    # 画 PML (左右是 X 轴的 PML)
    ax2.axvspan(0, pml, color='gray', alpha=0.2)
    ax2.axvspan(Nx - pml, Nx, color='gray', alpha=0.2)

    # 画结构侧面投影
    ax2.contourf(struct_mask[:, Ny // 2, :].T, levels=[0.5, 1.5], colors=['darkgreen'], alpha=0.7)

    # 画探测器，找到探测器所在的 Z 索引

    z_indices = np.where(np.any(det_mask, axis=(0, 1)))[0]
    for zi in z_indices:
        ax2.axhline(y=zi, color='blue', linewidth=2, linestyle='--', label='Detector' if zi == z_indices[0] else "")
        ax2.text(Nx - pml - 20, zi + 2, f"Z={zi}", color='blue', fontsize=9)

    # 画光源 (假设在 Z ~ 17)
    # 这里本来打算在 config 里定义一个 src_z_pos 方便引用，目前手动标一下
    src_z = 17  # 对应你图中的 Z ≈ 17
    ax2.axhline(y=src_z, color='red', linewidth=3, label='Source (激励源)')
    ax2.text(pml + 5, src_z - 5, "电磁波向 +Z 传播 ->", color='red', fontsize=10, fontweight='bold')

    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Z Axis (Propagation)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("simulation_preview_3d.png", dpi=150)
    print(" 双视图预演图已生成：simulation_preview_3d.png")
    plt.show()


if __name__ == "__main__":
    preview_simulation_setup()
