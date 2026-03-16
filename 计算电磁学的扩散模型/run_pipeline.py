import os
import sys
import time
import subprocess

# 核心控制台：在这里统一开启/关闭各个模块————把要跑的步骤改成 False，想跑的改成 True

class PipelineConfig:
    #  执行开关 (True 表示执行，False 表示跳过)s
    # Step 1: 物理引擎单次测试 (验证 FDTD 求解器与 CUDA 加速合理性)
    RUN_PHYSICS_TEST = True
    # Step 2: 批量生成训练数据集
    RUN_DATA_GEN =False
    # Step 3: 训练条件扩散模型
    RUN_TRAINING = False
    # Step 4: 逆向设计推理与结果可视化
    RUN_INFERENCE = True

# 自动化调度脚本

def run_command(command, step_name):
    print(f"\n{'=' * 60}")
    print(f" [执行阶段] : {step_name}")
    print(f" [内部命令] : {command}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # 使用 subprocess 启动独立进程，打印终端信息
    process = subprocess.Popen(command, shell=True)
    process.wait()  # 阻塞等待该任务执行完毕

    end_time = time.time()
    cost_time = end_time - start_time

    if process.returncode != 0:
        print(f"\n 发生致命错误！【{step_name}】 执行失败。")
        print("已强制终止，请检查上方的报错信息解决问题后重试。")
        sys.exit(1)
    else:
        print(f"\n 【{step_name}】 执行圆满成功！(耗时: {cost_time:.2f} 秒)")
        time.sleep(1)  # 显存释放的缓冲时间


def main():
    print(" 启动 ")
    print("环境检查：请确保当前已处于 .venv 虚拟环境中。\n")

    total_start = time.time()

    # Step 1: 物理引擎单次测试
    if PipelineConfig.RUN_PHYSICS_TEST:
        run_command("python -m fdtd_em.examples.run_field_plane",
                    "Step 1: 物理引擎单次测试 (验证 FDTD 与 CUDA 加速)")
    else:
        print("[跳过] Step 1: 物理引擎单次测试")

    # Step 2: 生成训练数据集
    if PipelineConfig.RUN_DATA_GEN:
        run_command("python -m fdtd_em.data.generator",
                    "Step 2: 批量生成训练数据集")
    else:
        print("[跳过] Step 2: 批量生成训练数据集")

    # Step 3: 训练条件扩散模型
    if PipelineConfig.RUN_TRAINING:
        run_command("python -m fdtd_em.train.train",
                    "Step 3: 训练条件去噪扩散模型 (Conditional DDPM)")
    else:
        print("[跳过] Step 3: 训练条件扩散模型")

    # Step 4: 逆向推理与可视化出图
    if PipelineConfig.RUN_INFERENCE:
        run_command("python -m fdtd_em.examples.forward_surrogate",
                    "Step 4: 逆向设计推理出图与可视化对比")
    else:
        print("[跳过] Step 4: 逆向设计推理与可视化")

    total_time = time.time() - total_start
    print(f"\n执行完毕！所有任务均已成功。(总耗时: {total_time:.2f} 秒)")


if __name__ == "__main__":
    main()