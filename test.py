# -*- coding: utf-8 -*-
import os
import subprocess

# 设置环境编码，防止在终端显示中文日志时乱码
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# 【关键路径配置】
# 源域：我们使用 TON_IoT 和 CICIoMT 作为联合预训练源
source_pth = "/home/njust/data/yqj/SimAD-main later/dataset/TON_IoT_3D/,/home/njust/data/yqj/SimAD-main later/dataset/CICIoMT2024_3D_128/"
# 目标域：刚刚生成的包含单包 UDP 扫描脉冲的 MQTT 数据集
target_pth = "/home/njust/data/yqj/SimAD-main later/dataset/MQTT_3D_128/"

# 遵循 Few-shot 实验标准，测试 1%, 3%, 6% 的标注样本量
fractions = [0.01, 0.03, 0.06]

print("=======================================================")
print("  ER-ADAPT MQTT-IDS [UDP-Scan Optimization] Auto-Runner")
print("  Feature: Zero-Padding Pulse Detection Enabled")
print("=======================================================")

for fraction in fractions:
    print("\n" + "="*55)
    print(f"  [EXEC] Running Test | Shot Fraction: {fraction*100}%")
    print("="*55)
    
    # 构建执行命令
    # 使用 -u 参数保证日志实时输出到 nohup.log 中
    cmd = [
        "python", "-u", "main.py",
        "--data_name", "MQTT_IoT_Experiment",  # 实验名称
        "--data_pth", target_pth,              # 目标域路径
        "--source_pth", source_pth,            # 源域路径
        "--fraction", str(fraction),           # 少量样本比例
        "--device", "cuda:2"                   # 使用指定的 GPU
    ]
    
    # 执行命令并实时监控
    try:
        # 使用 subprocess.run 并在失败时抛出异常
        subprocess.run(cmd, check=True)
        print(f"\n  [SUCCESS] Evaluation for Fraction {fraction} completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n  [CRITICAL ERROR] Fraction {fraction} aborted!")
        print(f"  Return Code: {e.returncode}")
        # 如果某一组比例报错，停止后续实验，方便检查逻辑错误
        break

print("\n=======================================================")
print(">>> [FINISH] All MQTT Scenarios Evaluation Finished! <<<")
print("=======================================================")