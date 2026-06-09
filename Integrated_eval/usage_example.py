#!/usr/bin/env python3
"""
Integrated Evaluation Usage Examples
整合评估脚本的使用示例
"""

import subprocess
import os

def run_evaluation_command(command):
    """运行评估命令"""
    print(f"执行命令: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ 命令执行成功")
        print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
        print("错误输出:", e.stderr)

if __name__ == "__main__":
    print("Integrated Evaluation 使用示例")
    print("=" * 50)

    # 示例1: 评估单个数据集的单个方法
    print("\n1. 评估单个数据集的单个方法:")
    cmd1 = [
        "python", "integrated_evaluation.py",
        "--dataset", "issaacseq",
        "--method", "midas"
    ]
    print(f"python integrated_evaluation.py --dataset issaacseq --method midas")

    # 示例2: 评估单个数据集的多个方法
    print("\n2. 评估单个数据集的多个方法:")
    cmd2 = [
        "python", "integrated_evaluation.py",
        "--dataset", "issaacseq",
        "--method", "midas",
        "--method", "totalvi"
    ]
    print(f"python integrated_evaluation.py --dataset issaacseq --method midas --method totalvi")

    # 示例3: 评估多个数据集的单个方法
    print("\n3. 评估多个数据集的单个方法:")
    cmd3 = [
        "python", "integrated_evaluation.py",
        "--dataset", "issaacseq",
        "--dataset", "SNAREseq_mouse",
        "--method", "midas"
    ]
    print(f"python integrated_evaluation.py --dataset issaacseq --dataset SNAREseq_mouse --method midas")

    # 示例4: 指定输出文件
    print("\n4. 指定输出文件:")
    cmd4 = [
        "python", "integrated_evaluation.py",
        "--dataset", "issaacseq",
        "--method", "midas",
        "--output", "my_custom_results.csv"
    ]
    print(f"python integrated_evaluation.py --dataset issaacseq --method midas --output my_custom_results.csv")

    print("\n" + "=" * 50)
    print("💡 提示:")
    print("- 每次运行结果都会追加到输出文件，不会覆盖之前的结果")
    print("- 输出文件包含5个评估方面的综合指标")
    print("- 可以多次运行不同method，结果会累积保存")

    # 检查脚本是否存在
    if os.path.exists("integrated_evaluation.py"):
        print("\n✅ 整合评估脚本已创建: integrated_evaluation.py")
    else:
        print("\n❌ 整合评估脚本不存在，请先创建")