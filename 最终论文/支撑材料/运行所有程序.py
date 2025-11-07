#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有数学建模问题的程序
自动执行5个问题的代码，并将结果统一保存到数据文件文件夹

使用方法：
    python 运行所有程序.py
或
    python3 运行所有程序.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, cwd=None):
    """运行shell命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def main():
    print("=" * 80)
    print("数学建模竞赛 - 全部程序自动运行脚本")
    print("=" * 80)
    print()

    # 获取当前脚本所在目录，并切换到代码目录的父级
    script_dir = Path(__file__).parent
    code_dir = script_dir / "代码"

    # 切换到代码文件的父目录（以便相对路径可以找到各个问题文件夹）
    os.chdir(script_dir.parent.parent)

    print(f"工作目录: {os.getcwd()}")
    print()

    # 定义5个问题的代码文件
    problems = [
        ("问题一：USDT与USDC对比分析", "第一个问题/第一个问题_实现代码_20251105.py"),
        ("问题二：储备资产配置优化", "第二个问题/第二个问题_实现代码_20251105.py"),
        ("问题三：需求预测与市场份额", "第三个问题/第三个问题_实现代码_20251105.py"),
        ("问题四：货币主权影响评估", "第四个问题/第四个问题_实现代码_20251105.py"),
        ("问题五：政策简报生成", "第五个问题/第五个问题_实现代码_20251105.py")
    ]

    # 数据文件输出目录
    output_dir = Path("最终论文/支撑材料/数据文件")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行每个问题
    success_count = 0
    for i, (title, code_file) in enumerate(problems, 1):
        print(f"【{i}/5】运行{title}...")
        print(f"  代码文件: {code_file}")

        # 运行Python代码
        cmd = f"python {code_file}"
        success, output = run_command(cmd)

        if success:
            print(f"  ✓ 运行成功")
            success_count += 1

            # 移动生成的文件到统一输出目录
            problem_dir = Path(code_file).parent

            # 移动CSV文件
            for csv_file in problem_dir.glob("*.csv"):
                target = output_dir / csv_file.name
                shutil.copy2(csv_file, target)
                print(f"    → {csv_file.name}")

            # 移动TXT文件
            for txt_file in problem_dir.glob("*.txt"):
                target = output_dir / txt_file.name
                shutil.copy2(txt_file, target)
                print(f"    → {txt_file.name}")

            # 移动PNG图片文件
            for png_file in problem_dir.glob("*.png"):
                target = output_dir / png_file.name
                shutil.copy2(png_file, target)
                print(f"    → {png_file.name}")

        else:
            print(f"  ✗ 运行失败")
            print(f"  错误信息: {output}")

        print()

    print("=" * 80)
    print("运行完成！")
    print("=" * 80)
    print(f"成功运行: {success_count}/{len(problems)} 个问题")
    print(f"输出目录: {output_dir.absolute()}")
    print()

    # 列出生成的文件
    data_files = list(output_dir.glob("*"))
    if data_files:
        print("生成的数据文件：")
        for f in sorted(data_files):
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name:40s} ({size:.1f} KB)")
    else:
        print("警告：未找到生成的数据文件")

    print()
    print("提示：所有计算结果已保存到 '最终论文/支撑材料/数据文件/' 目录")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
