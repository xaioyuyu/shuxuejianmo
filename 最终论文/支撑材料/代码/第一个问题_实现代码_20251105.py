#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USDT与USDC综合对比分析模型
基于层次分析法(AHP)和模糊综合评价的稳定币竞争力评估系统

作者：数学建模团队
日期：2025-11-05
版本：1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体显示
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证结果可复现
np.random.seed(42)


class AHPAnalyzer:
    """
    层次分析法(AHP)分析器
    用于计算指标权重并进行一致性检验
    """

    # 平均随机一致性指标RI值表
    RI_TABLE = {
        1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }

    def __init__(self, matrix, criteria_names=None):
        """
        初始化AHP分析器

        参数:
            matrix: numpy数组，判断矩阵
            criteria_names: 列表，指标名称
        """
        self.matrix = np.array(matrix, dtype=float)
        self.n = len(matrix)
        self.criteria_names = criteria_names or [
            f"C{i+1}" for i in range(self.n)]
        self.weights = None
        self.lambda_max = None
        self.CI = None
        self.CR = None

    def calculate_weights(self):
        """
        使用特征值法计算权重向量

        返回:
            weights: numpy数组，归一化的权重向量
        """
        # 计算判断矩阵的最大特征值和对应的特征向量
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)

        # 获取最大特征值的索引
        max_index = np.argmax(eigenvalues.real)
        self.lambda_max = eigenvalues[max_index].real

        # 获取对应的特征向量并归一化
        max_eigenvector = eigenvectors[:, max_index].real
        self.weights = max_eigenvector / max_eigenvector.sum()

        return self.weights

    def consistency_check(self):
        """
        进行一致性检验

        返回:
            is_consistent: 布尔值，是否通过一致性检验
            CI: 一致性指标
            CR: 一致性比率
        """
        if self.weights is None:
            self.calculate_weights()

        # 计算一致性指标CI
        self.CI = (self.lambda_max - self.n) / (self.n - 1)

        # 获取平均随机一致性指标RI
        RI = self.RI_TABLE.get(self.n, 1.49)

        # 计算一致性比率CR
        self.CR = self.CI / RI if RI != 0 else 0

        # CR < 0.1 表示通过一致性检验
        is_consistent = self.CR < 0.1

        return is_consistent, self.CI, self.CR

    def print_results(self):
        """打印AHP分析结果"""
        print("=" * 60)
        print("层次分析法(AHP)计算结果")
        print("=" * 60)
        print(f"判断矩阵维度: {self.n} × {self.n}")
        print(f"最大特征值 λmax: {self.lambda_max:.4f}")
        print(f"一致性指标 CI: {self.CI:.4f}")
        print(f"一致性比率 CR: {self.CR:.4f}")
        print(f"一致性检验: {'通过' if self.CR < 0.1 else '不通过'}")
        print("\n指标权重:")
        for i, (name, weight) in enumerate(zip(self.criteria_names, self.weights)):
            print(f"  {name}: {weight:.4f} ({weight*100:.2f}%)")
        print("=" * 60 + "\n")


class StablecoinEvaluator:
    """
    稳定币综合评价系统
    整合多维度指标进行综合评价和对比分析
    """

    def __init__(self):
        """初始化评价系统"""
        # 定义一级指标
        self.criteria = [
            "监管合规性", "透明度", "技术能力",
            "市场表现", "应用场景", "风险水平"
        ]

        # 定义二级指标
        self.sub_criteria = {
            "监管合规性": ["牌照数量", "合规评分", "法律纠纷"],
            "透明度": ["信息披露频率", "审计报告", "储备透明度"],
            "技术能力": ["区块链网络数", "交易速度", "安全性"],
            "市场表现": ["市场份额", "交易量", "流动性"],
            "应用场景": ["交易所支持", "DeFi集成", "机构合作"],
            "风险水平": ["储备风险", "脱锚风险", "监管风险"]
        }

        # 一级指标权重（通过AHP计算得到）
        self.criteria_weights = None

        # 二级指标权重（简化处理，使用平均权重）
        self.sub_criteria_weights = {}

        # 评价数据
        self.data = {}

    def set_criteria_weights(self, weights):
        """
        设置一级指标权重

        参数:
            weights: numpy数组或列表，权重向量
        """
        self.criteria_weights = np.array(weights)

    def normalize_data(self, data, method='minmax'):
        """
        数据标准化

        参数:
            data: numpy数组，原始数据
            method: 字符串，标准化方法 ('minmax' 或 'zscore')

        返回:
            normalized: numpy数组，标准化后的数据(0-10分)
        """
        if method == 'minmax':
            # 极差标准化：归一化到0-10分
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            # 避免除零错误
            range_val = data_max - data_min
            range_val[range_val == 0] = 1
            normalized = (data - data_min) / range_val * 10
        elif method == 'zscore':
            # Z-score标准化后映射到0-10分
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1
            normalized = (data - mean) / std
            # 映射到0-10分
            normalized = (normalized + 3) / 6 * 10
            normalized = np.clip(normalized, 0, 10)
        else:
            raise ValueError("不支持的标准化方法")

        return normalized

    def load_data(self):
        """
        加载USDT和USDC的评价数据
        这里使用模拟数据，实际应用中应从数据库或API获取
        """
        # USDT数据（原始值）
        usdt_data = {
            "监管合规性": [2, 4, 3],  # [牌照数量, 合规评分, 法律纠纷次数(负向)]
            "透明度": [4, 6, 6],       # [披露频率, 审计报告, 储备透明度]
            "技术能力": [15, 8, 9],    # [网络数, 速度评分, 安全性评分]
            "市场表现": [62.5, 580, 2],  # [市场份额%, 交易量(亿$), 流动性bp(负向)]
            "应用场景": [95, 150, 45],   # [交易所%, DeFi数, 机构数]
            "风险水平": [5.8, 4.5, 7.2]  # [储备风险, 脱锚风险, 监管风险(都是负向)]
        }

        # USDC数据（原始值）
        usdc_data = {
            "监管合规性": [5, 9, 0],   # [牌照数量, 合规评分, 法律纠纷次数]
            "透明度": [52, 10, 10],    # [披露频率(周), 审计报告, 储备透明度]
            "技术能力": [12, 7, 8],    # [网络数, 速度评分, 安全性评分]
            "市场表现": [24.3, 85, 5],  # [市场份额%, 交易量(亿$), 流动性bp]
            "应用场景": [90, 180, 68],  # [交易所%, DeFi数, 机构数]
            "风险水平": [3.2, 5.0, 3.5]  # [储备风险, 脱锚风险, 监管风险]
        }

        self.data = {
            "USDT": usdt_data,
            "USDC": usdc_data
        }

    def calculate_scores(self):
        """
        计算两种稳定币的综合得分

        返回:
            results: 字典，包含详细得分信息
        """
        if self.criteria_weights is None:
            raise ValueError("请先设置一级指标权重")

        if not self.data:
            self.load_data()

        results = {}

        # 对每种稳定币进行评分
        for coin_name in ["USDT", "USDC"]:
            coin_data = self.data[coin_name]
            criteria_scores = []

            # 计算每个一级指标的得分
            for i, criterion in enumerate(self.criteria):
                # 获取该指标下的二级指标原始值
                sub_values = np.array(coin_data[criterion])

                # 对负向指标进行处理（如法律纠纷次数、流动性bp、风险值等）
                if criterion == "监管合规性":
                    # 第3个指标是负向的（法律纠纷）
                    sub_values[2] = 10 - sub_values[2]  # 转换为正向
                elif criterion == "市场表现":
                    # 第3个指标是负向的（买卖价差）
                    sub_values[2] = 10 - sub_values[2]
                elif criterion == "风险水平":
                    # 所有风险指标都是负向的，需要取反
                    sub_values = 10 - sub_values

                # 标准化到0-10分（这里简化处理，实际应与其他币种对比）
                # 为了演示，我们对两种币的同一指标进行归一化
                sub_values = np.clip(sub_values, 0, 10)

                # 计算该一级指标得分（二级指标简单平均）
                criterion_score = np.mean(sub_values)
                criteria_scores.append(criterion_score)

            criteria_scores = np.array(criteria_scores)

            # 计算综合得分
            total_score = np.dot(self.criteria_weights, criteria_scores)

            results[coin_name] = {
                "criteria_scores": criteria_scores,
                "total_score": total_score
            }

        return results

    def calculate_detailed_scores(self):
        """
        更精细的得分计算，考虑相对比较

        返回:
            results: 字典，包含详细得分信息
        """
        if not self.data:
            self.load_data()

        # 将两种币的数据合并，以便进行相对标准化
        all_data = {}
        for criterion in self.criteria:
            usdt_vals = np.array(self.data["USDT"][criterion])
            usdc_vals = np.array(self.data["USDC"][criterion])
            all_data[criterion] = np.vstack([usdt_vals, usdc_vals])

        results = {}

        for idx, coin_name in enumerate(["USDT", "USDC"]):
            criteria_scores = []

            for i, criterion in enumerate(self.criteria):
                # 获取标准化后的数据
                combined_data = all_data[criterion]

                # 处理负向指标
                if criterion == "监管合规性":
                    combined_data[:, 2] = np.max(
                        combined_data[:, 2]) - combined_data[:, 2]
                elif criterion == "市场表现":
                    combined_data[:, 2] = np.max(
                        combined_data[:, 2]) - combined_data[:, 2]
                elif criterion == "风险水平":
                    combined_data = np.max(
                        combined_data, axis=0) - combined_data

                # 标准化
                normalized = self.normalize_data(
                    combined_data, method='minmax')

                # 获取当前币种的得分
                coin_normalized = normalized[idx]

                # 计算该一级指标得分
                criterion_score = np.mean(coin_normalized)
                criteria_scores.append(criterion_score)

            criteria_scores = np.array(criteria_scores)

            # 计算综合得分
            total_score = np.dot(self.criteria_weights, criteria_scores)

            results[coin_name] = {
                "criteria_scores": criteria_scores,
                "total_score": total_score
            }

        return results


class RiskAnalyzer:
    """
    风险分析器
    评估稳定币的各类风险并计算总体风险值
    """

    def __init__(self):
        """初始化风险分析器"""
        self.risk_categories = [
            "监管风险", "流动性风险", "储备资产风险",
            "脱锚风险", "技术风险"
        ]

    def calculate_risk_matrix(self, probabilities, impacts, weights=None):
        """
        计算风险矩阵

        参数:
            probabilities: numpy数组，风险发生概率(0-1)
            impacts: numpy数组，风险影响程度(0-10)
            weights: numpy数组，风险类别权重(可选)

        返回:
            risk_values: numpy数组，各类风险值
            total_risk: 浮点数，总体风险值
        """
        probabilities = np.array(probabilities)
        impacts = np.array(impacts)

        # 计算风险值 = 概率 × 影响
        risk_values = probabilities * impacts

        # 如果没有提供权重，使用平均权重
        if weights is None:
            weights = np.ones(len(risk_values)) / len(risk_values)
        else:
            weights = np.array(weights)

        # 计算加权总风险
        total_risk = np.dot(weights, risk_values)

        return risk_values, total_risk

    def evaluate_coins_risk(self):
        """
        评估USDT和USDC的风险

        返回:
            results: 字典，包含两种币的风险评估结果
        """
        # USDT风险数据
        usdt_prob = np.array([0.6, 0.2, 0.5, 0.3, 0.25])  # 发生概率
        usdt_impact = np.array([8, 5, 7, 6, 5])           # 影响程度

        # USDC风险数据
        usdc_prob = np.array([0.3, 0.3, 0.25, 0.4, 0.2])
        usdc_impact = np.array([5, 6, 4, 7, 4])

        # 风险类别权重
        weights = np.array([0.25, 0.20, 0.25, 0.20, 0.10])

        # 计算风险值
        usdt_risks, usdt_total = self.calculate_risk_matrix(
            usdt_prob, usdt_impact, weights)
        usdc_risks, usdc_total = self.calculate_risk_matrix(
            usdc_prob, usdc_impact, weights)

        results = {
            "USDT": {
                "probabilities": usdt_prob,
                "impacts": usdt_impact,
                "risk_values": usdt_risks,
                "total_risk": usdt_total
            },
            "USDC": {
                "probabilities": usdc_prob,
                "impacts": usdc_impact,
                "risk_values": usdc_risks,
                "total_risk": usdc_total
            }
        }

        return results


class Visualizer:
    """
    可视化工具类
    生成各种对比分析图表
    """

    @staticmethod
    def plot_radar_chart(categories, values_dict, title="雷达图对比"):
        """
        绘制雷达图

        参数:
            categories: 列表，指标名称
            values_dict: 字典，{名称: 数值列表}
            title: 字符串，图表标题
        """
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories),
                             endpoint=False).tolist()

        # 闭合图形
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(10, 8), subplot_kw=dict(projection='polar'))

        # 为每个数据系列绘制雷达图
        for name, values in values_dict.items():
            values = values.tolist() + values.tolist()[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.15)

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.grid(True)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_bar_comparison(categories, values_dict, title="柱状图对比"):
        """
        绘制分组柱状图

        参数:
            categories: 列表，类别名称
            values_dict: 字典，{系列名: 数值列表}
            title: 字符串，图表标题
        """
        x = np.arange(len(categories))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # 绘制每个系列
        for i, (name, values) in enumerate(values_dict.items()):
            offset = width * (i - len(values_dict)/2 + 0.5)
            ax.bar(x + offset, values, width, label=name,
                   color=colors[i % len(colors)])

        ax.set_xlabel('评价指标', fontsize=12, fontweight='bold')
        ax.set_ylabel('得分', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha='right')
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_risk_matrix(risk_data, title="风险矩阵"):
        """
        绘制风险矩阵散点图

        参数:
            risk_data: 字典，{名称: {"probabilities": [...], "impacts": [...]}}
            title: 字符串，图表标题
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {'USDT': '#3498db', 'USDC': '#e74c3c'}
        markers = {'USDT': 'o', 'USDC': 's'}

        for name, data in risk_data.items():
            probs = data["probabilities"]
            impacts = data["impacts"]
            ax.scatter(probs, impacts, s=200, alpha=0.6,
                       c=colors[name], marker=markers[name],
                       label=name, edgecolors='black', linewidth=2)

        # 添加风险区域划分
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)

        # 标注风险等级区域
        ax.text(0.75, 8, '高风险区', fontsize=12, color='red', fontweight='bold')
        ax.text(0.1, 8, '中风险区', fontsize=12, color='orange', fontweight='bold')
        ax.text(0.1, 2, '低风险区', fontsize=12, color='green', fontweight='bold')

        ax.set_xlabel('风险发生概率', fontsize=12, fontweight='bold')
        ax.set_ylabel('风险影响程度', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def main():
    """
    主程序：执行完整的USDT与USDC对比分析
    """
    print("\n" + "="*80)
    print("USDT与USDC综合对比分析系统".center(80))
    print("="*80 + "\n")

    # ========== 第一步：使用AHP确定一级指标权重 ==========
    print("【步骤1】使用层次分析法(AHP)计算一级指标权重...\n")

    # 构造判断矩阵（基于专家评分）
    # 指标顺序：监管合规性、透明度、技术能力、市场表现、应用场景、风险水平
    judgment_matrix = np.array([
        [1,   1/2, 1/3, 2,   1/2, 1],
        [2,   1,   1/2, 3,   1,   2],
        [3,   2,   1,   4,   2,   3],
        [1/2, 1/3, 1/4, 1,   1/3, 1/2],
        [2,   1,   1/2, 3,   1,   2],
        [1,   1/2, 1/3, 2,   1/2, 1]
    ])

    criteria_names = ["监管合规性", "透明度", "技术能力", "市场表现", "应用场景", "风险水平"]

    # 创建AHP分析器
    ahp = AHPAnalyzer(judgment_matrix, criteria_names)

    # 计算权重
    weights = ahp.calculate_weights()

    # 一致性检验
    is_consistent, CI, CR = ahp.consistency_check()

    # 打印结果
    ahp.print_results()

    if not is_consistent:
        print("⚠️  警告：判断矩阵未通过一致性检验，请重新调整！\n")
        return

    # ========== 第二步：综合评价计算 ==========
    print("【步骤2】计算USDT和USDC的综合得分...\n")

    # 创建评价器
    evaluator = StablecoinEvaluator()
    evaluator.set_criteria_weights(weights)
    evaluator.load_data()

    # 计算详细得分
    results = evaluator.calculate_detailed_scores()

    # 打印评价结果
    print("=" * 60)
    print("综合评价结果")
    print("=" * 60)

    for coin_name in ["USDT", "USDC"]:
        print(f"\n{coin_name} 评分详情:")
        print("-" * 60)
        criteria_scores = results[coin_name]["criteria_scores"]
        for i, (criterion, score) in enumerate(zip(criteria_names, criteria_scores)):
            print(f"  {criterion}: {score:.2f}/10 (权重: {weights[i]:.3f})")
        print(f"\n  综合得分: {results[coin_name]['total_score']:.2f}/10")
        print(f"  百分制得分: {results[coin_name]['total_score']*10:.2f}/100")
        print("-" * 60)

    # 对比分析
    score_diff = results["USDC"]["total_score"] - \
        results["USDT"]["total_score"]
    print(f"\n综合对比:")
    print(f"  USDC相对USDT的得分差距: {score_diff:+.2f}分")
    if abs(score_diff) < 0.5:
        print(f"  结论: 两者竞争力相当，差距不明显")
    elif score_diff > 0:
        print(f"  结论: USDC综合竞争力略优于USDT")
    else:
        print(f"  结论: USDT综合竞争力略优于USDC")
    print("=" * 60 + "\n")

    # ========== 第三步：风险分析 ==========
    print("【步骤3】进行风险评估分析...\n")

    # 创建风险分析器
    risk_analyzer = RiskAnalyzer()
    risk_results = risk_analyzer.evaluate_coins_risk()

    # 打印风险分析结果
    print("=" * 60)
    print("风险评估结果")
    print("=" * 60)

    for coin_name in ["USDT", "USDC"]:
        print(f"\n{coin_name} 风险分析:")
        print("-" * 60)
        risk_data = risk_results[coin_name]
        for i, category in enumerate(risk_analyzer.risk_categories):
            prob = risk_data["probabilities"][i]
            impact = risk_data["impacts"][i]
            risk_val = risk_data["risk_values"][i]
            print(f"  {category}:")
            print(
                f"    发生概率: {prob:.2f} | 影响程度: {impact:.1f} | 风险值: {risk_val:.2f}")
        print(f"\n  总体风险评分: {risk_data['total_risk']:.2f}/10")

        # 风险等级判断
        total_risk = risk_data['total_risk']
        if total_risk < 3:
            risk_level = "低风险 ✓"
        elif total_risk < 6:
            risk_level = "中等风险"
        elif total_risk < 8:
            risk_level = "高风险 ⚠"
        else:
            risk_level = "极高风险 ⚠️"
        print(f"  风险等级: {risk_level}")
        print("-" * 60)

    print("=" * 60 + "\n")

    # ========== 第四步：数据可视化 ==========
    print("【步骤4】生成可视化图表...\n")

    visualizer = Visualizer()

    # 1. 雷达图
    radar_data = {
        "USDT": results["USDT"]["criteria_scores"],
        "USDC": results["USDC"]["criteria_scores"]
    }
    fig1 = visualizer.plot_radar_chart(criteria_names, radar_data,
                                       "USDT vs USDC 多维度雷达图对比")
    fig1.savefig("第一个问题/USDT_USDC_雷达图.png", dpi=300, bbox_inches='tight')
    print("  ✓ 雷达图已保存: USDT_USDC_雷达图.png")

    # 2. 柱状图
    fig2 = visualizer.plot_bar_comparison(criteria_names, radar_data,
                                          "USDT vs USDC 各维度得分对比")
    fig2.savefig("第一个问题/USDT_USDC_柱状图.png", dpi=300, bbox_inches='tight')
    print("  ✓ 柱状图已保存: USDT_USDC_柱状图.png")

    # 3. 风险矩阵
    fig3 = visualizer.plot_risk_matrix(risk_results, "USDT vs USDC 风险矩阵分析")
    fig3.savefig("第一个问题/USDT_USDC_风险矩阵.png", dpi=300, bbox_inches='tight')
    print("  ✓ 风险矩阵图已保存: USDT_USDC_风险矩阵.png")

    # ========== 第五步：生成对比报告 ==========
    print("\n【步骤5】生成详细对比报告...\n")

    # 创建DataFrame用于展示
    comparison_df = pd.DataFrame({
        "评价维度": criteria_names,
        "权重": [f"{w:.3f}" for w in weights],
        "USDT得分": [f"{s:.2f}" for s in results["USDT"]["criteria_scores"]],
        "USDC得分": [f"{s:.2f}" for s in results["USDC"]["criteria_scores"]],
        "差距": [f"{u-t:+.2f}" for t, u in zip(results["USDT"]["criteria_scores"],
                                             results["USDC"]["criteria_scores"])]
    })

    print("详细对比表:")
    print(comparison_df.to_string(index=False))

    # 保存到CSV
    comparison_df.to_csv("第一个问题/USDT_USDC_对比表.csv",
                         index=False, encoding='utf-8-sig')
    print("\n  ✓ 对比表已保存: USDT_USDC_对比表.csv")

    # ========== 总结 ==========
    print("\n" + "="*80)
    print("分析完成！".center(80))
    print("="*80)
    print("\n核心发现:")
    print(f"  1. USDT综合得分: {results['USDT']['total_score']*10:.2f}/100")
    print(f"  2. USDC综合得分: {results['USDC']['total_score']*10:.2f}/100")
    print(f"  3. USDT总体风险: {risk_results['USDT']['total_risk']:.2f}/10")
    print(f"  4. USDC总体风险: {risk_results['USDC']['total_risk']:.2f}/10")
    print("\n建议:")
    if results["USDT"]["total_score"] > results["USDC"]["total_score"]:
        print("  • USDT在市场表现和流动性方面具有明显优势")
        print("  • 但需关注监管合规性和透明度问题")
    else:
        print("  • USDC在合规性和透明度方面表现优异")
        print("  • 未来增长潜力较大，但需提升市场份额")

    print("\n所有结果文件已保存至 '第一个问题' 文件夹")
    print("="*80 + "\n")

    # 显示图表（可选）
    # plt.show()


if __name__ == "__main__":
    """程序入口"""
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
