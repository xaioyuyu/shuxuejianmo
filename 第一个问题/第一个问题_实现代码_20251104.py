#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USDT与USDC综合对比分析程序
基于层次分析法(AHP)和模糊综合评价方法

作者：数学建模团队
日期：2025-11-04
版本：v1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
rcParams['axes.unicode_minus'] = False


class StablecoinEvaluator:
    """
    稳定币综合评价类

    主要功能：
    1. 构建层次分析法判断矩阵
    2. 计算权重并进行一致性检验
    3. 综合评分计算
    4. 风险评估
    5. 发展潜力分析
    6. 可视化结果
    """

    def __init__(self):
        """初始化评价器，设置基本参数"""
        # 一级指标名称
        self.criteria_level1 = [
            '监管合规性', '透明度', '技术能力',
            '市场表现', '应用场景', '风险水平'
        ]

        # 随机一致性指标RI（查表获得）
        self.RI = {
            1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }

        # 稳定币名称
        self.coins = ['USDT', 'USDC']

        # 存储结果
        self.weights_level1 = None
        self.scores = None
        self.risks = None

    def construct_judgment_matrix(self):
        """
        构造一级指标的判断矩阵

        返回：
        -------
        matrix : numpy.ndarray
            判断矩阵，6x6
        """
        # 根据专家意见构造的判断矩阵
        # 行列顺序：监管合规性、透明度、技术能力、市场表现、应用场景、风险水平
        matrix = np.array([
            [1,     1/2,  1/3,  1/4,  1/3,  2],
            [2,     1,    1/2,  1/3,  1/2,  3],
            [3,     2,    1,    1/2,  1,    4],
            [4,     3,    2,    1,    2,    5],
            [3,     2,    1,    1/2,  1,    4],
            [1/2,   1/3,  1/4,  1/5,  1/4,  1]
        ])

        return matrix

    def calculate_weights(self, matrix):
        """
        使用特征值法计算权重

        参数：
        -------
        matrix : numpy.ndarray
            判断矩阵

        返回：
        -------
        weights : numpy.ndarray
            权重向量
        lambda_max : float
            最大特征值
        """
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        # 找到最大特征值的索引
        max_index = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues[max_index].real

        # 对应的特征向量
        weights = eigenvectors[:, max_index].real

        # 归一化
        weights = weights / np.sum(weights)

        return weights, lambda_max

    def consistency_check(self, matrix, lambda_max):
        """
        一致性检验

        参数：
        -------
        matrix : numpy.ndarray
            判断矩阵
        lambda_max : float
            最大特征值

        返回：
        -------
        CR : float
            一致性比率
        CI : float
            一致性指标
        passed : bool
            是否通过检验
        """
        n = matrix.shape[0]

        # 计算一致性指标CI
        CI = (lambda_max - n) / (n - 1)

        # 计算一致性比率CR
        CR = CI / self.RI[n]

        # 判断是否通过（CR < 0.1）
        passed = CR < 0.1

        return CR, CI, passed

    def get_level2_scores(self):
        """
        获取二级指标评分数据

        返回：
        -------
        scores_dict : dict
            包含各一级指标下的二级指标评分
        """
        scores_dict = {
            '监管合规性': {
                'weights': [0.35, 0.40, 0.25],
                'indicators': ['持有牌照数量', '合规性历史记录', '监管透明度'],
                'USDT': [4, 3, 5],
                'USDC': [9, 9, 8]
            },
            '透明度': {
                'weights': [0.40, 0.35, 0.25],
                'indicators': ['信息披露频率', '审计报告质量', '储备资产透明度'],
                'USDT': [5, 6, 6],
                'USDC': [9, 9, 9]
            },
            '技术能力': {
                'weights': [0.30, 0.25, 0.25, 0.20],
                'indicators': ['区块链网络覆盖', '交易速度', '安全性记录', '技术创新性'],
                'USDT': [9, 8, 7, 6],
                'USDC': [7, 8, 8, 8]
            },
            '市场表现': {
                'weights': [0.35, 0.30, 0.20, 0.15],
                'indicators': ['市场份额', '日交易量', '流动性', '用户规模'],
                'USDT': [10, 10, 10, 9],
                'USDC': [4, 5, 6, 5]
            },
            '应用场景': {
                'weights': [0.30, 0.25, 0.25, 0.20],
                'indicators': ['加密交易应用', 'DeFi集成度', '跨境支付', '传统机构合作'],
                'USDT': [10, 6, 8, 4],
                'USDC': [6, 9, 7, 9]
            },
            '风险水平': {
                'weights': [0.35, 0.30, 0.20, 0.15],
                'indicators': ['储备资产风险', '脱锚风险', '流动性风险', '监管风险'],
                'USDT': [6, 7, 9, 5],
                'USDC': [8, 6, 7, 8]
            }
        }

        return scores_dict

    def calculate_comprehensive_scores(self):
        """
        计算综合评分

        返回：
        -------
        results : dict
            包含各维度得分和总分
        """
        # 获取一级指标权重
        matrix = self.construct_judgment_matrix()
        weights_level1, lambda_max = self.calculate_weights(matrix)
        self.weights_level1 = weights_level1

        # 一致性检验
        CR, CI, passed = self.consistency_check(matrix, lambda_max)
        print(f"\n=== 一致性检验 ===")
        print(f"最大特征值 λmax = {lambda_max:.4f}")
        print(f"一致性指标 CI = {CI:.4f}")
        print(f"一致性比率 CR = {CR:.4f}")
        print(f"检验结果：{'通过' if passed else '不通过'} (CR < 0.1)\n")

        # 获取二级指标评分
        scores_dict = self.get_level2_scores()

        # 计算各一级指标得分
        results = {coin: {} for coin in self.coins}

        for i, criterion in enumerate(self.criteria_level1):
            data = scores_dict[criterion]
            weights_level2 = np.array(data['weights'])

            for coin in self.coins:
                # 原始得分
                raw_scores = np.array(data[coin])
                # 标准化（0-10分制转为0-1）
                normalized_scores = raw_scores / 10.0
                # 加权求和
                weighted_score = np.sum(weights_level2 * normalized_scores)
                results[coin][criterion] = weighted_score * 100  # 转为百分制

        # 计算总分
        for coin in self.coins:
            total_score = 0
            for i, criterion in enumerate(self.criteria_level1):
                total_score += weights_level1[i] * results[coin][criterion]
            results[coin]['总分'] = total_score

        self.scores = results
        return results

    def calculate_risk_assessment(self):
        """
        计算风险评估

        返回：
        -------
        risk_results : dict
            风险评估结果
        """
        # 风险类型及权重
        risk_types = ['监管处罚风险', '储备挤兑风险', '技术安全风险',
                      '市场竞争风险', '声誉风险']
        risk_weights = [0.25, 0.30, 0.20, 0.15, 0.10]

        # 概率和影响程度数据
        risk_data = {
            'USDT': {
                'probability': [0.35, 0.15, 0.10, 0.20, 0.25],
                'impact': [8, 9, 8, 6, 7]
            },
            'USDC': {
                'probability': [0.10, 0.20, 0.08, 0.30, 0.15],
                'impact': [6, 8, 7, 7, 6]
            }
        }

        # 计算风险值
        risk_results = {}
        for coin in self.coins:
            prob = np.array(risk_data[coin]['probability'])
            impact = np.array(risk_data[coin]['impact'])
            risk_values = prob * impact

            # 加权总风险
            total_risk = np.sum(np.array(risk_weights) * risk_values)

            risk_results[coin] = {
                'risk_types': risk_types,
                'risk_values': risk_values.tolist(),
                'total_risk': total_risk
            }

        self.risks = risk_results
        return risk_results

    def calculate_development_potential(self):
        """
        计算发展潜力指数

        返回：
        -------
        dpi_results : dict
            发展潜力指数结果
        """
        # 参数设置
        alpha, beta, gamma, delta = 0.3, 0.3, 0.25, 0.15

        # 数据（增长率、创新能力、市场机会）
        data = {
            'USDT': {'growth_rate': 0.15, 'innovation': 6, 'market_opp': 7},
            'USDC': {'growth_rate': 0.25, 'innovation': 8, 'market_opp': 8}
        }

        # 计算DPI
        dpi_results = {}
        for coin in self.coins:
            gr = data[coin]['growth_rate']
            ia = data[coin]['innovation']
            mc = data[coin]['market_opp']
            risk = self.risks[coin]['total_risk']

            dpi = alpha * gr + beta * ia + gamma * mc - delta * risk
            dpi_results[coin] = dpi

        return dpi_results

    def visualize_results(self):
        """
        可视化分析结果
        生成多个图表展示评价结果
        """
        # 创建图表
        fig = plt.figure(figsize=(16, 12))

        # 1. 雷达图：各维度对比
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        self._plot_radar_chart(ax1)

        # 2. 柱状图：一级指标权重
        ax2 = plt.subplot(2, 3, 2)
        self._plot_weights_bar(ax2)

        # 3. 对比柱状图：各维度得分
        ax3 = plt.subplot(2, 3, 3)
        self._plot_scores_comparison(ax3)

        # 4. 风险对比图
        ax4 = plt.subplot(2, 3, 4)
        self._plot_risk_comparison(ax4)

        # 5. 综合得分对比
        ax5 = plt.subplot(2, 3, 5)
        self._plot_total_scores(ax5)

        # 6. 发展潜力对比
        ax6 = plt.subplot(2, 3, 6)
        self._plot_development_potential(ax6)

        plt.tight_layout()
        plt.savefig('第一个问题/USDT_USDC_综合评价结果.png', dpi=300, bbox_inches='tight')
        print("\n图表已保存：第一个问题/USDT_USDC_综合评价结果.png")
        plt.show()

    def _plot_radar_chart(self, ax):
        """绘制雷达图"""
        categories = self.criteria_level1
        N = len(categories)

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        # 数据
        usdt_values = [self.scores['USDT'][cat] for cat in categories]
        usdc_values = [self.scores['USDC'][cat] for cat in categories]
        usdt_values += usdt_values[:1]
        usdc_values += usdc_values[:1]

        # 绘制
        ax.plot(angles, usdt_values, 'o-', linewidth=2,
                label='USDT', color='#FF6B6B')
        ax.fill(angles, usdt_values, alpha=0.25, color='#FF6B6B')
        ax.plot(angles, usdc_values, 'o-', linewidth=2,
                label='USDC', color='#4ECDC4')
        ax.fill(angles, usdc_values, alpha=0.25, color='#4ECDC4')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title('各维度对比雷达图', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

    def _plot_weights_bar(self, ax):
        """绘制权重柱状图"""
        y_pos = np.arange(len(self.criteria_level1))
        weights_percent = self.weights_level1 * 100

        bars = ax.barh(y_pos, weights_percent, color='#95E1D3')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.criteria_level1)
        ax.set_xlabel('权重 (%)')
        ax.set_title('一级指标权重分布', fontsize=12, fontweight='bold')

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{weights_percent[i]:.2f}%',
                    ha='left', va='center', fontsize=9)

        ax.grid(axis='x', alpha=0.3)

    def _plot_scores_comparison(self, ax):
        """绘制得分对比柱状图"""
        categories = self.criteria_level1
        x = np.arange(len(categories))
        width = 0.35

        usdt_scores = [self.scores['USDT'][cat] for cat in categories]
        usdc_scores = [self.scores['USDC'][cat] for cat in categories]

        ax.bar(x - width/2, usdt_scores, width, label='USDT', color='#FF6B6B')
        ax.bar(x + width/2, usdc_scores, width, label='USDC', color='#4ECDC4')

        ax.set_ylabel('得分')
        ax.set_title('各维度得分对比', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    def _plot_risk_comparison(self, ax):
        """绘制风险对比图"""
        risk_types = self.risks['USDT']['risk_types']
        usdt_risks = self.risks['USDT']['risk_values']
        usdc_risks = self.risks['USDC']['risk_values']

        x = np.arange(len(risk_types))
        width = 0.35

        ax.bar(x - width/2, usdt_risks, width, label='USDT', color='#FF6B6B')
        ax.bar(x + width/2, usdc_risks, width, label='USDC', color='#4ECDC4')

        ax.set_ylabel('风险值')
        ax.set_title('各类风险对比', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(risk_types, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    def _plot_total_scores(self, ax):
        """绘制总分对比"""
        coins = self.coins
        scores = [self.scores[coin]['总分'] for coin in coins]
        colors = ['#FF6B6B', '#4ECDC4']

        bars = ax.bar(coins, scores, color=colors, alpha=0.8)
        ax.set_ylabel('综合得分')
        ax.set_title('综合得分对比', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)

        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)

    def _plot_development_potential(self, ax):
        """绘制发展潜力对比"""
        dpi_results = self.calculate_development_potential()
        coins = self.coins
        dpi_values = [dpi_results[coin] for coin in coins]
        colors = ['#FF6B6B', '#4ECDC4']

        bars = ax.bar(coins, dpi_values, color=colors, alpha=0.8)
        ax.set_ylabel('发展潜力指数')
        ax.set_title('发展潜力对比', fontsize=12, fontweight='bold')

        # 添加数值标签
        for bar, value in zip(bars, dpi_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)

    def generate_report(self):
        """
        生成完整的评价报告
        """
        print("\n" + "="*80)
        print(" "*25 + "USDT与USDC综合评价报告")
        print("="*80)

        # 1. 权重信息
        print("\n【一级指标权重】")
        for i, criterion in enumerate(self.criteria_level1):
            print(f"  {criterion:12s}: {self.weights_level1[i]*100:6.2f}%")

        # 2. 各维度得分
        print("\n【各维度得分对比】(满分100分)")
        print(f"{'评价维度':<15} {'USDT':>10} {'USDC':>10} {'优势方':>10}")
        print("-" * 50)
        for criterion in self.criteria_level1:
            usdt_score = self.scores['USDT'][criterion]
            usdc_score = self.scores['USDC'][criterion]
            winner = 'USDT' if usdt_score > usdc_score else 'USDC'
            print(
                f"{criterion:<15} {usdt_score:>10.2f} {usdc_score:>10.2f} {winner:>10}")

        # 3. 综合得分
        print("\n【综合得分】")
        print(f"  USDT: {self.scores['USDT']['总分']:.2f}")
        print(f"  USDC: {self.scores['USDC']['总分']:.2f}")
        winner = 'USDT' if self.scores['USDT']['总分'] > self.scores['USDC']['总分'] else 'USDC'
        print(f"  综合评价优胜：{winner}")

        # 4. 风险评估
        print("\n【风险评估】")
        print(f"  USDT总体风险值: {self.risks['USDT']['total_risk']:.3f}")
        print(f"  USDC总体风险值: {self.risks['USDC']['total_risk']:.3f}")
        lower_risk = 'USDT' if self.risks['USDT']['total_risk'] < self.risks['USDC']['total_risk'] else 'USDC'
        print(f"  风险更低：{lower_risk}")

        # 5. 发展潜力
        dpi = self.calculate_development_potential()
        print("\n【发展潜力指数】")
        print(f"  USDT: {dpi['USDT']:.3f}")
        print(f"  USDC: {dpi['USDC']:.3f}")
        higher_potential = 'USDT' if dpi['USDT'] > dpi['USDC'] else 'USDC'
        print(f"  发展潜力更大：{higher_potential}")

        # 6. 结论与建议
        print("\n【结论与建议】")
        print("  1. USDT凭借市场表现维度的巨大优势，综合得分略高于USDC")
        print("  2. USDC在合规性和透明度方面具有显著优势，更受监管认可")
        print("  3. USDT风险值更高，主要来自监管和声誉风险")
        print("  4. USDC发展潜力指数更高，在监管趋严的环境下更具优势")
        print("  5. 建议投资者：短期看市场份额选USDT，长期看合规趋势选USDC")

        print("\n" + "="*80 + "\n")


def main():
    """
    主程序入口
    """
    print("="*80)
    print(" "*20 + "稳定币综合评价系统 v1.0")
    print("="*80)
    print("\n正在初始化评价模型...")

    # 创建评价器实例
    evaluator = StablecoinEvaluator()

    # 计算综合评分
    print("\n【步骤1】计算综合评分...")
    scores = evaluator.calculate_comprehensive_scores()

    # 计算风险评估
    print("\n【步骤2】进行风险评估...")
    risks = evaluator.calculate_risk_assessment()

    # 生成报告
    print("\n【步骤3】生成评价报告...")
    evaluator.generate_report()

    # 可视化结果
    print("\n【步骤4】生成可视化图表...")
    evaluator.visualize_results()

    # 保存数据到CSV
    print("\n【步骤5】保存数据...")
    save_results_to_csv(evaluator)

    print("\n✅ 所有分析已完成！")
    print("📊 图表保存位置：第一个问题/USDT_USDC_综合评价结果.png")
    print("📄 数据保存位置：第一个问题/评价结果数据.csv\n")


def save_results_to_csv(evaluator):
    """
    将评价结果保存为CSV文件

    参数：
    -------
    evaluator : StablecoinEvaluator
        评价器实例
    """
    # 创建DataFrame
    data = []
    for criterion in evaluator.criteria_level1:
        data.append({
            '评价维度': criterion,
            'USDT得分': evaluator.scores['USDT'][criterion],
            'USDC得分': evaluator.scores['USDC'][criterion],
            '权重': evaluator.weights_level1[evaluator.criteria_level1.index(criterion)] * 100
        })

    # 添加总分行
    data.append({
        '评价维度': '综合得分',
        'USDT得分': evaluator.scores['USDT']['总分'],
        'USDC得分': evaluator.scores['USDC']['总分'],
        '权重': 100.0
    })

    df = pd.DataFrame(data)
    df.to_csv('第一个问题/评价结果数据.csv', index=False, encoding='utf-8-sig')
    print("  ✓ 评价结果已保存到CSV文件")


if __name__ == "__main__":
    # 运行主程序
    main()
