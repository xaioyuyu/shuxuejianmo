#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定币储备资产配置优化系统
基于多目标优化和投资组合理论的资产配置方案

作者：数学建模团队
日期：2025-11-05
版本：1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, LinearConstraint, Bounds
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体显示
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)


class Asset:
    """
    资产类
    封装单个资产的特征参数
    """

    def __init__(self, name, symbol, expected_return, volatility,
                 liquidation_days, liquidity_score):
        """
        初始化资产对象

        参数:
            name: 资产名称
            symbol: 资产代号
            expected_return: 预期年化收益率(%)
            volatility: 年化波动率(%)
            liquidation_days: 变现所需天数
            liquidity_score: 流动性评分(0-1)
        """
        self.name = name
        self.symbol = symbol
        self.expected_return = expected_return / 100  # 转换为小数
        self.volatility = volatility / 100
        self.liquidation_days = liquidation_days
        self.liquidity_score = liquidity_score

    def __repr__(self):
        return (f"Asset({self.name}, r={self.expected_return*100:.2f}%, "
                f"σ={self.volatility*100:.2f}%, L={self.liquidity_score:.2f})")


class PortfolioOptimizer:
    """
    投资组合优化器
    实现多目标优化、风险控制和资产配置
    """

    def __init__(self, assets, correlation_matrix=None):
        """
        初始化优化器

        参数:
            assets: 资产对象列表
            correlation_matrix: 资产收益率相关系数矩阵(可选)
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.asset_names = [asset.name for asset in assets]

        # 提取资产特征向量
        self.returns = np.array([asset.expected_return for asset in assets])
        self.volatilities = np.array([asset.volatility for asset in assets])
        self.liquidities = np.array(
            [asset.liquidity_score for asset in assets])
        self.liquidation_days = np.array(
            [asset.liquidation_days for asset in assets])

        # 设置相关系数矩阵（默认为独立，除黄金和比特币）
        if correlation_matrix is None:
            self.corr_matrix = np.eye(self.n_assets)
            # 假设黄金(索引4)和比特币(索引5)有0.3的相关性
            if self.n_assets >= 6:
                self.corr_matrix[4, 5] = 0.3
                self.corr_matrix[5, 4] = 0.3
        else:
            self.corr_matrix = correlation_matrix

        # 计算协方差矩阵
        self.cov_matrix = np.outer(
            self.volatilities, self.volatilities) * self.corr_matrix

        # 优化结果
        self.optimal_weights = None
        self.optimal_performance = None

    def portfolio_return(self, weights):
        """
        计算组合预期收益率

        参数:
            weights: 资产配置权重向量

        返回:
            portfolio_return: 组合预期收益率
        """
        return np.dot(weights, self.returns)

    def portfolio_volatility(self, weights):
        """
        计算组合波动率（标准差）

        参数:
            weights: 资产配置权重向量

        返回:
            portfolio_volatility: 组合波动率
        """
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        return np.sqrt(portfolio_variance)

    def portfolio_liquidity(self, weights):
        """
        计算组合流动性指标

        参数:
            weights: 资产配置权重向量

        返回:
            portfolio_liquidity: 组合流动性评分
        """
        return np.dot(weights, self.liquidities)

    def objective_function(self, weights, w_return=0.4, w_risk=0.3, w_liquidity=0.3):
        """
        多目标优化的目标函数（负值，因为scipy.optimize.minimize是最小化）

        参数:
            weights: 资产配置权重
            w_return: 收益目标权重
            w_risk: 风险目标权重
            w_liquidity: 流动性目标权重

        返回:
            objective_value: 目标函数值（越小越好）
        """
        # 计算各个指标
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        liq = self.portfolio_liquidity(weights)

        # 归一化（使用最大可能值）
        ret_norm = ret / np.max(self.returns)
        vol_norm = vol / np.max(self.volatilities)
        liq_norm = liq / 1.0

        # 组合目标：最大化收益和流动性，最小化风险
        # 转换为最小化问题：-收益 + 风险 - 流动性
        objective = -w_return * ret_norm + w_risk * \
            vol_norm - w_liquidity * liq_norm

        return objective

    def optimize(self, strategy='balanced', constraints_dict=None):
        """
        执行投资组合优化

        参数:
            strategy: 策略类型 ('conservative', 'balanced', 'aggressive')
            constraints_dict: 自定义约束条件字典

        返回:
            optimal_weights: 最优权重向量
            performance: 组合性能指标字典
        """
        # 根据策略设置目标权重
        strategy_weights = {
            'conservative': {'w_return': 0.3, 'w_risk': 0.4, 'w_liquidity': 0.3},
            'balanced': {'w_return': 0.4, 'w_risk': 0.3, 'w_liquidity': 0.3},
            'aggressive': {'w_return': 0.5, 'w_risk': 0.2, 'w_liquidity': 0.3}
        }

        w_params = strategy_weights.get(
            strategy, strategy_weights['balanced'])

        # 初始猜测（等权重）
        x0 = np.ones(self.n_assets) / self.n_assets

        # 设置约束条件
        constraints = []

        # 约束1：权重和为1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # 约束2：最低流动性要求
        min_liquidity = 0.70 if constraints_dict is None else constraints_dict.get(
            'min_liquidity', 0.70)
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.portfolio_liquidity(x) - min_liquidity
        })

        # 约束3：最低现金比例（资产索引0）
        min_cash = 0.05 if constraints_dict is None else constraints_dict.get(
            'min_cash', 0.05)
        constraints.append({'type': 'ineq', 'fun': lambda x: x[0] - min_cash})

        # 约束4：7日流动性覆盖率（假设正常情景需要覆盖3%赎回）
        min_7day_coverage = 0.12 if constraints_dict is None else constraints_dict.get(
            'min_7day_coverage', 0.12)

        def liquidity_coverage_7day(x):
            # 计算7天内可变现资产的加权流动性
            mask = self.liquidation_days <= 7
            coverage = np.sum(x[mask] * self.liquidities[mask])
            return coverage - min_7day_coverage

        constraints.append({'type': 'ineq', 'fun': liquidity_coverage_7day})

        # 设置变量边界
        bounds = Bounds(
            lb=np.array([0.0] * self.n_assets),  # 下界：所有资产>=0
            ub=np.array([0.50, 0.60, 0.30, 0.30, 0.10, 0.05])  # 上界：单一资产限制
        )

        # 执行优化
        result = minimize(
            fun=lambda x: self.objective_function(x, **w_params),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"⚠️  优化警告: {result.message}")

        # 保存结果
        self.optimal_weights = result.x

        # 计算性能指标
        self.optimal_performance = {
            'return': self.portfolio_return(result.x) * 100,
            'volatility': self.portfolio_volatility(result.x) * 100,
            'liquidity': self.portfolio_liquidity(result.x),
            'sharpe_ratio': self.portfolio_return(result.x) / self.portfolio_volatility(result.x) if self.portfolio_volatility(result.x) > 0 else 0
        }

        return self.optimal_weights, self.optimal_performance

    def efficient_frontier(self, n_points=50, target_returns=None):
        """
        计算有效前沿

        参数:
            n_points: 前沿上的点数
            target_returns: 目标收益率数组（可选）

        返回:
            frontier_returns: 前沿上的收益率
            frontier_volatilities: 前沿上的波动率
            frontier_weights: 前沿上的权重
        """
        if target_returns is None:
            # 自动生成目标收益率范围
            min_return = np.min(self.returns)
            max_return = np.max(self.returns)
            target_returns = np.linspace(
                min_return * 0.8, max_return * 0.6, n_points)

        frontier_volatilities = []
        frontier_returns_actual = []
        frontier_weights = []

        # 基本约束（权重和为1，非负）
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = Bounds(lb=np.zeros(
            self.n_assets), ub=np.ones(self.n_assets))

        for target_ret in target_returns:
            # 添加目标收益率约束
            constraints_with_return = constraints + [
                {'type': 'eq', 'fun': lambda x,
                    tr=target_ret: self.portfolio_return(x) - tr}
            ]

            # 最小化波动率
            result = minimize(
                fun=lambda x: self.portfolio_volatility(x),
                x0=np.ones(self.n_assets) / self.n_assets,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_with_return,
                options={'maxiter': 500, 'ftol': 1e-7}
            )

            if result.success:
                frontier_returns_actual.append(
                    self.portfolio_return(result.x))
                frontier_volatilities.append(
                    self.portfolio_volatility(result.x))
                frontier_weights.append(result.x)

        return (np.array(frontier_returns_actual) * 100,
                np.array(frontier_volatilities) * 100,
                frontier_weights)


class ScenarioAnalyzer:
    """
    场景分析器
    模拟不同市场环境和赎回压力下的组合表现
    """

    def __init__(self, optimizer):
        """
        初始化场景分析器

        参数:
            optimizer: PortfolioOptimizer对象
        """
        self.optimizer = optimizer

        # 定义标准场景
        self.scenarios = {
            'normal': {
                'probability': 0.70,
                'daily_redemption': 0.005,
                '7day_redemption': 0.03,
                '30day_redemption': 0.10,
                'description': '正常市场环境'
            },
            'stress': {
                'probability': 0.25,
                'daily_redemption': 0.02,
                '7day_redemption': 0.12,
                '30day_redemption': 0.30,
                'description': '压力情景'
            },
            'extreme': {
                'probability': 0.05,
                'daily_redemption': 0.05,
                '7day_redemption': 0.25,
                '30day_redemption': 0.50,
                'description': '极端情景'
            }
        }

    def analyze_scenario(self, weights, scenario_name):
        """
        分析特定场景下的组合表现

        参数:
            weights: 资产配置权重
            scenario_name: 场景名称

        返回:
            analysis: 场景分析结果字典
        """
        scenario = self.scenarios[scenario_name]

        # 计算基本性能指标
        ret = self.optimizer.portfolio_return(weights)
        vol = self.optimizer.portfolio_volatility(weights)
        liq = self.optimizer.portfolio_liquidity(weights)

        # 计算流动性覆盖率
        mask_7day = self.optimizer.liquidation_days <= 7
        coverage_7day = np.sum(
            weights[mask_7day] * self.optimizer.liquidities[mask_7day])

        mask_30day = self.optimizer.liquidation_days <= 30
        coverage_30day = np.sum(
            weights[mask_30day] * self.optimizer.liquidities[mask_30day])

        # 流动性充足性判断
        is_7day_sufficient = coverage_7day >= scenario['7day_redemption']
        is_30day_sufficient = coverage_30day >= scenario['30day_redemption']

        analysis = {
            'scenario': scenario_name,
            'description': scenario['description'],
            'probability': scenario['probability'],
            'expected_return': ret * 100,
            'volatility': vol * 100,
            'liquidity_score': liq,
            '7day_coverage': coverage_7day * 100,
            '7day_requirement': scenario['7day_redemption'] * 100,
            '7day_sufficient': is_7day_sufficient,
            '30day_coverage': coverage_30day * 100,
            '30day_requirement': scenario['30day_redemption'] * 100,
            '30day_sufficient': is_30day_sufficient,
            'risk_level': self._assess_risk_level(
                is_7day_sufficient, is_30day_sufficient)
        }

        return analysis

    def _assess_risk_level(self, is_7day_ok, is_30day_ok):
        """评估风险等级"""
        if is_7day_ok and is_30day_ok:
            return "低风险 ✓"
        elif is_7day_ok:
            return "中风险"
        else:
            return "高风险 ⚠️"

    def analyze_all_scenarios(self, weights):
        """
        分析所有场景

        参数:
            weights: 资产配置权重

        返回:
            results: 所有场景的分析结果列表
        """
        results = []
        for scenario_name in self.scenarios.keys():
            analysis = self.analyze_scenario(weights, scenario_name)
            results.append(analysis)
        return results


class StressTester:
    """
    压力测试器
    模拟极端事件对组合的冲击
    """

    def __init__(self, optimizer):
        """
        初始化压力测试器

        参数:
            optimizer: PortfolioOptimizer对象
        """
        self.optimizer = optimizer

    def test_asset_shock(self, weights, asset_index, shock_percent):
        """
        测试单一资产价格冲击

        参数:
            weights: 资产配置权重
            asset_index: 受冲击的资产索引
            shock_percent: 冲击幅度(%)，负值表示下跌

        返回:
            impact: 对组合价值的影响(%)
        """
        # 计算该资产对组合的贡献
        asset_contribution = weights[asset_index]

        # 组合价值损失
        portfolio_loss = asset_contribution * (shock_percent / 100)

        return portfolio_loss * 100

    def test_multiple_shocks(self, weights, shock_dict):
        """
        测试多个资产同时受冲击

        参数:
            weights: 资产配置权重
            shock_dict: {资产索引: 冲击幅度(%)}

        返回:
            total_impact: 总影响(%)
        """
        total_loss = 0
        for asset_idx, shock_pct in shock_dict.items():
            loss = self.test_asset_shock(weights, asset_idx, shock_pct)
            total_loss += loss

        return total_loss

    def test_redemption_pressure(self, weights, redemption_rate):
        """
        测试赎回压力

        参数:
            weights: 资产配置权重
            redemption_rate: 赎回率(%)

        返回:
            result: 赎回压力测试结果
        """
        # 计算立即可用资产（现金 + 1天内可变现）
        mask_immediate = self.optimizer.liquidation_days <= 1
        immediate_liquidity = np.sum(
            weights[mask_immediate] * self.optimizer.liquidities[mask_immediate])

        # 7天内可用
        mask_7day = self.optimizer.liquidation_days <= 7
        liquidity_7day = np.sum(
            weights[mask_7day] * self.optimizer.liquidities[mask_7day])

        # 判断是否能应对
        can_handle_immediate = immediate_liquidity >= redemption_rate / 100
        can_handle_7day = liquidity_7day >= redemption_rate / 100

        result = {
            'redemption_rate': redemption_rate,
            'immediate_liquidity': immediate_liquidity * 100,
            'liquidity_7day': liquidity_7day * 100,
            'can_handle_immediate': can_handle_immediate,
            'can_handle_7day': can_handle_7day,
            'assessment': self._assess_redemption_capability(
                can_handle_immediate, can_handle_7day)
        }

        return result

    def _assess_redemption_capability(self, immediate_ok, week_ok):
        """评估赎回应对能力"""
        if immediate_ok:
            return "可立即应对 ✓"
        elif week_ok:
            return "需要1-7天变现部分资产"
        else:
            return "流动性不足，存在脱锚风险 ⚠️"


class Visualizer:
    """
    可视化工具类
    生成各种分析图表
    """

    @staticmethod
    def plot_allocation(weights, asset_names, title="资产配置方案"):
        """
        绘制饼图展示资产配置

        参数:
            weights: 权重向量
            asset_names: 资产名称列表
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1',
                  '#FFA07A', '#98D8C8', '#F7DC6F']
        explode = [0.05 if w > 0.15 else 0 for w in weights]

        wedges, texts, autotexts = ax.pie(
            weights, labels=asset_names, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )

        # 美化百分比文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_efficient_frontier(frontier_returns, frontier_vols,
                                optimal_point=None, title="有效前沿"):
        """
        绘制有效前沿曲线

        参数:
            frontier_returns: 前沿上的收益率
            frontier_vols: 前沿上的波动率
            optimal_point: 最优点(return, volatility)元组
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制有效前沿
        ax.plot(frontier_vols, frontier_returns, 'b-',
                linewidth=2.5, label='有效前沿')
        ax.fill_between(frontier_vols, frontier_returns, alpha=0.2)

        # 标记最优点
        if optimal_point is not None:
            opt_ret, opt_vol = optimal_point
            ax.scatter(opt_vol, opt_ret, color='red', s=200,
                       marker='*', zorder=5, label='最优配置')
            ax.annotate(f'({opt_vol:.2f}%, {opt_ret:.2f}%)',
                        xy=(opt_vol, opt_ret), xytext=(
                            opt_vol+0.3, opt_ret+0.2),
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xlabel('波动率 (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('预期收益率 (%)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_scenario_comparison(scenario_results, title="场景分析对比"):
        """
        绘制场景分析对比图

        参数:
            scenario_results: 场景分析结果列表
            title: 图表标题
        """
        scenarios = [r['scenario'] for r in scenario_results]
        coverage_7day = [r['7day_coverage'] for r in scenario_results]
        requirement_7day = [r['7day_requirement'] for r in scenario_results]

        x = np.arange(len(scenarios))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width/2, coverage_7day, width,
                       label='流动性覆盖率', color='#4ECDC4')
        bars2 = ax.bar(x + width/2, requirement_7day, width,
                       label='赎回需求', color='#FF6B6B')

        ax.set_xlabel('场景', fontsize=13, fontweight='bold')
        ax.set_ylabel('比例 (%)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return fig


def main():
    """
    主程序：执行完整的稳定币储备资产配置优化
    """
    print("\n" + "="*80)
    print("稳定币储备资产配置优化系统".center(80))
    print("="*80 + "\n")

    # ========== 第一步：定义资产 ==========
    print("【步骤1】定义资产类别及其特征参数...\n")

    assets = [
        Asset("现金", "Cash", 0, 0, 0, 1.00),
        Asset("短期国债", "T-Bills", 4.5, 1.2, 1, 0.95),
        Asset("商业票据", "CP", 5.2, 2.5, 5, 0.70),
        Asset("货币基金", "MMF", 4.8, 1.5, 2, 0.85),
        Asset("黄金", "Gold", 6.0, 15.0, 2, 0.60),
        Asset("比特币", "BTC", 15.0, 60.0, 0, 0.50)
    ]

    # 打印资产信息
    print("=" * 70)
    print("资产特征参数表")
    print("=" * 70)
    print(f"{'资产名称':<10} {'预期收益率':>12} {'波动率':>10} "
          f"{'变现时间':>10} {'流动性':>10}")
    print("-" * 70)
    for asset in assets:
        print(f"{asset.name:<10} {asset.expected_return*100:>11.2f}% "
              f"{asset.volatility*100:>9.2f}% {asset.liquidation_days:>9}天 "
              f"{asset.liquidity_score:>9.2f}")
    print("=" * 70 + "\n")

    # ========== 第二步：创建优化器 ==========
    print("【步骤2】创建投资组合优化器并设置约束条件...\n")

    optimizer = PortfolioOptimizer(assets)
    print(f"✓ 优化器已初始化")
    print(f"  - 资产数量: {optimizer.n_assets}")
    print(f"  - 相关系数矩阵维度: {optimizer.corr_matrix.shape}")
    print(f"  - 协方差矩阵已计算\n")

    # ========== 第三步：执行多策略优化 ==========
    print("【步骤3】执行不同风险偏好下的资产配置优化...\n")

    strategies = ['conservative', 'balanced', 'aggressive']
    strategy_names = {'conservative': '保守型', 'balanced': '平衡型',
                      'aggressive': '激进型'}

    results_summary = []

    for strategy in strategies:
        print(f"正在优化: {strategy_names[strategy]}策略...")

        weights, performance = optimizer.optimize(strategy=strategy)

        results_summary.append({
            'strategy': strategy_names[strategy],
            'weights': weights,
            'return': performance['return'],
            'volatility': performance['volatility'],
            'liquidity': performance['liquidity'],
            'sharpe': performance['sharpe_ratio']
        })

        print(f"  ✓ 优化完成")
        print(f"    预期收益率: {performance['return']:.2f}%")
        print(f"    波动率: {performance['volatility']:.2f}%")
        print(f"    流动性评分: {performance['liquidity']:.3f}")
        print(f"    夏普比率: {performance['sharpe']:.3f}\n")

    # ========== 第四步：输出详细配置方案 ==========
    print("【步骤4】生成详细配置方案表...\n")

    print("=" * 90)
    print("不同策略下的最优资产配置方案")
    print("=" * 90)

    # 创建DataFrame
    allocation_data = []
    for result in results_summary:
        row = {'策略': result['strategy']}
        for i, asset in enumerate(assets):
            row[asset.name] = f"{result['weights'][i]*100:.1f}%"
        row['预期收益'] = f"{result['return']:.2f}%"
        row['波动率'] = f"{result['volatility']:.2f}%"
        row['流动性'] = f"{result['liquidity']:.3f}"
        allocation_data.append(row)

    df = pd.DataFrame(allocation_data)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    # 保存到CSV
    df.to_csv("第二个问题/资产配置方案对比.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 配置方案已保存: 资产配置方案对比.csv\n")

    # ========== 第五步：场景分析 ==========
    print("【步骤5】执行场景分析...\n")

    scenario_analyzer = ScenarioAnalyzer(optimizer)

    # 选择平衡型配置进行场景分析
    balanced_weights = results_summary[1]['weights']

    scenario_results = scenario_analyzer.analyze_all_scenarios(
        balanced_weights)

    print("=" * 80)
    print("场景分析结果（基于平衡型配置）")
    print("=" * 80)

    for result in scenario_results:
        print(
            f"\n场景: {result['description']} (概率: {result['probability']*100:.0f}%)")
        print("-" * 80)
        print(f"  7日流动性覆盖率: {result['7day_coverage']:.2f}% "
              f"(需求: {result['7day_requirement']:.2f}%) "
              f"{'✓ 充足' if result['7day_sufficient'] else '⚠️ 不足'}")
        print(f"  30日流动性覆盖率: {result['30day_coverage']:.2f}% "
              f"(需求: {result['30day_requirement']:.2f}%) "
              f"{'✓ 充足' if result['30day_sufficient'] else '⚠️ 不足'}")
        print(f"  风险等级: {result['risk_level']}")

    print("=" * 80 + "\n")

    # ========== 第六步：压力测试 ==========
    print("【步骤6】执行压力测试...\n")

    stress_tester = StressTester(optimizer)

    print("=" * 70)
    print("压力测试结果")
    print("=" * 70)

    # 测试1：比特币暴跌50%
    print("\n测试1: 比特币价格下跌50%")
    btc_shock = stress_tester.test_asset_shock(balanced_weights, 5, -50)
    print(f"  组合价值影响: {btc_shock:+.2f}%")
    print(f"  评估: {'影响可控 ✓' if abs(btc_shock) < 2 else '影响较大 ⚠️'}")

    # 测试2：商业票据违约5%
    print("\n测试2: 商业票据部分违约(5%)")
    cp_shock = stress_tester.test_asset_shock(balanced_weights, 2, -5)
    print(f"  组合价值影响: {cp_shock:+.2f}%")
    print(f"  评估: {'影响可控 ✓' if abs(cp_shock) < 2 else '影响较大 ⚠️'}")

    # 测试3：多重冲击
    print("\n测试3: 多重冲击场景")
    print("  - 比特币下跌40%")
    print("  - 黄金下跌20%")
    print("  - 商业票据违约3%")
    multi_shock = stress_tester.test_multiple_shocks(
        balanced_weights, {5: -40, 4: -20, 2: -3})
    print(f"  组合价值总影响: {multi_shock:+.2f}%")
    print(f"  评估: {'可承受 ✓' if abs(multi_shock) < 5 else '风险较高 ⚠️'}")

    # 测试4：极端赎回压力
    print("\n测试4: 极端赎回压力(20%)")
    redemption_test = stress_tester.test_redemption_pressure(
        balanced_weights, 20)
    print(f"  立即可用流动性: {redemption_test['immediate_liquidity']:.2f}%")
    print(f"  7日内可用流动性: {redemption_test['liquidity_7day']:.2f}%")
    print(f"  应对能力: {redemption_test['assessment']}")

    print("=" * 70 + "\n")

    # ========== 第七步：计算有效前沿 ==========
    print("【步骤7】计算投资组合有效前沿...\n")

    print("正在计算有效前沿（这可能需要几秒钟）...")
    frontier_returns, frontier_vols, frontier_weights = optimizer.efficient_frontier(
        n_points=30)
    print(f"  ✓ 有效前沿计算完成，共{len(frontier_returns)}个点\n")

    # ========== 第八步：可视化 ==========
    print("【步骤8】生成可视化图表...\n")

    visualizer = Visualizer()

    # 1. 资产配置饼图（平衡型）
    fig1 = visualizer.plot_allocation(
        balanced_weights,
        [asset.name for asset in assets],
        "平衡型策略 - 最优资产配置方案"
    )
    fig1.savefig("第二个问题/资产配置饼图.png", dpi=300, bbox_inches='tight')
    print("  ✓ 配置饼图已保存: 资产配置饼图.png")

    # 2. 有效前沿图
    balanced_perf = results_summary[1]
    fig2 = visualizer.plot_efficient_frontier(
        frontier_returns, frontier_vols,
        optimal_point=(balanced_perf['volatility'], balanced_perf['return']),
        title="投资组合有效前沿"
    )
    fig2.savefig("第二个问题/有效前沿曲线.png", dpi=300, bbox_inches='tight')
    print("  ✓ 有效前沿图已保存: 有效前沿曲线.png")

    # 3. 场景对比图
    fig3 = visualizer.plot_scenario_comparison(
        scenario_results, "不同场景下的流动性覆盖分析")
    fig3.savefig("第二个问题/场景分析对比.png", dpi=300, bbox_inches='tight')
    print("  ✓ 场景对比图已保存: 场景分析对比.png\n")

    # ========== 总结 ==========
    print("=" * 80)
    print("优化分析完成！".center(80))
    print("=" * 80)

    print("\n核心发现:")
    print(f"  1. 平衡型配置预期收益率: {balanced_perf['return']:.2f}%")
    print(f"  2. 平衡型配置波动率: {balanced_perf['volatility']:.2f}%")
    print(f"  3. 平衡型配置流动性评分: {balanced_perf['liquidity']:.3f}")
    print(f"  4. 短期国债占比最高: {balanced_weights[1]*100:.1f}%")

    print("\n建议:")
    print("  • 短期国债作为核心资产，提供稳定收益和高流动性")
    print("  • 保持8-10%现金储备应对日常赎回")
    print("  • 比特币和黄金配置保守(合计<10%)，控制波动风险")
    print("  • 定期（每月/每季度）进行再平衡和压力测试")

    print("\n所有结果文件已保存至 '第二个问题' 文件夹")
    print("=" * 80 + "\n")

    # 可选：显示图表
    # plt.show()


if __name__ == "__main__":
    """程序入口"""
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
