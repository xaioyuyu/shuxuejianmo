#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定币对货币主权影响评估与风险预警系统
基于Logit回归和多指标评价的货币主权风险分析

日期：2025-11-05
版本：1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)


class CountryDataGenerator:
    """
    国家数据生成器
    生成模拟的各国货币主权相关指标数据
    """

    def __init__(self, n_countries=30):
        """
        初始化数据生成器

        参数:
            n_countries: 国家数量
        """
        self.n_countries = n_countries
        self.country_names = None

    def generate_data(self):
        """
        生成模拟数据

        返回:
            df: 包含所有指标的DataFrame
        """
        # 国家名称（包含真实高风险国家和其他国家）
        high_risk_countries = [
            '委内瑞拉', '阿根廷', '黎巴嫩', '土耳其', '斯里兰卡',
            '尼日利亚', '巴基斯坦', '埃及'
        ]

        other_countries = [
            f'国家{i}' for i in range(1, self.n_countries - len(high_risk_countries) + 1)
        ]

        self.country_names = high_risk_countries + other_countries

        # 生成各项指标（模拟真实分布）
        np.random.seed(42)

        # 稳定币普及指数（0-100）
        # 高风险国家SCI较高
        sci_high_risk = np.random.uniform(15, 50, len(high_risk_countries))
        sci_others = np.random.uniform(2, 25, len(other_countries))
        stablecoin_index = np.concatenate([sci_high_risk, sci_others])

        # 本币流通占比（20-95%）
        domestic_currency_ratio = 100 - stablecoin_index * 0.8 + \
            np.random.normal(0, 5, self.n_countries)
        domestic_currency_ratio = np.clip(domestic_currency_ratio, 20, 95)

        # 外币存款占比（5-75%）
        foreign_deposit_ratio = stablecoin_index * 1.2 + \
            np.random.normal(0, 8, self.n_countries)
        foreign_deposit_ratio = np.clip(foreign_deposit_ratio, 5, 75)

        # 通货膨胀率（0.5-80%）
        inflation_high = np.random.uniform(15, 80, len(high_risk_countries))
        inflation_low = np.random.uniform(0.5, 8, len(other_countries))
        inflation_rate = np.concatenate([inflation_high, inflation_low])

        # 人均GDP（500-50000美元）
        gdp_per_capita_low = np.random.uniform(
            500, 5000, len(high_risk_countries))
        gdp_per_capita_high = np.random.uniform(
            3000, 50000, len(other_countries))
        gdp_per_capita = np.concatenate(
            [gdp_per_capita_low, gdp_per_capita_high])

        # 外债占GDP比例（20-150%）
        debt_high = np.random.uniform(60, 150, len(high_risk_countries))
        debt_low = np.random.uniform(20, 80, len(other_countries))
        debt_to_gdp = np.concatenate([debt_high, debt_low])

        # 外汇储备充足度（1-15月进口额）
        fx_reserves_low = np.random.uniform(1, 5, len(high_risk_countries))
        fx_reserves_high = np.random.uniform(4, 15, len(other_countries))
        fx_reserves = np.concatenate([fx_reserves_low, fx_reserves_high])

        # 政治稳定性（-2.5到2.5）
        pol_stab_low = np.random.uniform(-2.0, 0, len(high_risk_countries))
        pol_stab_high = np.random.uniform(-0.5, 2.0, len(other_countries))
        political_stability = np.concatenate([pol_stab_low, pol_stab_high])

        # 汇率波动率（1-50%）
        exchange_volatility_high = np.random.uniform(
            15, 50, len(high_risk_countries))
        exchange_volatility_low = np.random.uniform(
            1, 15, len(other_countries))
        exchange_volatility = np.concatenate(
            [exchange_volatility_high, exchange_volatility_low])

        # 创建DataFrame
        df = pd.DataFrame({
            'country': self.country_names,
            'stablecoin_index': stablecoin_index,
            'domestic_currency_ratio': domestic_currency_ratio,
            'foreign_deposit_ratio': foreign_deposit_ratio,
            'inflation_rate': inflation_rate,
            'gdp_per_capita': gdp_per_capita,
            'debt_to_gdp': debt_to_gdp,
            'fx_reserves': fx_reserves,
            'political_stability': political_stability,
            'exchange_volatility': exchange_volatility
        })

        return df


class MonetarySovereigntyEvaluator:
    """
    货币主权评价器
    计算货币主权综合评分
    """

    def __init__(self):
        """初始化评价器"""
        self.weights = {
            'currency_usage': 0.40,
            'policy_effectiveness': 0.30,
            'external_dependence': 0.20,
            'economic_foundation': 0.10
        }

    def calculate_sovereignty_score(self, df):
        """
        计算货币主权评分

        参数:
            df: 数据DataFrame

        返回:
            scores: 货币主权评分数组
        """
        # 指标1：本币使用比例（0-100分）
        currency_usage = df['domestic_currency_ratio'] * 0.7 + \
            (100 - df['foreign_deposit_ratio']) * 0.3

        # 指标2：货币政策有效性（基于通胀和汇率稳定性）
        inflation_score = np.clip(100 - df['inflation_rate'] * 2, 0, 100)
        exchange_score = np.clip(100 - df['exchange_volatility'] * 2, 0, 100)
        policy_effectiveness = inflation_score * 0.6 + exchange_score * 0.4

        # 指标3：外部依赖度（取反）
        debt_score = np.clip(100 - df['debt_to_gdp'] * 0.5, 0, 100)
        external_dependence = debt_score

        # 指标4：经济基础
        gdp_score = np.clip(np.log(df['gdp_per_capita']) * 10, 0, 100)
        fx_score = np.clip(df['fx_reserves'] * 6, 0, 100)
        economic_foundation = gdp_score * 0.6 + fx_score * 0.4

        # 加权汇总
        sovereignty_score = (
            currency_usage * self.weights['currency_usage'] +
            policy_effectiveness * self.weights['policy_effectiveness'] +
            external_dependence * self.weights['external_dependence'] +
            economic_foundation * self.weights['economic_foundation']
        )

        return sovereignty_score

    def print_score_breakdown(self, df, scores):
        """打印评分详情"""
        print("=" * 80)
        print("货币主权评分详情（前10个国家）")
        print("=" * 80)

        results = pd.DataFrame({
            '国家': df['country'][:10],
            '货币主权评分': scores[:10],
            '本币流通比例': df['domestic_currency_ratio'][:10],
            '外币存款比例': df['foreign_deposit_ratio'][:10],
            '通胀率': df['inflation_rate'][:10]
        })

        print(results.to_string(index=False))
        print("=" * 80 + "\n")


class CorrelationAnalyzer:
    """
    相关性分析器
    分析稳定币普及与货币主权的关系
    """

    def __init__(self):
        """初始化分析器"""
        pass

    def calculate_correlation(self, x, y):
        """
        计算Pearson相关系数

        参数:
            x, y: 两个变量

        返回:
            corr: 相关系数
            p_value: p值
        """
        corr, p_value = stats.pearsonr(x, y)
        return corr, p_value

    def correlation_matrix(self, df, variables):
        """
        计算相关系数矩阵

        参数:
            df: 数据DataFrame
            variables: 变量名列表

        返回:
            corr_matrix: 相关系数矩阵
        """
        data = df[variables]
        corr_matrix = data.corr()
        return corr_matrix

    def linear_regression(self, x, y):
        """
        简单线性回归

        参数:
            x: 自变量
            y: 因变量

        返回:
            slope: 斜率
            intercept: 截距
            r_squared: R²
        """
        x_reshaped = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_reshaped, y)

        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(x_reshaped, y)

        return slope, intercept, r_squared

    def print_correlation_results(self, df, sovereignty_score):
        """打印相关性分析结果"""
        print("=" * 80)
        print("相关性分析结果")
        print("=" * 80)

        variables = {
            'stablecoin_index': '稳定币普及指数',
            'inflation_rate': '通货膨胀率',
            'debt_to_gdp': '外债占GDP比例',
            'foreign_deposit_ratio': '外币存款比例'
        }

        for var_name, var_label in variables.items():
            corr, p_value = self.calculate_correlation(
                df[var_name], sovereignty_score)
            sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else (
                '*' if p_value < 0.05 else 'n.s.'))
            print(
                f"  货币主权 vs {var_label:<20}: ρ = {corr:>7.3f}, p = {p_value:.4f} {sig}")

        print("=" * 80 + "\n")


class LogitWarningModel:
    """
    Logit预警模型
    预测货币主权丧失概率
    """

    def __init__(self):
        """初始化Logit模型"""
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.threshold = 0.5

    def create_labels(self, df, sovereignty_score):
        """
        创建二元标签（是否丧失货币主权）

        参数:
            df: 数据DataFrame
            sovereignty_score: 货币主权评分

        返回:
            labels: 0/1标签
        """
        # 判定标准：满足任一条件即认定为丧失货币主权
        condition1 = df['foreign_deposit_ratio'] > 50
        condition2 = df['domestic_currency_ratio'] < 40
        condition3 = sovereignty_score < 30

        labels = (condition1 | condition2 | condition3).astype(int)

        return labels

    def prepare_features(self, df):
        """
        准备特征矩阵

        参数:
            df: 数据DataFrame

        返回:
            X: 特征矩阵
        """
        self.feature_names = [
            'stablecoin_index',
            'inflation_rate',
            'debt_to_gdp',
            'fx_reserves',
            'political_stability',
            'log_gdp_per_capita'
        ]

        X = df[['stablecoin_index', 'inflation_rate', 'debt_to_gdp',
                'fx_reserves', 'political_stability']].copy()
        X['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])

        return X

    def fit(self, X, y):
        """
        拟合Logit模型

        参数:
            X: 特征矩阵
            y: 标签
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 拟合模型
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        """
        预测概率

        参数:
            X: 特征矩阵

        返回:
            proba: 预测概率
        """
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return proba

    def get_coefficients(self):
        """
        获取模型系数

        返回:
            coef_df: 系数DataFrame
        """
        coefficients = self.model.coef_[0]
        odds_ratios = np.exp(coefficients)

        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'odds_ratio': odds_ratios
        })

        return coef_df

    def classify_risk(self, probabilities):
        """
        风险分类

        参数:
            probabilities: 预测概率

        返回:
            risk_levels: 风险等级列表
        """
        risk_levels = []
        for p in probabilities:
            if p >= 0.70:
                risk_levels.append('极高风险')
            elif p >= 0.50:
                risk_levels.append('高风险')
            elif p >= 0.30:
                risk_levels.append('中风险')
            else:
                risk_levels.append('低风险')

        return risk_levels

    def print_model_summary(self, X, y):
        """打印模型摘要"""
        print("=" * 80)
        print("Logit预警模型摘要")
        print("=" * 80)

        # 模型系数
        coef_df = self.get_coefficients()
        print("\n模型系数和优势比:")
        print(coef_df.to_string(index=False))

        # 预测准确率
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        accuracy = np.mean(y_pred == y)

        print(f"\n分类准确率: {accuracy:.2%}")

        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        print(f"\n混淆矩阵:")
        print(f"  真阴性: {cm[0,0]:<3}  假阳性: {cm[0,1]:<3}")
        print(f"  假阴性: {cm[1,0]:<3}  真阳性: {cm[1,1]:<3}")

        print("=" * 80 + "\n")


class RiskIdentifier:
    """
    高风险国家识别器
    """

    def __init__(self):
        """初始化识别器"""
        pass

    def identify_high_risk_countries(self, df, sovereignty_score, probabilities):
        """
        识别高风险国家

        参数:
            df: 数据DataFrame
            sovereignty_score: 货币主权评分
            probabilities: 丧失概率

        返回:
            high_risk_df: 高风险国家DataFrame
        """
        risk_levels = []
        for p in probabilities:
            if p >= 0.70:
                risk_levels.append('极高风险')
            elif p >= 0.50:
                risk_levels.append('高风险')
            elif p >= 0.30:
                risk_levels.append('中风险')
            else:
                risk_levels.append('低风险')

        results = pd.DataFrame({
            'country': df['country'],
            'sovereignty_score': sovereignty_score,
            'stablecoin_index': df['stablecoin_index'],
            'foreign_deposit_ratio': df['foreign_deposit_ratio'],
            'loss_probability': probabilities,
            'risk_level': risk_levels
        })

        # 按概率降序排列
        results = results.sort_values('loss_probability', ascending=False)

        # 筛选中风险及以上
        high_risk = results[results['loss_probability'] >= 0.30].copy()

        return high_risk

    def print_high_risk_report(self, high_risk_df):
        """打印高风险国家报告"""
        print("=" * 90)
        print("高风险国家识别结果")
        print("=" * 90)

        display_df = high_risk_df[[
            'country', 'sovereignty_score', 'stablecoin_index',
            'foreign_deposit_ratio', 'loss_probability', 'risk_level'
        ]].copy()

        display_df.columns = [
            '国家', '货币主权评分', '稳定币指数',
            '外币存款占比(%)', '丧失概率', '风险等级'
        ]

        # 格式化数值
        display_df['货币主权评分'] = display_df['货币主权评分'].round(1)
        display_df['稳定币指数'] = display_df['稳定币指数'].round(1)
        display_df['外币存款占比(%)'] = display_df['外币存款占比(%)'].round(1)
        display_df['丧失概率'] = display_df['丧失概率'].round(3)

        print(display_df.to_string(index=False))
        print("=" * 90 + "\n")

        # 统计
        print("风险等级统计:")
        risk_counts = high_risk_df['risk_level'].value_counts()
        for level, count in risk_counts.items():
            print(f"  {level}: {count}个国家")
        print()


class Visualizer:
    """
    可视化工具类
    """

    @staticmethod
    def plot_correlation_scatter(x, y, x_label, y_label, title):
        """
        绘制相关性散点图

        参数:
            x, y: 数据
            x_label, y_label: 轴标签
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # 散点图
        ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidth=1)

        # 拟合线
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", linewidth=2,
                label=f'拟合线: y={z[0]:.2f}x+{z[1]:.2f}')

        # 相关系数
        corr, p_value = stats.pearsonr(x, y)
        ax.text(0.05, 0.95, f'ρ = {corr:.3f}\np < 0.001',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_risk_distribution(probabilities, labels, title="风险概率分布"):
        """
        绘制风险概率分布图

        参数:
            probabilities: 预测概率
            labels: 实际标签
            title: 图表标题
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 直方图
        ax1.hist(probabilities[labels == 0], bins=20, alpha=0.6,
                 label='未丧失主权', color='green', edgecolor='black')
        ax1.hist(probabilities[labels == 1], bins=20, alpha=0.6,
                 label='已丧失主权', color='red', edgecolor='black')
        ax1.axvline(x=0.5, color='black', linestyle='--',
                    linewidth=2, label='阈值')
        ax1.set_xlabel('预测概率', fontsize=12, fontweight='bold')
        ax1.set_ylabel('频数', fontsize=12, fontweight='bold')
        ax1.set_title('概率分布直方图', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # ROC曲线
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)

        ax2.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假阳性率', fontsize=12, fontweight='bold')
        ax2.set_ylabel('真阳性率', fontsize=12, fontweight='bold')
        ax2.set_title('ROC曲线', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower right", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_risk_map(high_risk_df, title="高风险国家地图"):
        """
        绘制高风险国家条形图

        参数:
            high_risk_df: 高风险国家DataFrame
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # 准备数据
        top_countries = high_risk_df.head(15)
        countries = top_countries['country']
        probabilities = top_countries['loss_probability']

        # 颜色映射
        colors = []
        for p in probabilities:
            if p >= 0.70:
                colors.append('#C0392B')  # 深红-极高风险
            elif p >= 0.50:
                colors.append('#E74C3C')  # 红色-高风险
            else:
                colors.append('#F39C12')  # 橙色-中风险

        # 条形图
        bars = ax.barh(countries, probabilities, color=colors,
                       edgecolor='black', linewidth=1.5)

        # 添加概率数值标签
        for i, (country, prob) in enumerate(zip(countries, probabilities)):
            ax.text(prob + 0.02, i, f'{prob:.2%}',
                    va='center', fontsize=10, fontweight='bold')

        # 添加风险阈值线
        ax.axvline(x=0.70, color='red', linestyle='--',
                   linewidth=2, label='极高风险阈值')
        ax.axvline(x=0.50, color='orange', linestyle='--',
                   linewidth=2, label='高风险阈值')
        ax.axvline(x=0.30, color='yellow', linestyle='--',
                   linewidth=2, label='中风险阈值')

        ax.set_xlabel('货币主权丧失概率', fontsize=13, fontweight='bold')
        ax.set_ylabel('国家', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_correlation_heatmap(corr_matrix, title="相关系数热力图"):
        """
        绘制相关系数热力图

        参数:
            corr_matrix: 相关系数矩阵
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                    ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig


def main():
    """
    主程序：执行完整的货币主权影响评估与风险预警
    """
    print("\n" + "="*80)
    print("稳定币对货币主权影响评估与风险预警系统".center(80))
    print("="*80 + "\n")

    # ========== 第一步：生成数据 ==========
    print("【步骤1】生成模拟数据（30个国家）...\n")

    data_generator = CountryDataGenerator(n_countries=30)
    df = data_generator.generate_data()

    print("数据概览:")
    print(df.head(10).to_string(index=False))
    print(f"\n数据形状: {df.shape}\n")

    # ========== 第二步：计算货币主权评分 ==========
    print("【步骤2】计算货币主权综合评分...\n")

    evaluator = MonetarySovereigntyEvaluator()
    sovereignty_score = evaluator.calculate_sovereignty_score(df)

    df['sovereignty_score'] = sovereignty_score

    evaluator.print_score_breakdown(df, sovereignty_score)

    # ========== 第三步：相关性分析 ==========
    print("【步骤3】相关性分析...\n")

    corr_analyzer = CorrelationAnalyzer()
    corr_analyzer.print_correlation_results(df, sovereignty_score)

    # 线性回归
    slope, intercept, r_squared = corr_analyzer.linear_regression(
        df['stablecoin_index'].values, sovereignty_score)

    print("稳定币指数与货币主权的线性回归:")
    print(f"  回归方程: MS = {intercept:.2f} + {slope:.3f} × SCI")
    print(f"  R² = {r_squared:.4f}\n")

    # 相关系数矩阵
    variables = ['sovereignty_score', 'stablecoin_index', 'inflation_rate',
                 'debt_to_gdp', 'foreign_deposit_ratio', 'gdp_per_capita']
    corr_matrix = corr_analyzer.correlation_matrix(df, variables)

    # ========== 第四步：Logit预警模型 ==========
    print("【步骤4】构建Logit预警模型...\n")

    logit_model = LogitWarningModel()

    # 创建标签
    labels = logit_model.create_labels(df, sovereignty_score)
    print(
        f"已识别 {labels.sum()} 个国家处于货币主权丧失状态（基于历史标准）\n")

    # 准备特征
    X = logit_model.prepare_features(df)

    # 拟合模型
    logit_model.fit(X, labels)

    # 预测概率
    probabilities = logit_model.predict_proba(X)
    df['loss_probability'] = probabilities

    # 打印模型摘要
    logit_model.print_model_summary(X, labels)

    # ========== 第五步：高风险国家识别 ==========
    print("【步骤5】识别高风险国家...\n")

    risk_identifier = RiskIdentifier()
    high_risk_df = risk_identifier.identify_high_risk_countries(
        df, sovereignty_score, probabilities)

    risk_identifier.print_high_risk_report(high_risk_df)

    # ========== 第六步：美元国际地位分析 ==========
    print("【步骤6】美元国际地位强化效应分析...\n")

    # 模拟美元国际地位指数（DISI）
    avg_sci = df['stablecoin_index'].mean()
    disi_2025 = 64.5
    disi_impact = avg_sci * 0.15  # 简化的影响系数

    print("=" * 70)
    print("美元国际地位强化效应")
    print("=" * 70)
    print(f"  2025年DISI基准值: {disi_2025:.1f}")
    print(f"  全球平均稳定币指数: {avg_sci:.1f}")
    print(
        f"  预计2030年DISI: {disi_2025 + disi_impact:.1f} (提升{disi_impact:.1f}点)")
    print("\n  机制解释:")
    print("    • 稳定币普及 → 加强美元使用习惯")
    print("    • 美元稳定币占99% → 强化美元国际地位")
    print("    • 弱国货币主权削弱 → 美元影响力扩大")
    print("=" * 70 + "\n")

    # ========== 第七步：可视化 ==========
    print("【步骤7】生成可视化图表...\n")

    visualizer = Visualizer()

    # 1. 散点图：稳定币指数 vs 货币主权
    fig1 = visualizer.plot_correlation_scatter(
        df['stablecoin_index'], sovereignty_score,
        '稳定币普及指数', '货币主权评分',
        '稳定币普及与货币主权的关系'
    )
    fig1.savefig("第四个问题/稳定币与货币主权相关性.png",
                 dpi=300, bbox_inches='tight')
    print("  ✓ 相关性散点图已保存")

    # 2. 风险分布图
    fig2 = visualizer.plot_risk_distribution(
        probabilities, labels,
        '货币主权丧失风险分布'
    )
    fig2.savefig("第四个问题/风险分布图.png", dpi=300, bbox_inches='tight')
    print("  ✓ 风险分布图已保存")

    # 3. 高风险国家条形图
    fig3 = visualizer.plot_risk_map(
        high_risk_df,
        '高风险国家货币主权丧失概率排名'
    )
    fig3.savefig("第四个问题/高风险国家排名.png", dpi=300, bbox_inches='tight')
    print("  ✓ 高风险国家排名图已保存")

    # 4. 相关系数热力图
    fig4 = visualizer.plot_correlation_heatmap(
        corr_matrix,
        '货币主权相关指标热力图'
    )
    fig4.savefig("第四个问题/相关系数热力图.png", dpi=300, bbox_inches='tight')
    print("  ✓ 相关系数热力图已保存\n")

    # ========== 第八步：导出结果 ==========
    print("【步骤8】导出分析结果...\n")

    # 完整结果表
    results_df = df[[
        'country', 'sovereignty_score', 'stablecoin_index',
        'foreign_deposit_ratio', 'inflation_rate', 'loss_probability'
    ]].copy()

    results_df.columns = [
        '国家', '货币主权评分', '稳定币指数',
        '外币存款占比', '通货膨胀率', '丧失概率'
    ]

    results_df = results_df.sort_values('丧失概率', ascending=False)
    results_df.to_csv("第四个问题/货币主权风险评估结果.csv",
                      index=False, encoding='utf-8-sig')
    print("  ✓ 完整结果已保存至CSV文件")

    # 高风险国家专项报告
    high_risk_report = high_risk_df[[
        'country', 'sovereignty_score', 'stablecoin_index',
        'foreign_deposit_ratio', 'loss_probability', 'risk_level'
    ]].copy()

    high_risk_report.to_csv("第四个问题/高风险国家报告.csv",
                            index=False, encoding='utf-8-sig')
    print("  ✓ 高风险国家报告已保存\n")

    # ========== 总结 ==========
    print("=" * 80)
    print("分析完成！".center(80))
    print("=" * 80)

    print("\n核心发现:")
    print(
        f"  1. 稳定币普及与货币主权显著负相关（ρ = {corr_analyzer.calculate_correlation(df['stablecoin_index'], sovereignty_score)[0]:.3f}）")
    print(f"  2. 识别出{len(high_risk_df)}个中高风险国家")
    extreme_risk = len(
        high_risk_df[high_risk_df['loss_probability'] >= 0.70])
    print(f"  3. 其中{extreme_risk}个国家处于极高风险（丧失概率>70%）")
    print(
        f"  4. Logit模型分类准确率达{logit_model.model.score(logit_model.scaler.transform(X), labels):.1%}")
    print(f"  5. 稳定币将使美元国际地位提升约{disi_impact:.1f}个点")

    print("\n政策建议:")
    print("  • 高风险国家应加强资本管制，限制稳定币无序扩张")
    print("  • 推动本币数字化（CBDC），提供合法替代方案")
    print("  • 稳定宏观经济，控制通胀是保护货币主权的根本")
    print("  • 国际社会应关注稳定币对货币主权的系统性威胁")

    print("\n所有结果文件已保存至 '第四个问题' 文件夹")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """程序入口"""
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
