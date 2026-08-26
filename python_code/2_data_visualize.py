# -*- coding:utf-8 -*-

"""
!!!!!!再次运行需进入python_code内,详见运行调试手册!!!!!!
"""

"""
2. 描述统计与可视化模块
输入:cleaned_data.csv
输出:description_stats.txt + 5张分析图表
优化点：
1. 分维度流失率统计（地域、性别、产品数、活跃度）
2. 特征相关性热力图
3. 核心特征箱线图
4. 输出业务洞察结论
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
# 统一输出目录
OUTPUT_DIR = "../churn_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(OUTPUT_DIR + "cleaned_data.csv")
    report = []
    def log(s):
        print(s)
        report.append(s)

    log("=" * 60)
    log("描述性统计与业务洞察报告")
    log("=" * 60)

    # ========== 1. 整体数据概览 ==========
    log("\n1. 整体数据概览")
    log(f"总样本量：{len(df)}")
    log(f"特征数：{df.shape[1] - 1}")
    churn_rate = df["Exited"].mean() * 100
    log(f"整体流失率：{churn_rate:.2f}%")

    # ========== 2. 数值字段描述统计 ==========
    log("\n2. 数值字段描述统计")
    desc = df.describe().round(2)
    log(desc.to_string())

    # ========== 3. 分维度流失率交叉分析 ==========
    log("\n3. 分维度流失率分析")
    
    # 按地域
    geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False) * 100
    log("\n各地域流失率:")
    log(geo_churn.round(2).to_string())

    # 按性别
    gender_churn = df.groupby("Gender")["Exited"].mean().sort_values(ascending=False) * 100
    log("\n各性别流失率:")
    log(gender_churn.round(2).to_string())

    # 按产品数量
    prod_churn = df.groupby("NumOfProducts")["Exited"].mean().sort_values(ascending=False) * 100
    log("\n不同产品持有数流失率:")
    log(prod_churn.round(2).to_string())

    # 按是否活跃
    active_churn = df.groupby("IsActiveMember")["Exited"].mean().sort_values(ascending=False) * 100
    log("\n活跃/非活跃会员流失率：")
    log(active_churn.round(2).to_string())

    # ========== 4. 流失客户画像 ==========
    log("\n4. 流失客户 vs 未流失客户特征均值对比")
    group_mean = df.groupby("Exited")[["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]].mean().round(2)
    group_mean.index = ["未流失", "流失"]
    log(group_mean.to_string())

    # ========== 5. 保存统计文本 ==========
    with open(OUTPUT_DIR + "description_stats.txt", "w", encoding="utf-8") as f:
        for line in report:
            f.write(line + "\n")
    print("\n统计报告已保存:description_stats.txt")

    # ========== 6. 可视化图表 ==========
    
    # 图1：流失标签饼图
    plt.figure(figsize=(6, 6))
    plt.pie([1-churn_rate/100, churn_rate/100], labels=["未流失", "流失"], 
            autopct="%.2f%%", colors=["#52c41a", "#ff4d4f"], startangle=90)
    plt.title("银行客户流失整体分布")
    plt.savefig(OUTPUT_DIR + "churn_pie.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 图2：各地域流失率柱状图
    plt.figure(figsize=(8, 5))
    geo_churn.plot(kind="bar", color="#fa8c16")
    plt.title("不同地域客户流失率对比")
    plt.ylabel("流失率(%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "geo_churn_bar.png", dpi=150)
    plt.close()

    # 图3：年龄分布箱线图（按流失分组）
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Exited", y="Age", palette=["#52c41a", "#ff4d4f"])
    plt.title("流失与未流失客户年龄分布对比")
    plt.xticks([0, 1], ["未流失", "流失"])
    plt.savefig(OUTPUT_DIR + "age_boxplot.png", dpi=150)
    plt.close()

    # 图4：特征相关性热力图
    plt.figure(figsize=(10, 8))
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("特征相关性热力图")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "corr_heatmap.png", dpi=150)
    plt.close()

    # 图5：账户余额分布直方图
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="Balance", hue="Exited", bins=40, kde=True,
                 palette=["#52c41a", "#ff4d4f"])
    plt.title("客户账户余额分布（按流失状态）")
    plt.savefig(OUTPUT_DIR + "balance_hist.png", dpi=150)
    plt.close()

    print("已生成5张分析图表:churn_pie.png、geo_churn_bar.png、age_boxplot.png、corr_heatmap.png、balance_hist.png")
    log("\n业务洞察:")
    log("1. 德国客户流失率显著高于法国和西班牙")
    log("2. 女性客户流失率略高于男性")
    log("3. 持有产品数越少,流失率越高;持有4个产品的客户流失率极高")
    log("4. 非活跃会员流失率约为活跃会员的2倍")
    log("5. 流失客户平均年龄高于未流失客户")
    log("=" * 60)


if __name__ == "__main__":
    main()
