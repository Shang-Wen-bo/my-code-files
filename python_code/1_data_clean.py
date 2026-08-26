# -*- coding:utf-8 -*-

"""
!!!!!!再次运行需进入python_code内,详见运行调试手册!!!!!!
"""

"""
1. 数据清洗模块
输入:churn_practice_data.csv
输出:cleaned_data.csv + clean_report.txt
优化点：
1. 重复值检测与删除
2. 业务规则异常值(年龄999、信用分-999)修正
3. IQR统计法检测数值特征极端值,盖帽法(Winsorize)处理
4. 分字段缺失值策略（类别型众数、数值型中位数）
5. 输出清洗审计报告
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
# 统一输出目录
OUTPUT_DIR = "../churn_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def winsorize(series, lower=0.01, upper=0.99):
    """盖帽法：将分位数外的极端值替换为分位数值"""
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)


def main():
    report = []
    def log(s):
        print(s)
        report.append(s)

    log("=" * 60)
    log("数据清洗审计报告")
    log("=" * 60)

    # ========== 1. 读取原始数据 ==========
    df = pd.read_csv("churn_practice_data.csv")
    log(f"\n原始数据规模:{df.shape[0]} 行，{df.shape[1]} 列")
    log(f"原始字段列表：{list(df.columns)}")

    # ========== 2. 重复值处理 ==========
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        log(f"\n删除重复行:{dup_count} 条")
    else:
        log(f"\n无重复数据")

    # ========== 3. 删除无建模意义字段 ==========
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=drop_cols)
    log(f"\n删除ID类字段:{drop_cols}")
    log(f"剩余字段数：{df.shape[1]}")

    # ========== 4. 缺失值处理 ==========
    log("\n---------- 缺失值处理 ----------")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    log(f"处理前缺失值统计：\n{missing.to_string()}")

    # 类别型字段：Geography 用众数填充
    if df["Geography"].isnull().sum() > 0:
        geo_mode = df["Geography"].mode()[0]
        df["Geography"] = df["Geography"].fillna(geo_mode)
        log(f"Geography 缺失值 {df['Geography'].isnull().sum()} 个，用众数 {geo_mode} 填充")

    # 数值型字段：EstimatedSalary 用中位数填充
    if df["EstimatedSalary"].isnull().sum() > 0:
        salary_median = df["EstimatedSalary"].median()
        df["EstimatedSalary"] = df["EstimatedSalary"].fillna(salary_median)
        log(f"EstimatedSalary 缺失值 {df['EstimatedSalary'].isnull().sum()} 个，用中位数 {salary_median:.2f} 填充")

    log(f"处理后缺失值总数：{df.isnull().sum().sum()}")

    # ========== 5. 异常值处理（双层校验） ==========
    log("\n---------- 异常值处理 ----------")

    # 5.1 业务规则异常：明显录入错误
    # 年龄异常值999
    age_bad = (df["Age"] == 999).sum()
    if age_bad > 0:
        age_normal_median = df.loc[df["Age"] != 999, "Age"].median()
        df.loc[df["Age"] == 999, "Age"] = age_normal_median
        log(f"年龄录入异常值999共 {age_bad} 个，替换为正常中位数 {age_normal_median}")

    # 信用分异常值-999
    cs_bad = (df["CreditScore"] == -999).sum()
    if cs_bad > 0:
        cs_normal_median = df.loc[df["CreditScore"] != -999, "CreditScore"].median()
        df.loc[df["CreditScore"] == -999, "CreditScore"] = cs_normal_median
        log(f"信用分录入异常值-999共 {cs_bad} 个，替换为正常中位数 {cs_normal_median}")

    # 5.2 统计极端值：IQR 1%-99% 盖帽处理
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    log("\n数值字段极端值盖帽处理(1%~99%分位):")
    for col in num_cols:
        before_max = df[col].max()
        before_min = df[col].min()
        df[col] = winsorize(df[col])
        after_max = df[col].max()
        after_min = df[col].min()
        log(f"  {col}: 范围 [{before_min:.2f}, {before_max:.2f}] → [{after_min:.2f}, {after_max:.2f}]")

    # ========== 6. 数据类型校验 ==========
    log("\n---------- 数据类型校验 ----------")
    log(df.dtypes.to_string())

    # ========== 7. 保存结果 ==========
    df.to_csv(OUTPUT_DIR + "cleaned_data.csv", index=False, encoding="utf-8-sig")
    log(f"\n清洗后数据已保存:cleaned_data.csv")
    log(f"最终数据规模：{df.shape[0]} 行，{df.shape[1]} 列")

    # 保存审计报告
    with open(OUTPUT_DIR + "clean_report.txt", "w", encoding="utf-8") as f:
        for line in report:
            f.write(line + "\n")
    log("清洗审计报告已保存:clean_report.txt")
    log("=" * 60)


if __name__ == "__main__":
    main()
