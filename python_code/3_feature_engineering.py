# -*- coding:utf-8 -*-

"""
!!!!!!再次运行需进入python_code内,详见运行调试手册!!!!!!
"""

"""
3. 特征工程模块
输入:cleaned_data.csv
输出：训练集/测试集文件 + 特征名称列表
优化点：
1. 业务衍生特征（年龄分段、信用等级、余额薪资比等）
2. 类别特征独热编码
3. 训练集SMOTE过采样,解决类别不平衡
4. 特征标准化
5. 分层抽样划分数据集
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import os
# 统一输出目录
OUTPUT_DIR = "../churn_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(OUTPUT_DIR + "cleaned_data.csv")
    print("=" * 60)
    print("特征工程开始")
    print("=" * 60)

    # ========== 1. 特征衍生 ==========
    print("\n1. 业务特征衍生")
    
    # 年龄分段
    df["Age_Group"] = pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 100], 
                            labels=["青年", "中青年", "中年", "中老年", "老年"])
    
    # 信用评分等级
    df["Credit_Level"] = pd.cut(df["CreditScore"], bins=[0, 500, 600, 700, 800, 1000],
                                labels=["差", "一般", "良好", "优秀", "极好"])
    
    # 余额是否为0
    df["Balance_Zero"] = (df["Balance"] == 0).astype(int)
    
    # 余额薪资比
    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    
    # 产品数量相对值
    df["Product_Tenure_Ratio"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    
    print("新增特征:Age_Group、Credit_Level、Balance_Zero、Balance_Salary_Ratio、Product_Tenure_Ratio")

    # ========== 2. 类别特征编码 ==========
    print("\n2. 类别特征独热编码")
    cat_cols = ["Geography", "Gender", "Age_Group", "Credit_Level"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"编码后特征总数：{df.shape[1] - 1}")

    # ========== 3. 划分训练集与测试集 ==========
    print("\n3. 数据集划分(7:3分层抽样)")
    y = df["Exited"]
    X = df.drop(columns=["Exited"])
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集：{X_train.shape[0]} 样本，测试集：{X_test.shape[0]} 样本")
    print(f"训练集流失率：{y_train.mean():.2%}，测试集流失率：{y_test.mean():.2%}")

    # ========== 4. SMOTE过采样（仅训练集） ==========
    print("\n4. 训练集SMOTE过采样(处理类别不平衡)")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"采样前训练集：{len(y_train)}，流失样本 {y_train.sum()} 个")
    print(f"采样后训练集：{len(y_train_res)}，流失样本 {y_train_res.sum()} 个")
    print(f"采样后训练集流失率：{y_train_res.mean():.2%}")

    # ========== 5. 特征标准化 ==========
    print("\n5. 特征标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # 转回DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # ========== 6. 保存结果 ==========
    X_train_df.to_csv(OUTPUT_DIR + "X_train.csv", index=False, encoding="utf-8-sig")
    X_test_df.to_csv(OUTPUT_DIR + "X_test.csv", index=False, encoding="utf-8-sig")
    y_train_res.to_csv(OUTPUT_DIR + "y_train.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(OUTPUT_DIR + "y_test.csv", index=False, encoding="utf-8-sig")
    
    with open(OUTPUT_DIR + "feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    print("\n所有数据文件已保存")
    print("=" * 60)


if __name__ == "__main__":
    main()
