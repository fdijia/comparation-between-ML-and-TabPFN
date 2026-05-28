import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import constant


def process_excel_file(input_file=None, gender_column='性别'):
    df = input_file
    # 将性别列转换为0-1值（男=1，女=0）
    if gender_column in df.columns:
        gender_map = {
            '男': 1, 'male': 1, 'M': 1, 'm': 1, '先生': 1, '男生': 1,
            '女': 0, 'female': 0, 'F': 0, 'f': 0, '女士': 0, '女生': 0
        }
        df[gender_column] = df[gender_column].astype(str).str.strip().map(gender_map)
    return df


def make_D4(x=["第31题", "第13题"], gap=3.5):
    """
    找到特定x列的值, 对其取均值并进行筛选<=gap的行; 最后统计剩余行y_column="危机-自己"的分布
    返回筛选后的df
    """
    df = pd.read_csv("output/D1_data.csv", encoding='utf-8-sig')
    # 确保x是列表
    if isinstance(x, str):
        x = [x]
    
    # 1. 找到特定x列的值，计算每行的均值
    # 计算指定列的行均值
    df_copy = df.copy()
    df_copy['_mean_x'] = df_copy[x].mean(axis=1)
    
    # 2. 筛选均值 < gap 的行
    filtered_df = df[df_copy['_mean_x'] <= gap].copy()
    
    # 3. 统计剩余行 y_column="危机-自己" 的分布
    y_column = "危机-自己"
    
    if y_column in filtered_df.columns:
        distribution = filtered_df[y_column].value_counts()
        print(f"筛选后数据量: {len(filtered_df)} 行")
        print(f"\n'{y_column}' 的分布:")
        print(distribution)
        print(f"\n百分比分布:")
        print(filtered_df[y_column].value_counts(normalize=True))
    else:
        print(f"警告: 列 '{y_column}' 不存在于数据框中")
        distribution = None
    
    filtered_df.to_csv("output/D4_data.csv", index=False, encoding='utf-8-sig')
    print(len(filtered_df))



if __name__ == "__main__":
    x_columns = [f"第{i}题" for i in range(1, 37)] + ["性别"]   #第11题为身份辨认, 统计结果为学生; 第29题为首都辨认, 统一为北京
    # 删去第11题与第29题
    x_columns.remove("第11题")
    x_columns.remove("第29题")
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    os.makedirs("output", exist_ok=True)

    # D_1
    df = pd.read_excel('D1.xlsx')
    df = df[x_columns + y_columns].copy()  # 只保留题目列和目标列
    result = process_excel_file(df)
    result.to_csv("output/D1_data.csv", index=False, encoding='utf-8-sig')    # 登录号-36题目-16维度-3因变量

    # D_3
    grades = ["总分_分数"]
    df = pd.read_excel('D3.xlsx')
    df = df[x_columns + y_columns + grades].copy()  # 只保留题目列和目标列
    result = process_excel_file(df)
    result.to_csv("output/D3_data.csv", index=False, encoding='utf-8-sig')    # 登录号-36题目-16维度-3因变量

    # D_2
    datsets_path = [
        "D2_1_nback分析结果.xlsx",
        "D2_2_概率逆转分析结果.xlsx",
        "D2_3_气球模拟试验分析结果.xlsx",
        "D2_4_停止信号任务分析结果.xlsx",
    ]
    new_x = constant.new_x[1:5]
    for path, new_cols in zip(datsets_path, new_x):
        df = pd.read_excel(path)
        df = df[new_cols + x_columns + y_columns].copy()
        df = df.dropna()
        df = process_excel_file(df)
        df.to_csv(f"output/{path[:4]}_data.csv", index=False, encoding='utf-8-sig')

    make_D4()
