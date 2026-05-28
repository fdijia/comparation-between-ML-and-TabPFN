import json

import pandas as pd
import pickle
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    cohen_kappa_score,
    balanced_accuracy_score,
    average_precision_score
)
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

from tabpfn import TabPFNClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import util, constant

# ---------------------------------
# 将每个列的对应数值编码转化为文字编码
# ---------------------------------
def explain(X, y):
    # 修改选项
    for col in X.columns:
        if col.startswith("第"):
            for i in range(5):
                X[col] = X[col].replace(i + 1, constant.choices[col][i])
    if "性别" in X.columns:
        X["性别"] = X["性别"].replace(0, "女")
        X["性别"] = X["性别"].replace(1, "男")
    # 修改列名
    X.rename(columns = constant.questions, inplace=True)
    y.name = constant.questions[y.name]
    y.replace(0, "无风险")
    y.replace(1, "有风险")

# ----------
# 主训练模型
# ----------
def main(remove_columns=[], y_column="危机-自己", save_name="", dataset="D1", gap=3.5):
    print(f"Processing dataset: {dataset}, target: {y_column}, remove columns: {remove_columns}")
    folds, X, y = util.load_folds_and_split(
        df_path=f"../特征整理与数据构建/output/{dataset}_data.csv",
        folds_path=f"../特征整理与数据构建/output/{dataset}_folds.pkl",
        y_column=y_column,
        remove_columns=remove_columns,
        gap=gap,
        y_columns=y_columns
    )
    if save_name=="best":
        explain(X, y)
    all_results = {}

    # 初始化 TabPFN 模型
    for model_path in models:
        model_name = model_path[:-5]  # 从文件名中提取模型名称
        print(f"Using model: {model_name}")
        print(model_path)
        model = TabPFNClassifier(model_path=f"model/{model_path}")
        metrics = util.train(model, folds, X, y, model_name='pfn')
        all_results[model_name] = metrics
    
    os.makedirs(f"{y_column}_{gap}_output", exist_ok=True)
    with open(f"{y_column}_{gap}_output/{dataset}_{y_column}_{save_name}_{gap}_results.json", "w", encoding="utf-8") as f:
        json.dump(util.prepare_for_json(all_results), f, indent=4)


if __name__ == "__main__":
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    gaps = [3.5, 3.5, 3.75]
    myDatasets = ["D1", "D2_1", "D2_2", "D2_3", "D2_4", "D3", "D4"]
    # # # model 目录下的 *classify* 文件
    models = os.listdir("model")
    new_x = constant.new_x
    for i in range(3):
        y_column = y_columns[i]
        gap = gaps[i]
        # main(remove_columns=[], y_column=y_column, save_name="best", dataset="D1", gap=gap)
        for j in range(len(myDatasets)):
            d = myDatasets[j]
            if not d.startswith("D2"):
                continue
            if j != 0 and j != 6:
                main(remove_columns=new_x[j], y_column=y_column, save_name="normal_original", dataset=d, gap=gap)
            main(remove_columns=[], y_column=y_column, save_name="normal", dataset=d, gap=gap)
    # best 表明所有变量和值都带有信息, 比如 "下列哪一个图案更符合你现在的心情？" ["非常不开心", "不开心", "一般般", "开心", "非常开心"]
    # normal 表明所有变量都是按照编号来, 比如 "第一题" 值为 [1, 2, 3, 4, 5]
    # 结果表明 normal 比 best 要好



