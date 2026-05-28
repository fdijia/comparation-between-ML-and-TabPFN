import json
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util, constant

# -----------------------------
# 模型定义
# -----------------------------
def Logistic_Regression(folds, X, y, params):
    # 基础配置 + 搜寻到的最优参数
    base_config = {"solver": "saga", "max_iter": 5000, "random_state": 42}
    base_config.update(params)
    model = LogisticRegression(**base_config)
    return util.train(model, folds, X, y, "LogisticRegression")

def LDA_Model(folds, X, y, params):
    model = LDA(**params)
    return util.train(model, folds, X, y, "LDA")

def Random_Forest_Model(folds, X, y, params):
    base_config = {"random_state": 42}
    base_config.update(params)
    model = RandomForestClassifier(**base_config)
    return util.train(model, folds, X, y, "RandomForest")

def XGBoost_Model(folds, X, y, params):
    base_config = {"eval_metric": "mlogloss", "random_state": 42}
    base_config.update(params)
    model = XGBClassifier(**base_config)
    return util.train(model, folds, X, y, "XGBoost")

def SVM_Model(folds, X, y, params):
    base_config = {"probability": True, "random_state": 42}
    base_config.update(params)
    model = SVC(**base_config)
    return util.train(model, folds, X, y, "SVM")

def KNN_Model(folds, X, y, params):
    model = KNeighborsClassifier(**params)
    return util.train(model, folds, X, y, "KNN")


# -----------------------------
# 加载模型超参数
# -----------------------------
def load_best_params(json_path="comparation/best_combined_models.json"):
    if not os.path.exists(json_path):
        print(f"警告: 找不到 {json_path}，将使用默认参数。")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 将列表转为字典: {'RandomForest': {params...}, 'XGBoost': {params...}}
    return {item["model_name"]: item["best_combined_params"] for item in data}


# -----------------------------
# 主程序
# -----------------------------
def main(remove_columns=[], y_column="危机-自己", save_name="", dataset="D1", gap=3.5):
    best_params_dict = load_best_params(f"comparation/{y_column}_best_params.json")

    folds, X, y = util.load_folds_and_split(
        df_path=f"../特征整理与数据构建/output/{dataset}_data.csv",
        folds_path=f"../特征整理与数据构建/output/{dataset}_folds.pkl",
        y_column=y_column,
        remove_columns=remove_columns,
        gap=gap,
        y_columns=y_columns
    )

    model_funcs = {
        "LogisticRegression": Logistic_Regression,
        "LDA": LDA_Model,
        "RandomForest": Random_Forest_Model,
        "XGBoost": XGBoost_Model,
        "SVM": SVM_Model,
        "KNN": KNN_Model,
    }

    all_results = {}

    for name, func in model_funcs.items():
        current_model_params = best_params_dict.get(name, {})
        metrics = func(folds, X, y, current_model_params)
        all_results[name] = metrics 
    
    # 保存结果
    output_path = f"{y_column}_{gap}_output"
    os.makedirs(output_path, exist_ok=True)
    file_name = f"{output_path}/{dataset}_{y_column}_{save_name}_{gap}_results.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(util.prepare_for_json(all_results), f, indent=4)


if __name__ == "__main__":
    y_columns = constant.y_columns
    gaps = constant.gaps
    new_x = constant.new_x
    myDatasets = constant.myDatasets
    for i in range(3):
        y_column = y_columns[i]
        gap = gaps[i]
        for j in range(len(myDatasets)):
            d = myDatasets[j]
            if j != 0 and j != 6:
                main(remove_columns=new_x[j], y_column=y_column, save_name="original", dataset=d, gap=gap)
            main(remove_columns=[], y_column=y_column, save_name="", dataset=d, gap=gap)
