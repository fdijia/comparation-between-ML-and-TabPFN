import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 现在可以直接导入
from utils import constant

plt.rcParams["font.family"] = "SimHei"  # 设置中文字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False

need_standardization = ["LogisticRegression", "SVM", "KNN"]

def Logistic_Regression(params):
    base_config = {"max_iter": 5000, "random_state": 42}
    base_config.update(params)
    model = LogisticRegression(**base_config)
    return model

def LDA_Model(params):
    model = LDA(**params)
    return model

def Random_Forest_Model(params):
    base_config = {"random_state": 42}
    base_config.update(params)
    model = RandomForestClassifier(**base_config)
    return model

def XGBoost_Model(params):
    from xgboost import XGBClassifier
    base_config = {"eval_metric": "mlogloss", "random_state": 42}
    base_config.update(params)
    model = XGBClassifier(**base_config)
    return model

def SVM_Model(params):
    base_config = {"probability": True, "random_state": 42}
    base_config.update(params)
    model = SVC(**base_config)
    return model

def KNN_Model(params):
    return KNeighborsClassifier(**params)



def load_best_params(json_path="../机器学习模型/comparation/best_combined_models.json"):
    if not os.path.exists(json_path):
        print(f"警告: 找不到 {json_path}，将使用默认参数。")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 将列表转为字典: {'RandomForest': {params...}, 'XGBoost': {params...}}
    return {item["model_name"]: item["best_combined_params"] for item in data}


def load_folds_and_split(df_path, folds_path, y_column, remove_columns=[], gap=3.5):
    df = pd.read_csv(df_path, encoding='utf-8-sig')
    with open(folds_path, "rb") as f:
        data = pickle.load(f)

    folds = data["folds"]

    if isinstance(folds, dict):
        column_folds = folds[y_column]
    else:
        column_folds = folds
    df[y_column] = df[y_column].apply(lambda x: 1 if x > gap else 0)
    return column_folds, df.drop(columns=remove_columns + y_columns), df[y_column]



def train(X, y, model_type="RandomForest", y_column="危机-自己", model_weight="default"):
    if model_type == "tabpfn":
        from tabpfn import TabPFNClassifier
        model = TabPFNClassifier(model_path=f"../TabPFN/model/{model_weight}.ckpt", device="cuda", fit_mode="fit_with_cache")

    else:
        best_params_dict = load_best_params(f"../机器学习模型/comparation/{y_column}_best_params.json")
        params = best_params_dict.get(model_type, {})
        model = model_funcs[model_type](params)

    model.fit(X, y)
    return model


def shap_analysis_shapiq(folds, X, y, model_type="RandomForest", y_column="危机-自己", save_name="", model_weight=""):
    if os.path.exists(f'output/{y_dict[y_column]}_{save_name}_{model_type}{model_weight}_shap.joblib'):
        sv_explanation = joblib.load(f'output/{y_dict[y_column]}_{save_name}_{model_type}{model_weight}_shap.joblib')
    else:
        is_std = True if model_type in need_standardization else False
        values = []
        data = []
        base_values = []
        for fold_info in folds:
            train_idx = fold_info["train_idx"]
            val_idx = fold_info["val_idx"]
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            if is_std:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
            model = train(X_train, y_train, model_type, y_column, model_weight)
            if model_type in ["RandomForest", "XGBoost"]:
                # tree explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_val)
                if model_type == "XGBoost":
                    explanation = explainer(X_val)
                    joblib.dump(explanation, f'output/{y_dict[y_column]}_{save_name}_{model_type}{model_weight}_shap.joblib')
                    return explanation       

            elif model_type == "tabpfn":
                from tabpfn_extensions.interpretability import (
                    shapiq as tabpfn_shapiq,
                    shapiq_to_shap_explanation,
                )
                explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
                    model=model,
                    data=X_train,
                    index="SV",
                    max_order=1,
                )
                sv_explanation = shapiq_to_shap_explanation(
                    explainer,
                    X_val,
                    budget=256,
                    feature_names=X.columns.tolist()
                )
                joblib.dump(sv_explanation, f'output/{y_dict[y_column]}_{save_name}_{model_type}{model_weight}_shap.joblib')
                return sv_explanation
            else:
                # kernel explainer
                explainer = shap.KernelExplainer(model.predict_proba, X_train)
                shap_values = explainer.shap_values(X_val)
            
            values.append(shap_values[:, :, 1])
            data.append(X_val)
            base_values.append(shap_values[:, 1])
        values = np.concatenate(values, axis=0)
        data = pd.concat(data, axis=0)
        base_values = np.concatenate(base_values, axis=0)
        sv_explanation = shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            feature_names=X.columns.tolist()
        )
        joblib.dump(sv_explanation, f'output/{y_dict[y_column]}_{save_name}_{model_type}{model_weight}_shap.joblib')
    return sv_explanation


def draw_shapiq(results, title="Model", y_column="危机-自己", save_name="", save=True):
    plt.figure(figsize=(3.5, 4.5))
    shap.summary_plot(
        results,
        max_display=12,
        show=False,
        plot_size=None,
        cmap="coolwarm"
    )
    fig = plt.gcf()
    if len(fig.axes) > 1:
        cb = fig.axes[-1]
        cb.set_ylabel("")
        cb.set_yticks([0, 1]) 
        cb.set_yticklabels(["1", "5"], fontname="SimHei", fontsize=12)
    ax = plt.gca()
    plt.ylabel("")
    plt.xlabel("")

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    labels = ax.get_yticklabels()
    if save_name == "D1":
        for label in labels:
            if label.get_text() == "快被压力压垮":  # 这里的文本需要和你的数据中的文本完全匹配
                label.set_fontweight('bold')        # 设置加粗
                label.set_color('red')
            if label.get_text() in ["有问题怨自己", "学习能力自信"]:  # 这里的文本需要和你的数据中的文本完全匹配
                label.set_fontweight('bold')        # 设置加粗
                label.set_color('blue')
            else:
                label.set_fontweight('normal')      # 其他标签正常显示
    plt.tight_layout()

    plt.savefig(
        f"figure/{y_dict[y_column]}_{save_name}_{title}.png",
        dpi=600,
        bbox_inches="tight"
    )
    if save:
        # 推荐同时保存pdf
        plt.savefig(
            f"../../论文正文/latex/fig/{y_dict[y_column]}_{save_name}_{title}.pdf",
            bbox_inches="tight"
        )

    plt.close()


def singal(y_column="危机-自己", model="RandomForest", gap=3.5, model_weight=[], dataset="D1", save=True):
    folds, X, y = load_folds_and_split(
        df_path=f"../特征整理与数据构建/output/{dataset}_data.csv",
        folds_path=f"../特征整理与数据构建/output/{dataset}_folds.pkl",
        y_column=y_column,
        gap=gap
    )
    
    X.rename(columns = constant.question_summary, inplace=True)

    if isinstance(model, str):
        model = [model]
    if isinstance(model_weight, str):
        model_weight = [model_weight]
    # ML
    for ml in model:
        print(f"start shap analysis for {ml}")
        results = shap_analysis_shapiq(folds, X, y, ml, y_column=y_column, save_name=dataset)
        draw_shapiq(results, ml, y_column=y_column, save_name=dataset, save=save)
    # tabpfn
    for tabpfn in model_weight:
        print("start shap analysis for tabpfn")
        tabpfn_results = shap_analysis_shapiq(folds, X, y, "tabpfn", y_column=y_column, save_name=dataset, model_weight=tabpfn)
        draw_shapiq(tabpfn_results, f"TabPFN{tabpfn[-2:]}", y_column=y_column, save_name=dataset, save=save)


def count_shap(dataset="D1"):
    y_columns = {"self-harm": "risk", "depressed": "depression", "nervous": "anxiety"}
    files = os.listdir("output")
    count_dict = {}
    for y in y_columns:
        save_dict = {}
        for file in files:
            if not file.startswith(f"{y}_{dataset}"):
                continue

            shap_values = joblib.load(f"output/{file}")

            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            feature_names = shap_values.feature_names

            sorted_idx = np.argsort(mean_abs_shap)[::-1]
            for rank, idx in enumerate(sorted_idx, start=1):
                feature = feature_names[idx]

                if feature not in save_dict:
                    save_dict[feature] = 0

                save_dict[feature] += rank

        save_dict = dict(
            sorted(save_dict.items(), key=lambda x: x[1], reverse=False)
        )
        count_dict[y_columns[y]] = save_dict
    os.makedirs("sem", exist_ok=True)
    with open("sem/top_values.json", 'w', encoding='utf-8') as f:
        json.dump(count_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    model_funcs = {
        "LogisticRegression": Logistic_Regression,
        "LDA": LDA_Model,
        "RandomForest": Random_Forest_Model,
        "XGBoost": XGBoost_Model,
        "SVM": SVM_Model,
        "KNN": KNN_Model,
    }
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    y_dict = {
        "危机-自己": "self-harm",
        "问题-抑郁": "depressed",
        "问题-焦虑": "nervous"
    }
    gaps = [3.5, 3.5, 3.75]
    os.makedirs("output", exist_ok=True)
    os.makedirs("figure", exist_ok=True)

    for i in range(3):
        y = y_columns[i]
        gap = gaps[i]
        singal(y, model=list(model_funcs.keys()), gap=gap, model_weight=["default-1", "default-2"], dataset="D1", save=False)
    count_shap("D1")