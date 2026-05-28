import json
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util



def run_dimension_wise_search(X, y, folds, config_path="model_params.json", y_column="危机-自己",):
    output_dir = "comparation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(config_path, "r") as f:
        config = json.load(f)

    model_classes = {
        "LogisticRegression": LogisticRegression,
        "LDA": LDA,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "SVM": SVC,
        "KNN": KNeighborsClassifier
    }

    final_comparison_results = []

    for model_name, dimensions in config.items():
        print(f"\n🚀 正在为模型 [{model_name}] 寻找各维度最优参数...")
        best_combined_params = {}
        if model_name == "LogisticRegression":
            best_combined_params = {"solver": "saga", "max_iter": 5000, "random_state": 42}
        elif model_name == "SVM":
            best_combined_params = {"probability": True, "random_state": 42}
        elif model_name == "XGBoost":
            best_combined_params = {"eval_metric": "mlogloss", "random_state": 42}
        elif model_name not in ["LDA", "KNN"]:
            best_combined_params = {"random_state": 42}

        # 2. 逐个维度进行测试
        for dim_name, candidates in dimensions.items():
            print(f"  🔍 维度测试: {dim_name} -> {candidates}")
            dim_best_val = None
            dim_best_score = -1
            for val in candidates:
                current_test_params = copy.deepcopy(best_combined_params)
                current_test_params[dim_name] = val
                if model_name == "LDA":
                    if dim_name == "shrinkage" and val is not None:
                        current_test_params["solver"] = "lsqr"
                    if dim_name == "solver" and val == "svd":
                        current_test_params["shrinkage"] = None
                    if current_test_params.get("solver") == "svd":
                        current_test_params["shrinkage"] = None
                try:
                    model_inst = model_classes[model_name](**current_test_params)
                    metrics = util.only_train_X(model_inst, folds, X, y, model_name)
                    score = metrics["PR-AUC"]
                    
                    if score > dim_best_score:
                        dim_best_score = score
                        dim_best_val = val
                except Exception as e:
                    print(f"⚠️ 跳过非法组合 {dim_name}={val}: {e}")
                    continue

            # 更新该维度的“冠军”
            if dim_best_val is not None:
                print(f"  ✅ 维度 [{dim_name}] 最优选: {dim_best_val} (PR-AUC: {dim_best_score:.4f})")
                best_combined_params[dim_name] = dim_best_val
                
                # 同步更新 LDA 的 solver 状态，防止后续维度冲突
                if model_name == "LDA" and dim_name == "shrinkage" and dim_best_val is not None:
                    best_combined_params["solver"] = "lsqr"
            else:
                print(f"  ❌ 维度 [{dim_name}] 未能找到有效参数，保持原样。")

        # 3. 最终组合参数跑一次
        # 再次确保 LDA 最终参数不冲突
        if model_name == "LDA" and best_combined_params.get("solver") == "svd":
            best_combined_params["shrinkage"] = None

        print(f"  🏆 最终组合参数: {best_combined_params}")
        final_comparison_results.append({
            "model_name": model_name,
            "best_combined_params": best_combined_params,
        })

    # 4. 保存结果
    output_path = os.path.join(output_dir, f"{y_column}_best_params.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_comparison_results, f, indent=4, ensure_ascii=False)

    print(f"\n✨ 任务完成！结果已保存至: {output_path}")


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    gap = 3.5
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    dataset = "D1"
    for y_column in y_columns:
        if y_column != "问题-抑郁":
            continue
        folds, X, y = util.load_folds_and_split(
            df_path=f"../特征整理与数据构建/output/{dataset}_data.csv",
            folds_path=f"../特征整理与数据构建/output/{dataset}_folds.pkl",
            y_column=y_column,
            remove_columns=[]
        )
        # 2. 运行搜索
        run_dimension_wise_search(X, y, folds, "model_params.json", y_column)