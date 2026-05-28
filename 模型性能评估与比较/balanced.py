from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
)
import json
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import util

model_classes = {
    "BalancedLR": LogisticRegression,
    "BalancedRF": BalancedRandomForestClassifier,
    "BalancedXGB": XGBClassifier,
    "BalancedSVM": SVC,
    "EasyEnsemble": EasyEnsembleClassifier,
}


def run_dimension_wise_search(X, y, folds, config_path="model_params.json"):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    final_comparison_results = []
    
    for model_name, dimensions in config.items():
        print(f"\n🚀 正在为模型 [{model_name}] 寻找各维度最优参数...")
        best_combined_params = load_best_param(model_name)

        for dim_name, candidates in dimensions.items():
            print(f"  🔍 维度测试: {dim_name} -> {candidates}")
            dim_best_val = None
            dim_best_score = -1
            for val in candidates:
                current_test_params = copy.deepcopy(best_combined_params)
                current_test_params[dim_name] = val
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
            # =================================================
            # 更新最佳参数
            # =================================================
            if dim_best_val is not None:
                print(f"  ✅ 维度 [{dim_name}] 最优选: {dim_best_val} (PR-AUC: {dim_best_score:.4f})")
                best_combined_params[dim_name] = dim_best_val
        print(f"  🏆 最终组合参数: {best_combined_params}")
        final_comparison_results.append({
            "model_name": model_name,
            "best_combined_params": best_combined_params,
        })

    output_path = os.path.join(output_dir, f"{y_column}_best_params.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_comparison_results, f, indent=4, ensure_ascii=False)
    print(f"\n✨ 任务完成！结果已保存至: {output_path}")

def load_best_param(model_name, configs=None):
    if model_name == "BalancedLR":
        best_combined_params = {
            "solver": "saga",
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
    elif model_name == "BalancedSVM":
        best_combined_params = {
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42
        }
    elif model_name == "BalancedRF":
        best_combined_params = {
            "sampling_strategy": "auto",
            "replacement": False,
            "random_state": 42,
            "n_jobs": -1
        }
    elif model_name == "BalancedXGB":
        pos = 24
        neg = 248
        best_combined_params = {
            "eval_metric": "logloss",
            "scale_pos_weight": neg / pos,
            "random_state": 42,
            "n_jobs": -1
        }
    elif model_name == "EasyEnsemble":
        best_combined_params = {
            "sampling_strategy": "auto",
            "random_state": 42,
            "n_jobs": -1
        }
    if configs != None:
        best_combined_params.update(configs.get("best_combined_params", {}))
    return best_combined_params

def get_model(model_name, config_path="output/危机-自己_best_params.json"):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    for c in configs:
        if c["model_name"] == model_name:
            best_para = load_best_param(model_name, c)
            return model_classes[model_name](**best_para)
    exit(1)

def train_best_model(folds, X, y):
    all_results = {}
    for model_name in model_classes.keys():
        model = get_model(model_name)
        results = util.train(model, folds, X, y, model_name)
        all_results[model_name] = results
    save_path = "危机-自己_3.5_output/D4_best.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(util.prepare_for_json(all_results), f, indent=2)
    return all_results

def main():
    folds, X, y = util.load_folds_and_split(
        df_path=f"../特征整理与数据构建/output/{dataset}_data.csv",
        folds_path=f"../特征整理与数据构建/output/{dataset}_folds.pkl",
        y_column=y_column,
        remove_columns=[],
        gap=gap
    )
    if not os.path.exists("output/危机-自己_best_params.json"):
        run_dimension_wise_search(X, y, folds)
    results = train_best_model(folds, X, y)
    latex_lines = []
    for model, result in results.items():
        overall = result['overall']
        row = [model]
        row.append(f"{overall['PR-AUC']:.3f}")
        row.append(f"{overall['ROC-AUC']:.3f}")
        row.append(f"{overall['Precision']:.3f}")
        row.append(f"{overall['Recall']:.3f}")
        row.append(f"{overall['F1']:.3f}")
        line = '&'.join(row) + "\\\\"
        latex_lines.append(line)
    with open("output/D4_comparison_best.txt", "w", encoding="utf-8") as f:
        for line in latex_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    dataset = "D4"
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    y_column = y_columns[0]
    gap = 3.5
    main()