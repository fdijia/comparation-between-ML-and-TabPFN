import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
import seaborn as sns
from tqdm import tqdm
import os


def load_folds_and_split(df_path, y_column, gap=3.5):
    df = pd.read_csv(df_path, encoding='utf-8-sig')

    df[y_column] = df[y_column].apply(lambda x: 1 if x > gap else 0)
    return df[y_column]


def boostrap_compare(stores_res, base_model_name=None, results=None, y_true=None, metric_func=None, label="oof_pred_proba", n_bootstrap=2000, alpha=0.95):
    """
    store_res: 最终保存json文件中的newtarget对应的字典, 比如final_res["ROC-AUC (95% CI)"]
    """
    base_model = results[base_model_name]
    y_base_model = np.array(base_model[label])
    n = len(y_base_model)

    compare_scores = {}
    for _ in results.keys():
        compare_scores[_] = []

    y_true = np.array(y_true)
    print(n)
    for i in tqdm(range(n_bootstrap)):
        indices = np.random.choice(n, n, replace=True)

        while len(np.unique(y_true[indices])) < 2:
            indices = np.random.choice(n, n, replace=True)

        base_score = metric_func(y_true[indices], y_base_model[indices])
        compare_scores[base_model_name].append(base_score)
        for model_name in results:
            if model_name == base_model_name:
                continue
            new_model = results[model_name]
            y_new_model = np.array(new_model[label])
            new_score = metric_func(y_true[indices], y_new_model[indices])
            compare_scores[model_name].append(base_score - new_score)
        
    for model_name in results.keys():
        lower = np.percentile(compare_scores[model_name], (1 - alpha) / 2 * 100)
        upper = np.percentile(compare_scores[model_name], (1 + alpha) / 2 * 100)
        stores_res[model_name] = (lower, upper)
        stores_res[model_name+"_compare_scores"] = compare_scores[model_name]



def load_results(file_path):
    with open(file_path, "r") as f:
        results = json.load(f)
    return results


def compare_results(y_column="危机-自己", targets=["OOF_ROC-AUC"], ml_name="", tabpfn_name="best", alpha=0.95, dataset="D1", n_boostrap=2000, gap=3.0):
    tabpfn_results = load_results(f"../TabPFN/{y_column}_{gap}_output/{dataset}_{y_column}_{tabpfn_name}_{gap}_results.json")
    model_results = load_results(f"../机器学习模型/{y_column}_{gap}_output/{dataset}_{y_column}_{ml_name}_{gap}_results.json")
    merge_results = tabpfn_results | model_results
    comparison = {}

    for target in targets:
        comparison[target] = {}
        for model_name in merge_results:
            comparison[target][model_name] = merge_results[model_name]["overall"].get(target, None)

    # target = "ROC-AUC (95% CI)", "PR-AUC (95% CI)", 以 tabpfn 为基准，比较其他模型的 ROC-AUC 是否显著提升
    new_targets = [f"ROC-AUC ({alpha * 100:.0f}% CI)", f"PR-AUC ({alpha * 100:.0f}% CI)"]
    new_targets = [f"PR-AUC ({alpha * 100:.0f}% CI)"]
    y_true = load_folds_and_split(
        df_path=f"../特征整理与数据构建/output/{dataset}_data.csv",
        y_column=y_column, gap=gap
    )
    for new_target in new_targets:
        if n_boostrap != 0:
            comparison[new_target] = {}
            if new_target.startswith("ROC-AUC ("):
                metric_func = roc_auc_score
                label = "oof_pred_proba"
                target = "ROC-AUC"
            elif new_target.startswith("PR-AUC ("):
                metric_func = average_precision_score
                label = "oof_pred_proba"
                target = "PR-AUC"
    
        # 使用tabpfn模型中["default", "default-2". "default-3"]中ROC-AUC最高的一个作为基准模型，比较其他模型的ROC-AUC是否显著提升
        tabpfn_roc_auc = []
        for model_name in tabpfn_model_names:
            tabpfn_roc_auc.append((model_name, merge_results[model_name]["overall"].get(target, 0)))

        base_model_name = max(tabpfn_roc_auc, key=lambda x: x[1])[0]
        if n_boostrap != 0:
            boostrap_compare(comparison[new_target], base_model_name=base_model_name, results=merge_results, y_true=y_true, metric_func=metric_func, label=label, alpha=alpha, n_bootstrap=n_boostrap)
    os.makedirs(f"{y_column}_{gap}_output", exist_ok=True)
    with open(f"{y_column}_{gap}_output/{dataset}_{y_column}_{tabpfn_name}_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)
    return f"{y_column}_{gap}_output/{dataset}_{y_column}_{tabpfn_name}_comparison_results.json"


def plot(comparison_path, alpha=0.95, name="", gap=3.5):
    """draw a tarbular to compare the results"""
    comparison = load_results(comparison_path)
    y_column = comparison_path.split("output/")[-1].split("_comparison_results.json")[0]
    targets = list(comparison.keys())
    model_names = list(comparison[targets[0]].keys())

    # 画表格
    # model | target1 | target2 | ...
    # -----|---------|---------|---
    # model1| value   | value   | ...
    # model2| value   | value   | ...
    fig, ax = plt.subplots(figsize=(14, len(model_names) * 0.5 + 1))
    ax.axis("off")
    table_data = [["Model"] + targets]
    for model_name in model_names:
        row = [model_name]
        for target in targets:
            if isinstance(comparison[target][model_name], float):
                row.append(f"{round(comparison[target][model_name], 3)}")
            elif isinstance(comparison[target][model_name], list) and len(comparison[target][model_name]) == 2:
                row.append(f"{round(comparison[target][model_name][0], 3)} ~ {round(comparison[target][model_name][1], 3)}")
            else:
                row.append(str(comparison[target][model_name]))
                # pass
        table_data.append(row)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    os.makedirs(f"{name}_{gap}_figure", exist_ok=True)
    plt.savefig(f"{name}_{gap}_figure/{y_column}_comparison_plot_{alpha * 100:.0f}%.png")
    plt.close()



def process_D1(json_file_path, output_txt_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    name_dict = {
        "default-1": "PFN1",
        "default-2": "PFN2",
        "default-3": "PFN3",
        "LogisticRegression": "LR",
        "LDA": "LDA",
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "KNN": "KNN",
    }
    
    # 提取各个指标的值
    pr_auc = data.get("PR-AUC", {})
    pr_auc_ci = data.get("PR-AUC (95% CI)", {})
    roc_auc = data.get("ROC-AUC", {})
    f1 = data.get("F1", {})
    recall = data.get("Recall", {})
    
    def format_number(value, should_bold=False):
        """格式化数字，支持加粗"""
        if isinstance(value, (int, float)):
            formatted = f"{value:.3f}"
            if should_bold:
                return f"\\textbf{{{formatted}}}"
            return formatted
        return str(value)
    
    def format_ci(ci_value):
        """格式化置信区间"""
        if isinstance(ci_value, (list, tuple)) and len(ci_value) >= 2:
            lower = ci_value[0]
            upper = ci_value[1]
            # 处理上下界，保留3位小数，如果是0.000则显示0.0
            lower_str = f"{lower:.3f}"
            upper_str = f"{upper:.3f}"
            return f"({lower_str}, {upper_str})"
        return "(0.000, 0.000)"
    
    # 收集所有模型的数据
    all_models_data = []
    for model_name, model_display in name_dict.items():
        if model_name in pr_auc:
            all_models_data.append({
                'name': model_name,
                'display': model_display,
                'recall': recall.get(model_name, 0.0),
                'pr_auc': pr_auc.get(model_name, 0.0),
                'pr_auc_ci': pr_auc_ci.get(model_name, [0.0, 0.0]),
            })
    
    # 找出每个指标的最佳值（根据指标性质，有些需要最大值，有些需要最小值）
    # 对于这些分类指标，通常越大越好
    best_recall = max(data['recall'] for data in all_models_data)
    best_pr_auc = max(data['pr_auc'] for data in all_models_data)
    
    # 准备输出数据，标记需要加粗的指标
    output_data = []
    for model_data in all_models_data:
        output_data.append({
            'display': model_data['display'],
            'recall': format_number(model_data['recall'], model_data['recall'] == best_recall),
            'pr_auc': format_number(model_data['pr_auc'], model_data['pr_auc'] == best_pr_auc),
            'ci': format_ci(model_data['pr_auc_ci']),
        })
    
    # 写入TXT文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for res in output_data:
            # 格式：model&recall&pr-auc&(ci)&f1&roc-auc&recall&pr-auc
            line = (f"\t\t{res['display']}&{res['recall']}&{res['pr_auc']}&{res['ci']}\\\\")
            f.write(line + "\n")
    
    print(f"结果已保存到 {output_txt_path}")
        

def process_D2():
    os.makedirs("output", exist_ok=True)
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    gaps = [3.5, 3.5, 3.75]
    datasets = ["D2_1", "D2_2", "D2_3", "D2_4"]
    metric_name = "PR-AUC"
    before_name = "normal_original"
    after_name = "normal"

    def load_metric_result(y_column, gap, dataset, save_name, metric_name):
        file_name = f"{y_column}_{gap}_output/{dataset}_{y_column}_{save_name}_comparison_results.json"
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[metric_name]

    model_names = list(name_dict.keys())

    # ==============================
    # 生成 LaTeX 表格行
    # ==============================
    all_boosts_str = [] # 3 * 9 * 4, 用于生成latex综合提升的表格文本
    all_boost_num = []  # 提升幅度, 用于画热力图
    for i in range(3):
        y_column = y_columns[i]
        gap = gaps[i]
        latex_lines = []
        boost_str = []
        boost_num = []
        for model in model_names:
            b_str = []
            b_num = []
            display_name = name_dict.get(model, model)
            row = [display_name]
            for dataset in datasets:
                before_result = load_metric_result(y_column, gap, dataset, before_name, metric_name)
                after_result = load_metric_result(y_column, gap, dataset, after_name, metric_name)

                before_score = before_result[model]
                after_score = after_result[model]
                # 提升百分比
                improvement = ((after_score - before_score) / before_score * 100)
                if improvement > 0:
                    text = f"{before_score:.3f} $\\to$ {after_score:.3f} (\\textbf{{{improvement:+.2f}}}\\%)"
                    b_str.append(f"\\textbf{{{improvement:+.2f}}}".replace('+', ''))
                else: 
                    text = f"{before_score:.3f} $\\to$ {after_score:.3f} ({improvement:+.2f}\\%)"
                    b_str.append(f"{improvement:+.2f}")
                row.append(text)
                b_num.append(improvement)

            # 用 & 拼接
            latex_line = "&".join(row) + r"\\"
            latex_lines.append(latex_line)
            boost_str.append(b_str)
            boost_num.append(b_num)
            if model == "default-3":
                latex_lines.append("\\midrule")
        all_boosts_str.append(boost_str)
        all_boost_num.append(boost_num)
        save_path = f"output/D2_{y_column}_comparison.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            for line in latex_lines:
                f.write(line + "\n")
        print(f"结果已保存至: {save_path}")

    latex_lines = []
    matrix = []
    for j in range(len(model_names)):
        model = model_names[j]
        display_name = name_dict.get(model, model)
        row = [display_name]
        m = []
        for k in range(4):
            for i in range(3):
                row.append(all_boosts_str[i][j][k])
                m.append(all_boost_num[i][j][k])
        latex_line = "&".join(row) + r"\\"
        latex_lines.append(latex_line)
        matrix.append(m)
    save_path = f"output/D2_boost_comparison.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for line in latex_lines:
            f.write(line + "\n")
    print(f"结果已保存至: {save_path}")

    # 依据m画热力图
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘图设置
    plt.figure(figsize=(6, 3.5)) # 预测因子多，建议增加高度
    sns.set_theme(style="white", font='SimHei', font_scale=1.0) # 确保中文显示

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    xtick_labels = ['S', 'D', 'N', 'S', 'D', 'N', 'S', 'D', 'N', 'S', 'D', 'N']
    ytick_labels = ["PFN1", "PFN2", "PFN3", "LR", "LDA", "RF", "XGB", "SVM", "KNN"]
    ax = sns.heatmap(matrix, 
                cmap=cmap, 
                xticklabels=xtick_labels, 
                yticklabels=ytick_labels, 
                vmax=5.0, center=0.0, vmin=-5.0, 
                annot=True, fmt='.1f',
                annot_kws={"size": 8, "color": "black"},
                square=False, # 长方形矩阵设为 False 比例更协调
                linewidths=.3, 
                linecolor="white",
                cbar_kws={"shrink": .85, "label": "增幅(%)"})
    split_indices = [3, 6, 9, 12] 
    group_names = ["情绪N-back", "概率逆转", "气球模拟", "停止信号"] # 对应你的四个任务
    
    # 获取起点坐标
    starts = [0] + split_indices[:-1]
    ends = split_indices
    
    ymin, ymax = ax.get_ylim()
    
    for i, (start, end) in enumerate(zip(starts, ends)):
        mid_point = (start + end) / 2
        # y 轴坐标设为 -0.2 (稍微在矩阵上方)，transform=ax.get_xaxis_transform() 确保位置对齐
        ax.text(mid_point, -0.2, group_names[i], 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
        
    # 获取 y 轴范围（确保线贯穿上下）
    ymin, ymax = ax.get_ylim()
    for i, x in enumerate(split_indices[:-1]):
        ax.vlines(
            x=x,
            ymin=ymin,
            ymax=ymax,
            colors="black",
            linewidth=1.5,
            linestyles="--",
            alpha=0.8
        )
    
    plt.xticks(ha='center', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel('')
    plt.tight_layout() # 防止坐标轴标签被遮挡
    plt.savefig('output/heatmap_D2.png', dpi=600, bbox_inches='tight')
    plt.savefig('../../论文正文/latex/fig/heatmap_D2.pdf', bbox_inches='tight')
    plt.close()


def process_D3():
    os.makedirs("output", exist_ok=True)
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    gaps = [3.5, 3.5, 3.75]
    dataset = "D3"
    metric_name = "PR-AUC"

    before_name = "normal_original"
    after_name = "normal"

    def load_metric_result(y_column, gap, dataset, save_name, metric_name):
        file_name = f"{y_column}_{gap}_output/{dataset}_{y_column}_{save_name}_comparison_results.json"
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[metric_name]

    model_names = list(name_dict.keys())
    # ==============================
    # 生成 LaTeX 表格行
    # ==============================
    latex_lines = []
    for model in model_names:
        display_name = name_dict.get(model, model)
        row = [display_name]
        for i in range(3):
            y_column = y_columns[i]
            gap = gaps[i]

            before_result = load_metric_result(y_column, gap, dataset, before_name, metric_name)
            after_result = load_metric_result(y_column, gap, dataset, after_name, metric_name)

            before_score = before_result[model]
            after_score = after_result[model]
            # 提升百分比
            improvement = ((after_score - before_score) / before_score * 100)
            if improvement > 0:
                text = f"{before_score:.3f}$\\to${after_score:.3f} (\\textbf{{{improvement:+.2f}}}\\%)"
            else: 
                text = f"{before_score:.3f}$\\to${after_score:.3f} ({improvement:+.2f}\\%)"
            row.append(text)
        # 用 & 拼接
        latex_line = "&".join(row) + r"\\"
        latex_lines.append(latex_line)
        if model == "default-3":
            latex_lines.append("\\midrule")
    save_path = f"output/D3_comparison.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for line in latex_lines:
            f.write(line + "\n")
    print(f"结果已保存至: {save_path}")
    
def process_D4():
    ml_name = ""
    tabpfn_name = "normal"
    tabpfn_results = load_results(f"../TabPFN/危机-自己_3.5_output/D4_危机-自己_{tabpfn_name}_3.5_results.json")
    model_results = load_results(f"../机器学习模型/危机-自己_3.5_output/D4_危机-自己_{ml_name}_3.5_results.json")
    merge_results = tabpfn_results | model_results

    latex_lines = []
    for model in merge_results.keys():
        if model not in name_dict:
            continue
        row = [name_dict[model]]
        row.append(f"{merge_results[model]['overall']['pr-auc']:.3f}")
        row.append(f"{merge_results[model]['overall']['roc-auc']:.3f}")
        row.append(f"{merge_results[model]['overall']['Precision']:.3f}")
        row.append(f"{merge_results[model]['overall']['Recall']:.3f}")
        row.append(f"{merge_results[model]['overall']['F1']:.3f}")
        line = '&'.join(row) + "\\\\"
        latex_lines.append(line)
    with open("output/D4_comparison.txt", "w", encoding="utf-8") as f:
        for line in latex_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    name_dict = {
        "default-1": "PFN1",
        "default-2": "PFN2",
        "default-3": "PFN3",
        "LogisticRegression": "LR",
        "LDA": "LDA",
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
        "KNN": "KNN",
    }
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    datasets = ["D1", "D2_1", "D2_2", "D2_3", "D2_4", "D3", "D4"]
    tabpfn_model_names = ["default-1", "default-2", "default-3"]
    gaps = [3.5, 3.5, 3.75]
    alpha = 0.95
    targets = ["PR-AUC", "ROC-AUC", "F1", "Recall",]
    n_boostrap = 10000
    for i in range(3):
        y_column = y_columns[i]
        gap = gaps[i]
        for d in datasets:
            if not d.startswith("D2"):
                continue
            if d != "D1":
                results_path = compare_results(y_column, targets, alpha=alpha, dataset=d, tabpfn_name="normal", n_boostrap=0, gap=gap)
            else:
                results_path = compare_results(y_column, targets, alpha=alpha, dataset=d, tabpfn_name="normal", n_boostrap=n_boostrap, gap=gap)
            plot(results_path, alpha=alpha, name=y_column, gap=gap)
            if d == "D4":
                continue
            if d != "D1":
                results_path = compare_results(y_column, targets, alpha=alpha, dataset=d, tabpfn_name="normal_original", ml_name="original", n_boostrap=0, gap=gap)
            else:
                results_path = compare_results(y_column, targets, alpha=alpha, dataset=d, tabpfn_name="best", n_boostrap=0, gap=gap)
            plot(results_path, alpha=alpha, name=y_column, gap=gap)

    for i in range(3):
        y_column = y_columns[i]
        gap = gaps[i]
        for file in os.listdir(f"{y_column}_{gap}_output"):
            if file.endswith(".json") and file.startswith("D1"):
                process_D1(f"{y_column}_{gap}_output/{file}", f"{y_column}_{gap}_output/{file[:-5]}.txt")
    process_D2()
    process_D3()
    process_D4()
