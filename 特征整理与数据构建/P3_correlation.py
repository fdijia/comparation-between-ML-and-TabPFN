import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import json
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from copy import deepcopy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import constant

def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

def describe_dependent_variables(save_name="D1_description"):
    df = pd.read_csv("output/D1_data.csv", encoding='utf-8-sig')
    # ===== 三个变量 =====
    cols = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    desc_stats = df[cols].describe().T  # 转置矩阵，让变量作为行，指标作为列
    desc_stats['variance'] = df[cols].var()  # 方差
    desc_stats['skewness'] = df[cols].skew()  # 偏度
    desc_stats['kurtosis'] = df[cols].kurt()  # 峰度
    academic_table = desc_stats[[
        'mean', '50%', 'std', 'skewness', 'kurtosis'
    ]].copy()
    academic_table.columns = [
        '均值(Mean)', '中位数(Med)','标准差(SD)', '偏度(Skew)', '峰度(Kurt)'
    ]

    print("===== 论文描述性统计表 =====")
    print(academic_table.round(3))
    # ===== 所有取值 =====
    all_values = np.arange(1, 5.25, 0.25)

    # ===== 统计频数 =====
    freq_dict = {}
    ratio_dict = {}

    for col in cols:
        value_counts = df[col].value_counts().sort_index()
        freq = np.array([value_counts.get(v, 0) for v in all_values])
        freq_dict[col] = freq
        
        # 占比（每个变量自己的）
        ratio_dict[col] = deepcopy(freq)
    
    for col in cols:
        last_count = 0
        for i in range(len(all_values)):
            ratio_dict[col][i] += last_count
            last_count = ratio_dict[col][i]
        ratio_dict[col] = ratio_dict[col] / last_count

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    bar_colors  = ["#4E79A7", "#59A14F", "#E15759"]
    line_colors = ["#2F5D8A", "#3D8B37", "#C23B3D"]
    x = np.arange(len(all_values))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    bars1 = ax1.bar(
        x - width,
        freq_dict["危机-自己"],
        width=width,
        label="自伤风险",
        color=bar_colors[0],
        edgecolor='black',
        linewidth=0.8,
        alpha=0.7
    )
    bars2 = ax1.bar(
        x,
        freq_dict["问题-抑郁"],
        width=width,
        label="抑郁",
        color=bar_colors[1],
        edgecolor='black',
        linewidth=0.8,
        alpha=0.7
    )

    bars3 = ax1.bar(
        x + width,
        freq_dict["问题-焦虑"],
        width=width,
        label="焦虑",
        color=bar_colors[2],
        edgecolor='black',
        linewidth=0.8,
        alpha=0.7
    )

    add_labels(bars1, ax1)
    add_labels(bars2, ax1)
    add_labels(bars3, ax1)

    # ===== 双y轴 =====
    ax2 = ax1.twinx()
    line1, = ax2.plot(
        x,
        ratio_dict["危机-自己"],
        marker='o',
        markersize=7,
        linewidth=2.5,
        color=line_colors[0],
        label='自伤风险',
        alpha=0.5
    )

    line2, = ax2.plot(
        x,
        ratio_dict["问题-抑郁"],
        marker='s',
        markersize=7,
        linewidth=2.5,
        color=line_colors[1],
        label='抑郁',
        alpha=0.5
    )

    line3, = ax2.plot(
        x,
        ratio_dict["问题-焦虑"],
        marker='^',
        markersize=7,
        linewidth=2.5,
        color=line_colors[2],
        label='焦虑',
        alpha=0.5
    )

    # ===== 右轴百分比 =====
    ax2.set_ylabel('累计占比')
    ax2.set_ylim(0, 1)  # 占比范围0-1
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_values, rotation=0)
    ax1.set_ylabel('频数')
    ax1.set_xlabel('得分')

    # ===== 网格 =====
    ax1.grid(True, axis='y', alpha=0.4, linestyle='--')
    # ===== 图例合并 =====
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='center right',
        bbox_to_anchor=(0.98, 0.75),
        frameon=False,
        ncol=2
    )

    plt.tight_layout()
    plt.savefig(f'figure/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'../../论文正文/latex/fig/{save_name}.pdf', bbox_inches='tight')

def star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def T1_cronbach_alpha():
    import pingouin as pg
    df = pd.read_excel('T1.xls', dtype={'登录名': str})
    # df = pd.read_excel('D1.xlsx', dtype={'登录名': str})
    x_columns = [f"第{i}题" for i in range(1, 37)]
    for col in constant.common_sense:
        x_columns.remove(col)
    for col in constant.opposite:
        df[col] = 6 - df[col]
    x = df[x_columns]
    x = x.dropna()
    alpha = pg.cronbach_alpha(data=x)[0]
    print(alpha)

def venn_plot():
    from matplotlib_venn import venn3
    df = pd.read_csv("output/D1_data.csv", encoding="utf-8-sig")
    cols = ['危机-自己', '问题-焦虑', '问题-抑郁']
    c1, c2, c3 = cols
    df[c1] = df[c1].apply(lambda x: 1 if x > 3.5 else 0)
    df[c2] = df[c2].apply(lambda x: 1 if x > 3.75 else 0)
    df[c3] = df[c3].apply(lambda x: 1 if x > 3.5 else 0)
    # ========== 各种组合的统计 ==========

    # 三列同时为 True
    all_true = (df[c1] & df[c2] & df[c3]).sum()

    # 两列为 True 的三种情况
    c1_c2_true = (df[c1] & df[c2] & ~df[c3]).sum()
    c1_c3_true = (df[c1] & ~df[c2] & df[c3]).sum()
    c2_c3_true = (~df[c1] & df[c2] & df[c3]).sum()

    # 一列为 True 的三种情况
    only_c1 = (df[c1] & ~df[c2] & ~df[c3]).sum()
    only_c2 = (~df[c1] & df[c2] & ~df[c3]).sum()
    only_c3 = (~df[c1] & ~df[c2] & df[c3]).sum()
    outcomes = (only_c1, only_c2, c1_c2_true, only_c3, c1_c3_true, c2_c3_true, all_true)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(3.5, 3.5))
    venn = venn3(subsets=outcomes, set_labels=('自伤风险', '焦虑', '抑郁'))
    colors = ['#5B8FF9', '#5B8FF9', '#61DDAA', '#5B8FF9', '#61DDAA', '#61DDAA', '#F6BD16']

    for patch, color in zip(venn.patches, colors):
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('white')
            patch.set_linewidth(1.5)

    for text in venn.subset_labels:
        if text:
            text.set_fontsize(12)
            text.set_fontweight('bold')
            # text.set_color('black')

    for text in venn.set_labels:
        text.set_fontsize(12)
        # text.set_fontweight('bold')

    plt.box(False)
    plt.tight_layout()

    # 导出为 PDF（LaTeX 插入 PDF 矢量图不失真）
    plt.savefig("figure/venn.png", dpi=600, bbox_inches='tight')
    plt.savefig('../../论文正文/latex/fig/venn.pdf', bbox_inches='tight')

def check_normality_and_variance(df, y_column, group_column, gap):
    """
    检查正态性和方差齐性，辅助判断适合用 t 检验还是 Mann-Whitney U
    """
    import scipy.stats as stats
    group1 = df[df[group_column] <= gap][y_column].dropna()
    group2 = df[df[group_column] > gap][y_column].dropna()
    
    n1, n2 = len(group1), len(group2)
    print(f"样本量: Group1 (≤{gap}) = {n1}, Group2 (>{gap}) = {n2}")
    print("-" * 50)
    
    # ========== 1. 正态性检验 (Shapiro-Wilk) ==========
    print("【正态性检验】")
    shapiro1 = stats.shapiro(group1)
    shapiro2 = stats.shapiro(group2)
    print(f"Group1 (≤{gap}): W = {shapiro1[0]:.4f}, p = {shapiro1[1]:.4f}")
    print(f"Group2 (>{gap}): W = {shapiro2[0]:.4f}, p = {shapiro2[1]:.4f}")
    
    normal1 = shapiro1[1] > 0.05
    normal2 = shapiro2[1] > 0.05
    
    if normal1 and normal2:
        print("结论: 两组均符合正态分布 (p > 0.05)")
    else:
        print("结论: 至少有一组不符合正态分布 (p ≤ 0.05)")
    
    # ========== 2. 方差齐性检验 (Levene) ==========
    print("【方差齐性检验】")
    levene_stat, levene_p = stats.levene(group1, group2, center='median')
    print(f"Levene 检验: W = {levene_stat:.4f}, p = {levene_p:.4f}")
    
    if levene_p > 0.05:
        print("结论: 方差齐 (p > 0.05)")
    else:
        print("结论: 方差不齐 (p ≤ 0.05)")
    print("-" * 50)

def t(df, y_column, group_column, gap):
    # check_normality_and_variance(df, y_column, group_column, gap)
    group1 = df[df[group_column] <= gap][y_column]
    group2 = df[df[group_column] > gap][y_column]
    
    # 计算统计量
    mean1, std1 = group1.mean(), group1.std()
    mean2, std2 = group2.mean(), group2.std()
    median1, median2 = group1.median(), group2.median()
    print(f"Group1 (≤{gap}): 成绩 = {mean1:.2f} ± {std1:.2f}, 中位数 = {median1:.2f}")
    print(f"Group2 (>{gap}): 成绩 = {mean2:.2f} ± {std2:.2f}, 中位数 = {median2:.2f}")
    t_stat, t_p = ttest_ind(group1, group2, equal_var=True)  # 方差齐，用标准t检验
    print(f"独立样本 t 检验: t = {t_stat:.4f}, p = {t_p:.4f}")

def u(df, y_column, group_column, gap):
    group1 = df[df[group_column] <= gap][y_column].dropna()
    group2 = df[df[group_column] > gap][y_column].dropna()
    mean1, std1 = group1.mean(), group1.std()
    mean2, std2 = group2.mean(), group2.std()
    median1, median2 = group1.median(), group2.median()
    n1, n2 = len(group1), len(group2)
    print(f"Group1 (≤{gap}): n={n1}, 均值={mean1:.2f} ± {std1:.2f}, 中位数={median1:.2f}")
    print(f"Group2 (>{gap}): n={n2}, 均值={mean2:.2f} ± {std2:.2f}, 中位数={median2:.2f}")
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Mann-Whitney U 检验: U = {u_stat:.2f}, p = {p_value:.4f}")
    n1, n2 = len(group1), len(group2)
    # 计算 Z 分数 (近似)
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mu) / sigma
    # 计算效应量 r
    r = abs(z) / np.sqrt(n1 + n2)
    print("r: ", r)

def chi(df, y_column, group_column, y_gap=None, group_gap=None):
    """
    进行卡方检验, 检验不同组别在y_column上的分布差异
    参数:
    df : DataFrame
    y_column : str 目标变量
    group_column : 分组变量
    y_gap : [0, 3.5, 5.0]
    group_gap : [0, 3.5, 5.0]
    """
    data = df[[y_column, group_column]].copy()
    if y_gap is not None:
        data[y_column] = pd.cut(data[y_column], bins=y_gap, include_lowest=True)
    if group_gap is not None:
        data[group_column] = pd.cut(data[group_column], bins=group_gap, include_lowest=True)
    contingency_table = pd.crosstab(data[group_column], data[y_column])
    # 计算各分组人数
    group_counts = data[group_column].value_counts().sort_index()
    # 执行卡方检验
    chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)
    # 打印结果
    print("="*50)
    print("列联表（观测频数）：")
    print(contingency_table)
    print("\n" + "="*50)
    print("各分组人数：")
    for group, count in group_counts.items():
        print(f"  {group}: {count}人")
    print(f"\n总样本数: {len(data)}人")
    print("="*50)
    print(f"卡方统计量: {chi2_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    print(f"自由度: {dof}")
    print("="*50)

# 主要相关性检验, 包括卡方检验, t检验以及spearman检验
def kafangjianyan():
    # 卡方检验 性别和自变量危机 与 因变量之间的关系
    df = pd.read_csv("output/D1_data.csv", encoding='utf-8-sig')
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    group_columns = ["性别", "危机"]
    df["危机"] = (df["第13题"] + df["第31题"]) / 2
    y_gaps = [[0, 3.5, 5.0], [0, 3.5, 5.0], [0, 3.75, 5.0]]
    group_gaps = [None, [0, 3.5, 5.0]]
    for i in range(3):
        for j in range(2):
            chi(df, y_column=y_columns[i], group_column=group_columns[j], y_gap=y_gaps[i], group_gap=group_gaps[j])

def t_u_description():
    df = pd.read_excel("D3.xlsx")
    df = df[df["登录号"] != 3101134001320240111].copy() # 304个
    print("自伤风险 与 成绩的关系：")
    t(df, y_column="总分_分数", group_column="危机-自己", gap=3.5)
    print("\n抑郁 与 成绩的关系：")
    t(df, y_column="总分_分数", group_column="问题-抑郁", gap=3.5)
    print("\n焦虑 与 成绩的关系：")
    t(df, y_column="总分_分数", group_column="问题-焦虑", gap=3.75)

    df = pd.read_excel("D1_complete.xlsx")
    df = df[df["登录号"] != 3101134001320240111].copy() # 304个
    name = "体测总分"
    print(f"\n\n\n\n自伤风险 与 {name}的关系：")
    u(df, y_column=name, group_column="危机-自己", gap=3.5)
    print(f"\n抑郁 与 {name}的关系：")
    u(df, y_column=name, group_column="问题-抑郁", gap=3.5)
    print(f"\n焦虑 与 {name}的关系：")
    u(df, y_column=name, group_column="问题-焦虑", gap=3.75)

def spearman_plot_heatmap_T1():
    df = pd.read_csv("output/D1_data.csv", encoding="utf-8-sig")
    # df.drop(constant.dependent_variable, axis=1, inplace=True)
    df.drop("性别", axis=1, inplace=True)
    df[constant.opposite] = 6 - df[constant.opposite]
    df.rename(columns=constant.question_summary, inplace=True)

    y_columns = ["自伤风险", "抑郁", "焦虑"]
    all_x_columns = [col for col in df.columns.tolist() if col not in y_columns]
    all_p_values = []
    all_results = []  # 保存所有结果
    for y_col in y_columns:
        for x_col in all_x_columns:
            rho, p = spearmanr(df[y_col], df[x_col])
            all_p_values.append(p)
            all_results.append({
                'y': y_col,
                'x': x_col,
                'rho': rho,
                'p': p
            })
    
    # ===== 对所有 102 个 p 值进行 FDR 校正 =====
    rejected, q_values = fdrcorrection(all_p_values, alpha=0.05)
    for i, result in enumerate(all_results):
        result['q'] = q_values[i]
    selected_x_columns = ["有过自杀想法", "有过自伤行为", "太难不想活了",
                          "开心分享家人", "喜欢家庭活动", "困难求助家人",
                          "纠结别人的话", "在意他人评价", "矛盾难以释怀",
                          "现在心情好坏", "总是心情不好",
                          "学习能力自信", "有问题怨自己",
                          "晚上总是失眠", "晚上难以入睡",
                          "快被压力压垮"]
    selected_results = [r for r in all_results if r['x'] in selected_x_columns]
    
    rho_matrix = pd.DataFrame(index=y_columns, columns=selected_x_columns)
    q_matrix = pd.DataFrame(index=y_columns, columns=selected_x_columns)
    for result in selected_results:
        rho_matrix.loc[result['y'], result['x']] = result['rho']
        q_matrix.loc[result['y'], result['x']] = result['q']
    rho_matrix = rho_matrix.astype(float)
    q_matrix = q_matrix.astype(float)
    annot_matrix = rho_matrix.round(2).astype(str)
    for i in rho_matrix.index:
        for j in rho_matrix.columns:
            annot_matrix.loc[i, j] += '\n' + star(q_matrix.loc[i, j])
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘图设置
    plt.figure(figsize=(8, 3.5)) # 预测因子多，建议增加高度
    sns.set_theme(style="white", font='SimHei', font_scale=1.0) # 确保中文显示

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax = sns.heatmap(rho_matrix, 
                cmap=cmap, 
                vmax=.5, center=0.15,
                annot=annot_matrix, fmt="", # 建议开启数值，更直观
                annot_kws={"size": 8, "color": "black"},
                square=False, # 长方形矩阵设为 False 比例更协调
                linewidths=.3, 
                linecolor="white",
                cbar_kws={"shrink": .85})
    split_indices = [3, 6, 9, 11, 13, 15] 
    
    # 获取 y 轴范围（确保线贯穿上下）
    ymin, ymax = ax.get_ylim()
    for i, x in enumerate(split_indices):
        ax.vlines(
            x=x,
            ymin=ymin,
            ymax=ymax,
            colors="black",
            linewidth=1.0,
            linestyles="--",
            alpha=0.8
        )
    
    plt.xticks(rotation=40, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout() # 防止坐标轴标签被遮挡
    plt.savefig('figure/heatmap_T1.png', dpi=600, bbox_inches='tight')
    plt.savefig('../../论文正文/latex/fig/heatmap_T1.pdf', bbox_inches='tight')

def spearman_plot_heatmap_T2():
    dfs = [
        pd.read_csv(f"output/D2_{i}_data.csv", encoding="utf-8-sig") 
        for i in range(1, 5)
    ]
    xs = constant.new_x[1:5]
    new_xs = [
        ["WMC", "RTC", "2back-CV", "RTCEI"],
        ["PE", "LSR", "RC"],
        ["AAP", "ER", "ML"],
        ["SSRT", "Mean RT", "sst-CV"]
    ]
    y_columns = ["自伤风险", "抑郁", "焦虑"]
    all_rhos = []
    all_ps = []
    for i in range(4):
        rename_dict = {xs[i][j]: new_xs[i][j] for j in range(len(xs[i]))}
        dfs[i].rename(columns=rename_dict, inplace=True)
        dfs[i].rename(columns=constant.question_summary, inplace=True)
        df = dfs[i][y_columns + new_xs[i]]

        rho_matrix = pd.DataFrame(index=y_columns, columns=new_xs[i], dtype=float)
        p_matrix = pd.DataFrame(index=y_columns, columns=new_xs[i], dtype=float)
        for y_col in y_columns:
            for x_col in new_xs[i]:
                rho, p = spearmanr(df[y_col], df[x_col])
                rho_matrix.loc[y_col, x_col] = rho
                p_matrix.loc[y_col, x_col] = p
        all_rhos.append(rho_matrix)
        all_ps.append(p_matrix)
    
    corr_matrix = pd.concat(all_rhos, axis=1)
    p_matrix = pd.concat(all_ps, axis=1)
    rho_matrix = rho_matrix.astype(float)
    p_matrix = p_matrix.astype(float)

    annot_matrix = corr_matrix.round(2).astype(str)
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            annot_matrix.loc[i, j] += '\n' + star(p_matrix.at[i, j])

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘图设置
    plt.figure(figsize=(8, 3.5)) # 预测因子多，建议增加高度
    sns.set_theme(style="white", font='SimHei', font_scale=1.0) # 确保中文显示

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax = sns.heatmap(corr_matrix, 
                cmap=cmap, 
                vmax=0.3, center=0.0, vmin=-0.3, 
                annot=annot_matrix, fmt='',
                annot_kws={"size": 8, "color": "black"},
                square=False, # 长方形矩阵设为 False 比例更协调
                linewidths=.3, 
                linecolor="white",
                cbar_kws={"shrink": .85})
    split_indices = [4, 7, 10, 13] 
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
            linewidth=1.0,
            linestyles="--",
            alpha=0.8
        )
    
    plt.xticks(rotation=30, ha='center', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout() # 防止坐标轴标签被遮挡
    plt.savefig('figure/heatmap_T2.png', dpi=600, bbox_inches='tight')
    plt.savefig('../../论文正文/latex/fig/heatmap_T2.pdf', bbox_inches='tight')

def filter_sem():
    df = pd.read_csv("output/D1_data.csv", encoding="utf-8-sig")
    df.drop("性别", axis=1, inplace=True)
    df[constant.opposite] = 6 - df[constant.opposite]
    # df.rename(columns=constant.question_summary, inplace=True)

    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]
    x_columns = [col for col in df.columns.tolist() if col not in y_columns]
    
    full_corr = df.corr(method='spearman')
    corr_matrix = full_corr.loc[y_columns, x_columns]
    result_list = [
        corr_matrix.columns[corr_matrix.loc[y_var] > 0.16].tolist() 
        for y_var in y_columns
    ]
    new_y_names = ["risk", "depression", "anxiety"]
    sem_x = {new_y_names[i]: result_list[i] for i in range(3)}
    with open("../可解释性分析/sem/x.json", 'w', encoding="utf-8") as f:
        json.dump(sem_x, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 危机-自己 <= 3.5, 问题-抑郁 <= 3.5, 问题-焦虑 <= 3.75
    describe_dependent_variables()
    T1_cronbach_alpha()
    venn_plot()
    kafangjianyan()
    t_u_description()
    spearman_plot_heatmap_T1()
    spearman_plot_heatmap_T2()
    filter_sem()
    
    
