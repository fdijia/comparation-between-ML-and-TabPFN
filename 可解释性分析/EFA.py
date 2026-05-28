import json
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer import FactorAnalyzer
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import constant

def quick_efa(data, n_factors=None, threshold=0.4):
    # 标准化
    from sklearn.preprocessing import StandardScaler
    data_scaled = pd.DataFrame(
        StandardScaler().fit_transform(data),
        columns=data.columns
    )
    
    # 自动确定因子数
    if n_factors is None:
        fa_temp = FactorAnalyzer(rotation=None, method='principal')
        fa_temp.fit(data_scaled)
        ev, _ = fa_temp.get_eigenvalues()
        n_factors = sum(ev > 1)
        print(f"自动确定因子数: {n_factors}")
    
    # 因子分析
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
    fa.fit(data_scaled)
    
    # 载荷矩阵
    loadings = pd.DataFrame(
        fa.loadings_,
        index=data.columns,
        columns=[f'F{i+1}' for i in range(n_factors)]
    )
    # 分配变量到因子（取最大载荷的因子）
    max_loadings = loadings.abs().max(axis=1)
    assigned_factor = loadings.abs().idxmax(axis=1)
    
    # 获取对应的载荷值
    assigned_loadings = [loadings.loc[var, factor] for var, factor in zip(loadings.index, assigned_factor)]
    
    # 判断是否达标
    is_valid = max_loadings >= threshold
    
    # 创建结果表
    detail_result = pd.DataFrame({
        '变量名': loadings.index,
        '归属因子': assigned_factor,
        '载荷值': assigned_loadings,
        '是否达标(>={})'.format(threshold): is_valid,
        '最高载荷(绝对值)': max_loadings
    })
    
    # 添加所有因子的载荷
    for col in loadings.columns:
        detail_result[f'{col}_载荷'] = loadings[col].values
    
    # 创建因子-变量字典（你想要的格式）
    factor_vars = {}
    for factor in loadings.columns:
        # 只提取达标（载荷>=threshold）的变量
        valid_vars = detail_result[
            (detail_result['归属因子'] == factor) & 
            (detail_result['是否达标(>={})'.format(threshold)] == True)
        ]['变量名'].tolist()
        
        # 如果为空，则取所有归属该因子的变量
        if not valid_vars:
            valid_vars = detail_result[detail_result['归属因子'] == factor]['变量名'].tolist()
        
        factor_vars[factor] = valid_vars
    
    return fa, loadings, detail_result, factor_vars

def run_vif_calculation(selected_columns, df):
    # 2. 检查变量是否存在于数据集中
    valid_cols = [col for col in selected_columns if col in df.columns]

    # 3. 准备数据：提取列、去除空值、转换为浮点数
    X = df[valid_cols].dropna().astype(float)
    
    # 4. 计算 VIF
    # statsmodels 要求显式添加常数项以计算截距，否则 VIF 结果不准确
    X_with_const = add_constant(X)
    
    vif_results = pd.DataFrame()
    vif_results["Variable"] = X_with_const.columns
    vif_results["VIF"] = [
        variance_inflation_factor(X_with_const.values, i) 
        for i in range(X_with_const.shape[1])
    ]

    # 5. 格式化输出
    # 移除常数项(const)并按 VIF 降序排列
    final_output = vif_results[vif_results["Variable"] != "const"].sort_values(by="VIF", ascending=False)
    
    print("\n" + "="*30)
    print("      多重共线性 (VIF) 分析报告")
    print("="*30)
    if final_output.empty:
        print("未发现有效变量或数据不足。")
    else:
        print(final_output.to_string(index=False, formatters={'VIF': '{:.3f}'.format}))
    print("="*30)
    print("💡 注: VIF > 10 通常暗示严重的共线性问题。")


rename_dict = {}
for k, v in constant.question_summary.items():
    if k.startswith("第"):
        rename_dict['x' + k[1: -1]] = v
data = pd.read_csv("sem/data.csv")
data.rename(columns=rename_dict, inplace=True)
# 提取符合条件的列名
with open("sem/top_values.json", 'r', encoding='utf-8') as f:
    x_data = json.load(f)

for key, value in x_data.items():
    selected_columns = []
    for v_k, v_v in value.items():
        if v_v > 105:
            continue
        # if len(selected_columns) == 9:
        #     break
        if v_k not in selected_columns:
            selected_columns.append(v_k)
    fa_result, loading_matrix, _, results = quick_efa(data[selected_columns])
    # run_vif_calculation(selected_columns, data)
    print(results)


