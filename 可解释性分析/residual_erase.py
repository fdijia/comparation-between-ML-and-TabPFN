import pandas as pd
import statsmodels.api as sm


def erase_gender(data):
    target_cols = [col for col in data.columns if col != "gender"]
    residual_data = data.copy()
    residual_data.drop("gender", axis=1, inplace=True)
    for col in target_cols:
        residual_data[col] = residual_data[col].astype(float)

    for col in target_cols:
        df_tmp = data[[col, "gender"]]
        X = sm.add_constant(df_tmp["gender"])
        y = df_tmp[col]
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        residual_data.loc[df_tmp.index, col] = residuals
    return residual_data

data = pd.read_csv("../特征整理与数据构建/output/D1_data.csv")
rename_dict = {f"第{i}题": f"x{i}" for i in range(1, 37) if i != 11 and i != 29}
rename_dict.update({"问题-焦虑": "anxiety", "问题-抑郁": "depression", "危机-自己": "risk", "性别": "gender"})
data = data.rename(columns=rename_dict)
new_data = erase_gender(data)
new_data.to_csv("sem/data.csv", index=False)
