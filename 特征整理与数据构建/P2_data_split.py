import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np

def _get_stratify_labels(y, sep=None):
    if sep is None:
        return y.astype("category")

    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        raise ValueError("sep 参数仅支持数值类型的因变量")

    bins = np.arange(1.0, 5.0 + sep, sep)
    if len(bins) < 2:
        bins = np.array([y_numeric.min() - 1e-8, y_numeric.max() + sep])

    labels = pd.cut(y_numeric, bins=bins, include_lowest=True, right=True)
    return labels.astype("category")


def split(df, random_state=49, k=5, y_columns=None, sep=None, save_path="folds.pkl"):
    """
    对每个因变量分别做 Stratified K-Fold，并保存每一折索引。

    参数:
      df: DataFrame
      y_columns: 因变量列名列表
      sep: 如果因变量是连续数值，可按间隔离散化后再分层，sep=0.2 表示每 0.2 一个区间
      save_path: 保存 fold 索引的路径
    """

    assert y_columns is not None and len(y_columns) > 0, "必须指定因变量列列表 y_columns"

    folds = {}

    print("=== Stratified K-Fold for each target column ===")

    for y_column in y_columns:
        y = df[y_column]
        if y_column == "危机-自己":
            strat_labels = _get_stratify_labels(y, sep=sep[0])
        elif y_column == "问题-抑郁":
            strat_labels = _get_stratify_labels(y, sep=sep[1])
        else:
            strat_labels = _get_stratify_labels(y, sep=sep[2])
        strat_codes = strat_labels.cat.codes

        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=random_state
        )

        column_folds = []
        print(f"\n--- 目标列: {y_column} ---")
        if sep is not None:
            print(f"分层标签分布 (sep={sep}):\n", strat_labels.value_counts().sort_index())

        for fold, (train_index, val_index) in enumerate(skf.split(df, strat_codes)):
            column_folds.append({
                "fold": fold,
                "train_idx": train_index,
                "val_idx": val_index
            })

        folds[y_column] = column_folds

    with open(save_path, "wb") as f:
        pickle.dump({
            "folds": folds,
            "all_index": df.index.tolist()
        }, f)

    return folds


def load_folds_and_split(df_path="data.csv", folds_path="folds.pkl", y_column=None):
    # 读取原始数据
    df = pd.read_csv(df_path, encoding='utf-8-sig')

    # 读取保存的fold
    with open(folds_path, "rb") as f:
        data = pickle.load(f)

    folds = data["folds"]

    if isinstance(folds, dict):
        if y_column is None:
            if len(folds) == 1:
                y_column = next(iter(folds))
            else:
                raise ValueError("请指定 y_column，或保存的 folds 仅包含一个目标列")
        column_folds = folds[y_column]
    else:
        if y_column is None:
            raise ValueError("请指定 y_column")
        column_folds = folds

    for fold_info in column_folds:
        fold_num = fold_info["fold"]
        train_idx = fold_info["train_idx"]
        val_idx = fold_info["val_idx"]

        # 生成训练集和验证集
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        X_train = train_df.drop(columns=[y_column])
        y_train = train_df[y_column]

        X_val = val_df.drop(columns=[y_column])
        y_val = val_df[y_column]

        print(f"\nFold {fold_num+1} sizes:")
        print("Train:", len(X_train), "Val:", len(X_val))
        print("Val label distribution:\n", y_val.value_counts())



if __name__ == "__main__":
    sep = [2.5, 2.5, 2.75]  # <= 3.5, 3.5, 3.75, 对应危机-自己、问题-抑郁、问题-焦虑
    y_columns = ["危机-自己", "问题-抑郁", "问题-焦虑"]

    df1 = pd.read_csv("output/D1_data.csv", encoding='utf-8-sig')
    df21 = pd.read_csv("output/D2_1_data.csv", encoding='utf-8-sig')
    df22 = pd.read_csv("output/D2_2_data.csv", encoding='utf-8-sig')
    df23 = pd.read_csv("output/D2_3_data.csv", encoding='utf-8-sig')
    df24 = pd.read_csv("output/D2_4_data.csv", encoding='utf-8-sig')
    df3 = pd.read_csv("output/D3_data.csv", encoding='utf-8-sig')
    df4 = pd.read_csv("output/D4_data.csv", encoding='utf-8-sig')

    print("\n\n D1")
    split(df1, random_state=49, k=5, y_columns=y_columns, sep=sep, save_path="output/D1_folds.pkl")
    for i in range(1, 5):
        print("\n\n D2", i)
        split(eval(f"df2{i}"), random_state=49, k=5, y_columns=y_columns, sep=sep, save_path=f"output/D2_{i}_folds.pkl")
    
    print("\n\n D3")
    split(df3, random_state=49, k=5, y_columns=y_columns, sep=sep, save_path="output/D3_folds.pkl")
    
    print("\n\n D4")
    split(df4, random_state=49, k=5, y_columns=y_columns, sep=sep, save_path="output/D4_folds.pkl")
