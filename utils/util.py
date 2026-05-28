from sklearn.model_selection import StratifiedKFold
import copy
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from . import constant

need_standardization_columns = []
for c in constant.new_x:
    need_standardization_columns.extend(c)
need_standardization = ["LR", "SVM", "KNN", "BalancedLR", "BalancedSVM"]


def only_train_X(model, folds, X, y, model_name, inner_splits=5, random_state=42):
    is_std = True if model_name in need_standardization else False
    outer_results = []
    outer_y = []
    for fold_info in folds:
        outer_train_idx = fold_info["train_idx"]
        X_train_full = X.iloc[outer_train_idx]
        y_train_full = y.iloc[outer_train_idx]
        skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        for inner_train_idx, inner_val_idx in skf.split(X_train_full, y_train_full):
            X_inner_train = X_train_full.iloc[inner_train_idx]
            y_inner_train = y_train_full.iloc[inner_train_idx]
            X_inner_val = X_train_full.iloc[inner_val_idx]
            y_inner_val = y_train_full.iloc[inner_val_idx]
            scaler = StandardScaler()
            if is_std:
                X_inner_train = scaler.fit_transform(X_inner_train)
                X_inner_val = scaler.transform(X_inner_val)
            else:
                standard_columns = [c for c in X_inner_train.columns if c in need_standardization_columns]
                if len(standard_columns) != 0:
                    X_inner_train.loc[:, standard_columns] = scaler.fit_transform(X_inner_train[standard_columns])
                    X_inner_val.loc[:, standard_columns] = scaler.transform(X_inner_val[standard_columns])

            temp_model = copy.deepcopy(model)
            temp_model.fit(X_inner_train, y_inner_train)
            inner_oof_proba = (temp_model.predict_proba(X_inner_val)[:, 1])
            outer_results.extend(inner_oof_proba.tolist())
            outer_y.extend(y_inner_val.tolist())

    final_metrics = {
        "model_name": model_name,
        "PR-AUC": average_precision_score(outer_y, outer_results)
    }
    return final_metrics

def train(model, folds, X, y, model_name):
    is_std = True if model_name in need_standardization else False
    results = {}
    oof_pred_proba = np.zeros(len(y))
    oof_pred_label = np.zeros(len(y))
    for fold_info in tqdm(folds):
        train_idx = fold_info["train_idx"]
        val_idx = fold_info["val_idx"]
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        scaler = StandardScaler()
        if is_std:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        else:
            standard_columns = [c for c in X_train.columns if c in need_standardization_columns]
            if len(standard_columns) != 0:
                X_train.loc[:, standard_columns] = scaler.fit_transform(X_train[standard_columns])
                X_val.loc[:, standard_columns] = scaler.transform(X_val[standard_columns])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        oof_pred_proba[val_idx] = y_proba
        oof_pred_label[val_idx] = y_pred
        fold_metrics = {
            "F1": f1_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "ROC-AUC": roc_auc_score(y_val, y_proba),
            "PR-AUC": average_precision_score(y_val, y_proba)
        }
        results[fold_info["fold"] + 1] = fold_metrics

    final_metrics = {}
    metrics_funcs = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1_score,
    }
    for name, func in metrics_funcs.items():
        score = func(y, oof_pred_label)
        final_metrics[name] = score

    final_metrics["ROC-AUC"] = roc_auc_score(y, oof_pred_proba)
    final_metrics["PR-AUC"] = average_precision_score(y, oof_pred_proba)
    results["overall"] = final_metrics
    results["oof_pred_proba"] = oof_pred_proba
    results["oof_pred_label"] = oof_pred_label

    return results

def load_folds_and_split(df_path, folds_path, y_column, remove_columns=[], gap=3.5, y_columns = ["危机-自己", "问题-焦虑", "问题-抑郁"]):
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


def prepare_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [prepare_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(prepare_for_json(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)
    