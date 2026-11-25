"""CardioAI 模块二：训练并保存心血管风险预测管线（一次性脚本）。

运行：
    F:\\software\\anaconda\\envs\\cardioenv\\python.exe module2_predictor\\train_and_save.py

产出：module2_predictor/cardio_predictor_model.pkl（预处理器 + XGBClassifier 的完整 Pipeline）
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "cardio_predictor_model.pkl"

DATA_PATH_CANDIDATES = (
    Path(r"J:\python-project\sh_1AI_LLM\ai_coding\data\心血管疾病.xlsx"),
    Path(r"D:\CZ\sh_1AI_LLM\ai_coding\data\心血管疾病.xlsx"),
)

DAYS_PER_YEAR = 365.25
SBP_RANGE = (90, 250)
DBP_RANGE = (60, 150)

NUMERIC_FEATURES = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "cardio"

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": 42,
}


def resolve_data_path() -> Path:
    for candidate in DATA_PATH_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到数据文件，候选路径：\n" + "\n".join(str(p) for p in DATA_PATH_CANDIDATES)
    )


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """按血压合理性规则清洗，返回清洗结果与每步统计。"""
    steps: list[tuple[str, int, int]] = [("原始记录", len(df), 0)]

    invalid_order = df["ap_lo"] >= df["ap_hi"]
    stage1 = df.loc[~invalid_order]
    steps.append(("舒张压 >= 收缩压", len(stage1), int(invalid_order.sum())))

    out_of_range = ~stage1["ap_hi"].between(*SBP_RANGE) | ~stage1["ap_lo"].between(*DBP_RANGE)
    cleaned = stage1.loc[~out_of_range].reset_index(drop=True)
    steps.append(("血压超出安全区间", len(cleaned), int(out_of_range.sum())))

    return cleaned, steps


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_years"] = (out["age"] / DAYS_PER_YEAR).round().astype(int)
    return out


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**XGB_PARAMS)),
        ]
    )


def print_step_report(steps: list[tuple[str, int, int]]) -> None:
    print("数据清洗：")
    for name, remaining, removed in steps:
        print(f"  {name:<14} 剩余 {remaining:>6,} 条，本步剔除 {removed:>5,} 条")


def main() -> int:
    path = resolve_data_path()
    print(f"数据文件：{path}")

    raw = load_raw(path)
    cleaned, steps = clean(raw)
    df = add_features(cleaned)

    print_step_report(steps)
    print(f"训练样本：{len(df):,} 条（原始 {len(raw):,} 条）")

    extreme_height = int((~df["height"].between(100, 250)).sum())
    extreme_weight = int((~df["weight"].between(30, 300)).sum())
    print(
        f"数据体检：身高越界 {extreme_height} 条，体重越界 {extreme_weight} 条"
        "（按需求未额外过滤，仅提示）"
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练集 {len(X_train):,} 条 / 测试集 {len(X_test):,} 条，特征 {len(FEATURE_COLUMNS)} 个")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n测试集评估：")
    print(f"  accuracy  {accuracy_score(y_test, pred):.4f}")
    print(f"  roc_auc   {roc_auc_score(y_test, proba):.4f}")
    print(f"  precision {precision_score(y_test, pred):.4f}")
    print(f"  recall    {recall_score(y_test, pred):.4f}")
    print(f"  f1        {f1_score(y_test, pred):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"  混淆矩阵  TN={tn} FP={fp} FN={fn} TP={tp}")

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    print(f"\n模型已保存：{MODEL_PATH}（{size_mb:.2f} MB）")
    print(f"特征顺序：{list(pipeline.feature_names_in_)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())