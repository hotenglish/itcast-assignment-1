"""CardioAI 模块一：心血管疾病数据看板。

对原始体检数据进行清洗、特征工程与交互式可视化。

运行（在 CODE_ROOT / aicodes 目录下）：
    F:\\software\\anaconda\\envs\\cardioenv\\python.exe -m streamlit run module1_dashboard\\cardio_dashboard.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------- 常量映射

DATA_PATH_CANDIDATES = (
    Path(r"J:\python-project\sh_1AI_LLM\ai_coding\data\心血管疾病.xlsx"),
    Path(__file__).resolve().parents[2] / "data" / "心血管疾病.xlsx",
)

DAYS_PER_YEAR = 365.25
SBP_RANGE = (90, 250)
DBP_RANGE = (60, 150)
AGE_BIN_WIDTH = 5

BMI_BINS = [0.0, 18.5, 25.0, 30.0, np.inf]
BMI_LABELS = ["偏瘦", "正常", "超重", "肥胖"]

GENDER_LABELS = {1: "女性", 2: "男性"}
CARDIO_LABELS = {0: "无心血管疾病", 1: "有心血管疾病"}
CHOLESTEROL_LABELS = {1: "正常", 2: "偏高", 3: "显著偏高"}
GLUC_LABELS = {1: "正常", 2: "偏高", 3: "显著偏高"}

CARDIO_VALUES = tuple(CARDIO_LABELS)
CARDIO_NAMES = tuple(CARDIO_LABELS.values())
CARDIO_COLORS = {"无心血管疾病": "#4C78A8", "有心血管疾病": "#E45756"}

# ---------------------------------------------------------------- 数据管线


def resolve_data_path() -> Path:
    for candidate in DATA_PATH_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到数据文件，候选路径：\n" + "\n".join(str(p) for p in DATA_PATH_CANDIDATES)
    )


@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(col).strip() for col in df.columns]
    return df


@st.cache_data(show_spinner=False)
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """剔除舒张压不低于收缩压、以及血压超出人体合理区间的记录。"""
    stage1 = df.loc[df["ap_lo"] < df["ap_hi"]]
    in_range = stage1["ap_hi"].between(*SBP_RANGE) & stage1["ap_lo"].between(*DBP_RANGE)
    return stage1.loc[in_range].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_years"] = np.floor(out["age"] / DAYS_PER_YEAR + 0.5).astype(int)
    out["bmi"] = out["weight"] / (out["height"] / 100) ** 2
    out["bmi_category"] = pd.cut(out["bmi"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    out["gender_label"] = out["gender"].map(GENDER_LABELS)
    out["cardio_label"] = out["cardio"].map(CARDIO_LABELS)
    out["cholesterol_label"] = out["cholesterol"].map(CHOLESTEROL_LABELS)
    out["gluc_label"] = out["gluc"].map(GLUC_LABELS)
    return out


@st.cache_data(show_spinner=False)
def build_dataset(path: str) -> pd.DataFrame:
    return add_features(clean(load_raw(path)))


# ---------------------------------------------------------------- 筛选


@dataclass(frozen=True)
class Filters:
    age_range: tuple[int, int]
    genders: tuple[int, ...]
    cardio_choices: tuple[int, ...]


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    mask = (
        df["age_years"].between(*filters.age_range)
        & df["gender"].isin(filters.genders)
        & df["cardio"].isin(filters.cardio_choices)
    )
    return df.loc[mask]


def summarize(view: pd.DataFrame) -> dict[str, str]:
    return {
        "总记录数": f"{len(view):,}",
        "心血管疾病风险率": f"{view['cardio'].mean() * 100:.2f}%",
    }


# ---------------------------------------------------------------- 图表


def age_bin_edges(age_years: pd.Series) -> list[int]:
    """以 AGE_BIN_WIDTH 为步长、对齐到整数倍，覆盖当前筛选数据的年龄范围。"""
    lo = int(math.floor(age_years.min() / AGE_BIN_WIDTH) * AGE_BIN_WIDTH)
    hi = int(math.ceil((age_years.max() + 1) / AGE_BIN_WIDTH) * AGE_BIN_WIDTH)
    return list(range(lo, max(hi, lo + AGE_BIN_WIDTH) + 1, AGE_BIN_WIDTH))


def binned_ages(age_years: pd.Series) -> tuple[pd.Series, list[str]]:
    edges = age_bin_edges(age_years)
    labels = [f"{start}-{end - 1}" for start, end in zip(edges[:-1], edges[1:])]
    binned = pd.cut(age_years, bins=edges, labels=labels, right=False)
    return binned, labels


def count_by_group(view: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """按分组列与患病标签计数，供柱状图直接消费。"""
    table = (
        view.groupby([group_column, "cardio_label"], observed=True)
        .size()
        .rename("记录数")
        .reset_index()
    )
    return table


def age_counts(view: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    binned, labels = binned_ages(view["age_years"])
    table = count_by_group(view.assign(age_group=binned), "age_group")
    return table, labels


def age_histogram(view: pd.DataFrame):
    table, labels = age_counts(view)
    fig = px.bar(
        table,
        x="age_group",
        y="记录数",
        color="cardio_label",
        barmode="stack",
        color_discrete_map=CARDIO_COLORS,
        category_orders={"age_group": labels, "cardio_label": list(CARDIO_NAMES)},
        labels={"age_group": f"年龄（{AGE_BIN_WIDTH} 岁一档）", "cardio_label": "心血管疾病"},
        title="年龄分布（固定分箱，按是否患病堆叠）",
    )
    fig.update_layout(template="plotly_white", legend_title_text="", bargap=0.15)
    fig.update_xaxes(type="category", tickmode="array", tickvals=labels, ticktext=labels)
    return fig


def bmi_stacked_chart(view: pd.DataFrame):
    table = count_by_group(view, "bmi_category")
    fig = px.bar(
        table,
        x="bmi_category",
        y="记录数",
        color="cardio_label",
        barmode="stack",
        color_discrete_map=CARDIO_COLORS,
        category_orders={"bmi_category": BMI_LABELS, "cardio_label": list(CARDIO_NAMES)},
        labels={"bmi_category": "BMI 分类", "cardio_label": "心血管疾病"},
        title="BMI 分类对心血管疾病的影响",
    )
    fig.update_layout(template="plotly_white", legend_title_text="", bargap=0.15)
    fig.update_xaxes(type="category")
    return fig


# ---------------------------------------------------------------- 页面


def render_sidebar(full: pd.DataFrame, raw_rows: int) -> Filters:
    with st.sidebar:
        st.header("筛选条件")

        age_min = int(full["age_years"].min())
        age_max = int(full["age_years"].max())
        age_range = st.slider(
            "年龄（岁）", min_value=age_min, max_value=age_max, value=(age_min, age_max)
        )

        genders = st.multiselect(
            "性别",
            options=list(GENDER_LABELS),
            default=list(GENDER_LABELS),
            format_func=lambda value: GENDER_LABELS[value],
        )
        cardio_choices = st.multiselect(
            "心血管疾病",
            options=list(CARDIO_LABELS),
            default=list(CARDIO_LABELS),
            format_func=lambda value: CARDIO_LABELS[value],
        )

        with st.expander("数据概览", expanded=True):
            st.write(f"原始记录：{raw_rows:,}")
            st.write(f"清洗后记录：{len(full):,}")
            st.write(f"剔除记录：{raw_rows - len(full):,}")

    return Filters(tuple(age_range), tuple(genders), tuple(cardio_choices))


def main() -> None:
    st.set_page_config(page_title="心血管疾病数据看板", layout="wide")
    st.title("心血管疾病数据看板")
    st.caption("对清洗后的心血管体检数据进行筛选与可视化分析")

    try:
        path = resolve_data_path()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    raw_rows = len(load_raw(str(path)))
    full = build_dataset(str(path))

    filters = render_sidebar(full, raw_rows)
    if not filters.genders or not filters.cardio_choices:
        st.warning("请至少选中一个性别与一种患病状态。")
        st.stop()

    view = apply_filters(full, filters)
    if view.empty:
        st.warning("当前筛选没有匹配记录，请调整筛选条件。")
        st.stop()

    metrics = summarize(view)
    col_total, col_risk = st.columns(2)
    col_total.metric("总记录数", metrics["总记录数"])
    col_risk.metric("心血管疾病风险率", metrics["心血管疾病风险率"])

    col_age, col_bmi = st.columns(2)
    with col_age:
        st.plotly_chart(age_histogram(view), use_container_width=True)
    with col_bmi:
        st.plotly_chart(bmi_stacked_chart(view), use_container_width=True)


if __name__ == "__main__":
    main()