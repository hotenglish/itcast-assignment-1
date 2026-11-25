"""CardioAI 模块二：心血管风险预测 API。

运行：
    F:\\software\\anaconda\\envs\\cardioenv\\python.exe module2_predictor\\app.py

启动前先确保 model 已生成：python module2_predictor\\train_and_save.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "cardio_predictor_model.pkl"

SBP_RANGE = (90, 250)
DBP_RANGE = (60, 150)
DECISION_THRESHOLD = 0.5

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


MODEL = load_model()
FEATURE_ORDER: list[str] = list(MODEL.feature_names_in_) if MODEL is not None else []


def parse_payload(payload: dict[str, Any]) -> tuple[dict[str, float], str | None]:
    missing = [name for name in FEATURE_ORDER if name not in payload]
    if missing:
        return {}, f"缺少字段：{', '.join(missing)}"

    values: dict[str, float] = {}
    for name in FEATURE_ORDER:
        try:
            values[name] = float(payload[name])
        except (TypeError, ValueError):
            return {}, f"字段 {name} 不是有效数值：{payload[name]!r}"

    ap_hi, ap_lo = values["ap_hi"], values["ap_lo"]
    if ap_lo >= ap_hi:
        return {}, "舒张压必须小于收缩压"
    if not SBP_RANGE[0] <= ap_hi <= SBP_RANGE[1]:
        return {}, f"收缩压需在 {SBP_RANGE[0]}–{SBP_RANGE[1]} mmHg 之间"
    if not DBP_RANGE[0] <= ap_lo <= DBP_RANGE[1]:
        return {}, f"舒张压需在 {DBP_RANGE[0]}–{DBP_RANGE[1]} mmHg 之间"

    return values, None


@app.get("/")
def index():
    return render_template("index.html", model_ready=MODEL is not None)


@app.post("/predict_cardio")
def predict_cardio():
    if MODEL is None:
        return jsonify({"error": "模型文件不存在，请先运行 module2_predictor/train_and_save.py"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "请求体必须是 JSON 对象"}), 400

    values, error = parse_payload(payload)
    if error is not None:
        return jsonify({"error": error}), 400

    row = pd.DataFrame([values], columns=FEATURE_ORDER)
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction = int(probability >= DECISION_THRESHOLD)

    return jsonify(
        {
            "probability": round(probability, 4),
            "prediction": prediction,
            "risk_level": "高风险" if prediction else "低风险",
        }
    )


if __name__ == "__main__":
    if MODEL is None:
        print(f"警告：模型文件不存在（{MODEL_PATH}），/predict_cardio 将返回 503。")
        print("请先运行：python module2_predictor/train_and_save.py")

    app.run(host="127.0.0.1", port=5000, debug=False)