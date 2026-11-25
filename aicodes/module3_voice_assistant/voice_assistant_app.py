"""CardioAI 模块三：语音问答助手。

DeepSeek 生成回答文本，CosyVoice 同步合成为 MP3，Base64 编码后随 JSON 返回。

运行：
    F:\\\\software\\\\anaconda\\\\envs\\\\cardioenv\\\\python.exe module3_voice_assistant\\\\voice_assistant_app.py
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import dashscope
from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_openai import ChatOpenAI

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE.parent / ".env"

LLM_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 600
LLM_TIMEOUT = 60

TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "longxiaochun_v2"
TTS_FORMAT = AudioFormat.MP3_22050HZ_MONO_256KBPS
MAX_TTS_CHARS = 1800

PORT = 5001

SYSTEM_PROMPT = """你是"CardioAI"系统中的专业心血管健康顾问。

回答规则：
1. 只围绕心血管健康、血压、血脂、血糖、体重、运动、饮食、用药依从性、就医建议等话题作答。
2. 不做疾病诊断，不开具具体处方药和剂量，不替代线下就医。需要判断病情时，明确建议就医检查。
3. 出现胸痛持续不缓解、呼吸困难、晕厥、言语不清等急症信号时，第一句就建议立即拨打急救电话。
4. 回答控制在 200 字以内，先给结论，再给 1 到 3 条具体可执行的建议。
5. 输出纯文本，不要使用 markdown 标记、列表符号、表格、代码、emoji、公式或链接，因为这段文字会被直接朗读成语音。
6. 不要重复用户的问题，不要输出"作为AI"之类的免责声明套话，结尾可以用一句话提醒遵医嘱。"""

load_dotenv(ENV_PATH, override=False)

DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY1") or "").strip()
DEEPSEEK_BASE_URL = (os.getenv("base_url1") or DEFAULT_BASE_URL).strip().strip("\"'") or DEFAULT_BASE_URL
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY

app = Flask(__name__)


def build_llm() -> ChatOpenAI | None:
    if not DEEPSEEK_API_KEY:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
        max_retries=2,
    )


LLM = build_llm()

_MARKDOWN_PATTERNS = (
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),
    (re.compile(r"(\*\*|__|\*|_)(.+?)\1"), r"\2"),
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),
    (re.compile(r"^\s*>\s?", re.M), ""),
    (re.compile(r"^\s*[-|: ]{3,}\s*$", re.M), ""),
    (re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]"), ""),
)

_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|\n+")


def strip_markdown(text: str) -> str:
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    text = text.replace("\u3000", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def count_billable_chars(text: str) -> int:
    """与 CosyVoice 计费口径一致：汉字算 2 个字符，其余算 1 个。"""
    return sum(2 if "\u4e00" <= char <= "\u9fff" else 1 for char in text)


def split_for_tts(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}" if current else sentence
        if count_billable_chars(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while count_billable_chars(sentence) > limit:
            cut = _hard_cut(sentence, limit)
            chunks.append(sentence[:cut])
            sentence = sentence[cut:]
        current = sentence
    if current:
        chunks.append(current)
    return chunks


def _hard_cut(text: str, limit: int) -> int:
    total = 0
    for index, char in enumerate(text):
        total += 2 if "\u4e00" <= char <= "\u9fff" else 1
        if total > limit:
            return max(index, 1)
    return len(text)


def synthesize(text: str) -> tuple[bytes, int]:
    chunks = split_for_tts(strip_markdown(text))
    if not chunks:
        return b"", 0

    frames: list[bytes] = []
    for chunk in chunks:
        synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE, format=TTS_FORMAT)
        audio = synthesizer.call(chunk)
        if not audio:
            response = synthesizer.get_response()
            raise RuntimeError(f"语音合成未返回音频：{response}")
        frames.append(audio)
    return b"".join(frames), len(chunks)


def ask_llm(question: str) -> str:
    if LLM is None:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY1，请检查 aicodes/.env")
    response = LLM.invoke([("system", SYSTEM_PROMPT), ("human", question)])
    answer = (response.content or "").strip()
    if not answer:
        raise RuntimeError("大模型返回了空内容")
    return answer


@app.get("/")
def index():
    return render_template("voice_index.html", llm_ready=LLM is not None, tts_ready=bool(DASHSCOPE_API_KEY))


@app.get("/health")
def health():
    return jsonify({"llm_ready": LLM is not None, "tts_ready": bool(DASHSCOPE_API_KEY)})


@app.post("/ask_voice")
def ask_voice():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "请求体必须是 JSON 对象"}), 400

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "问题不能为空"}), 400
    question = question.strip()
    if len(question) > 500:
        return jsonify({"error": "问题过长，请控制在 500 字以内"}), 400

    started = time.perf_counter()
    try:
        text_answer = ask_llm(question)
    except Exception as exc:
        return jsonify({"error": f"大模型服务异常：{exc}"}), 502
    llm_ms = round((time.perf_counter() - started) * 1000, 1)

    audio_base64: str | None = None
    tts_error: str | None = None
    tts_ms = 0.0
    chunks = 0

    if not DASHSCOPE_API_KEY:
        tts_error = "未配置 DASHSCOPE_API_KEY，请在 aicodes/.env 中填写阿里云百炼 API Key 后重试（当前仅返回文字）"
    else:
        tts_started = time.perf_counter()
        try:
            audio, chunks = synthesize(text_answer)
            if not audio:
                tts_error = "语音合成返回空音频"
            else:
                audio_base64 = base64.b64encode(audio).decode("ascii")
        except Exception as exc:
            tts_error = f"语音合成失败：{exc}"
        tts_ms = round((time.perf_counter() - tts_started) * 1000, 1)

    return jsonify(
        {
            "text_answer": text_answer,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
            "tts_error": tts_error,
            "tts_chunks": chunks,
            "elapsed_ms": {"llm": llm_ms, "tts": tts_ms},
        }
    )


if __name__ == "__main__":
    if LLM is None:
        print(f"警告：未读取到 DEEPSEEK_API_KEY1（{ENV_PATH}），/ask_voice 将返回 502。")
    if not DASHSCOPE_API_KEY:
        print(f"提示：未读取到 DASHSCOPE_API_KEY（{ENV_PATH}），语音将降级为纯文字。")
    print(f"DeepSeek base_url = {DEEPSEEK_BASE_URL}")
    app.run(host="127.0.0.1", port=PORT, debug=False)