"""Whisper 模型和歌词声学对齐的界面标识。"""

DEFAULT_MODEL = "large-v3"
WHISPER_MODELS = ("large-v3", "large-v3-turbo", "medium", "small")
LYRIC_MODEL = "lyrics-mms-tifa"
SUPPORTED_MODELS = (*WHISPER_MODELS, LYRIC_MODEL)


def model_label(model: str) -> str:
    if model == LYRIC_MODEL:
        return "歌词声学对齐"
    return f"Whisper {model}"
