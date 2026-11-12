"""Known Japanese lyrics aligned by WhisperX, with ASR used for localization."""
from __future__ import annotations

import gc
import json
import math
import unicodedata
from dataclasses import asdict
from pathlib import Path

from speech_catalog import ALIGN_MODELS
from speech_subtitles import _valid_span

PROFILE = "mv-whisperx-lyrics-v1"


def source_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if not unicodedata.category(c).startswith(("P", "S", "Z"))
                   and not c.isspace())


def metadata(transcript_metadata: dict, lyrics: list) -> dict:
    return {**transcript_metadata, "profile": PROFILE, "whisperx": "3.8.6",
            "align_model": ALIGN_MODELS["ja"], "source_lines": [line.ja for line in lyrics]}


def load_cache(path: Path, expected: dict, lyrics: list, helpers):
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))["whisperx"]
        records = saved["items"]
        if saved["metadata"] != expected or len(records) != len(lyrics):
            return None
        if not all(_valid_span(row) for row in records):
            return None
        items = [helpers.AlignmentItem(**row) for row in records]
        for item, line in zip(items, lyrics):
            item.text, item.source_text = line.zh, line.ja
        return items, saved
    except (OSError, ValueError, KeyError, TypeError):
        return None


def plan_windows(anchors: list, duration: float) -> list[dict]:
    """Unmatched runs use neighboring observed anchors, never interpolated times."""
    usable = [item.status in {"matched", "low_confidence"} and item.score is not None
              and _valid_span(asdict(item)) for item in anchors]
    windows = []
    index = 0
    while index < len(anchors):
        if usable[index]:
            first = last = index
            keep = [index]
            start, end = anchors[index].start, anchors[index].end
            index += 1
        else:
            first = index
            while index < len(anchors) and not usable[index]:
                index += 1
            last = index - 1
            keep = list(range(first, last + 1))
            first = first - 1 if first else first
            last = index if index < len(anchors) else last
            start = anchors[first].start if usable[first] else 0.0
            end = anchors[last].end if usable[last] else duration
        windows.append({"start": max(0.0, start - 0.5), "end": min(duration, end + 0.5),
                        "context": list(range(first, last + 1)), "keep": keep})
    return windows


def collect_lines(result: dict, texts: list[str], line_ids: list[int], offset: float) -> dict:
    chars = [char for row in result.get("segments", []) for char in row.get("chars", [])
             if not char["char"].isspace()]
    if "".join(c["char"] for c in chars) != "".join(texts):
        raise ValueError("WhisperX 返回的字符与输入歌词不一致")
    rows, cursor = {}, 0
    for line_id, text in zip(line_ids, texts):
        owned = chars[cursor:cursor + len(text)]
        cursor += len(text)
        valid = [char for char in owned if _valid_span(char)]
        coverage = len(valid) / len(owned) if owned else 0.0
        if not owned or len(valid) != len(owned):
            rows[line_id] = {"start": None, "end": None, "coverage": coverage,
                             "reason": "incomplete_character_alignment"}
            continue
        rows[line_id] = {"start": offset + valid[0]["start"], "end": offset + valid[-1]["end"],
                         "coverage": coverage,
                         "acoustic_score": sum(c.get("score", 0) for c in valid) / len(valid)}
    return rows


def align_lyrics(audio_path: Path, segments: list, lyrics: list, min_match_score: float, helpers):
    import numpy as np
    import soundfile as sf
    import torch
    import whisperx

    if not lyrics or any(not line.ja.strip() for line in lyrics):
        raise ValueError("MV WhisperX 对齐需要逐行对应的日文原词和中文字幕")
    clean = [source_text(line.ja) for line in lyrics]
    if not all(clean):
        raise ValueError("歌词中有不包含可对齐文字的行")
    words = helpers._flatten_words(segments)
    anchors = (helpers._align_with_text(words, lyrics, min_match_score, strict_anchors=True, fill_missing=False)
               if words else [helpers.AlignmentItem(None, None, line.zh, source_text=line.ja, status="unresolved")
                              for line in lyrics])
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError("MV WhisperX 对齐需要 16 kHz 单声道音频")
    duration = len(audio) / sample_rate
    windows = plan_windows(anchors, duration)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(6, torch.get_num_threads()))
    print(f"    对齐模型: {ALIGN_MODELS['ja']}", flush=True)
    model, model_metadata = whisperx.load_align_model("ja", device, model_name=ALIGN_MODELS["ja"])
    rows, diagnostics = {}, []
    try:
        for index, window in enumerate(windows):
            record = dict(window)
            try:
                left, right = int(window["start"] * sample_rate), min(len(audio), math.ceil(window["end"] * sample_rate))
                if right <= left or (right - left) / sample_rate > 60:
                    raise ValueError("缺少可靠歌词定位，连续音频窗口超过 60 秒")
                texts = [clean[line_id] for line_id in window["context"]]
                local = {"start": 0.0, "end": (right - left) / sample_rate, "text": " ".join(texts)}
                aligned = whisperx.align([local], model, model_metadata, np.ascontiguousarray(audio[left:right]),
                                        device, interpolate_method="ignore", return_char_alignments=True)
                selected = collect_lines(aligned, texts, window["context"], left / sample_rate)
                for line_id in window["keep"]:
                    rows[line_id] = selected[line_id]
            except (ValueError, RuntimeError) as error:
                record["error"] = str(error)
            diagnostics.append(record)
            print(f"MV 对齐进度: {index + 1}/{len(windows)}", flush=True)
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    items, unresolved = [], []
    for index, (line, anchor) in enumerate(zip(lyrics, anchors)):
        row = rows.get(index, {})
        if _valid_span(row):
            item = helpers.AlignmentItem(row["start"], row["end"], line.zh, source_text=line.ja,
                                         recognized_text=anchor.recognized_text, score=anchor.score, status="whisperx")
        elif anchor.status in {"matched", "low_confidence"} and _valid_span(asdict(anchor)):
            item = helpers.AlignmentItem(anchor.start, anchor.end, line.zh, source_text=line.ja,
                                         recognized_text=anchor.recognized_text, score=anchor.score, status="asr_anchor",
                                         review_reason="WhisperX 未完整定位，保留 ASR 文本锚点时间")
        else:
            item = helpers.AlignmentItem(None, None, line.zh, source_text=line.ja, status="unresolved",
                                         review_reason="WhisperX 未定位到这一行")
            unresolved.append(index + 1)
        items.append(item)
    for first, second in zip(items, items[1:]):
        if first.end is not None and second.start is not None and first.end > second.start + 0.05:
            first.review_reason = second.review_reason = "相邻歌词对齐区间重叠"
    backend = {"profile": PROFILE, "model": ALIGN_MODELS["ja"], "windows": diagnostics,
               "line_details": [{"line": i + 1, **rows.get(i, {})} for i in range(len(lyrics))],
               "aligned_lines": sum(item.status == "whisperx" for item in items),
               "asr_anchor_lines": sum(item.status == "asr_anchor" for item in items),
               "unresolved_lines": unresolved}
    return items, backend
