"""General video captions: faster-whisper transcription + WhisperX alignment."""
from __future__ import annotations

import gc
import json
import math
import re
import tempfile
from importlib.metadata import version
from pathlib import Path

from asr_models import WHISPER_MODELS
from speech_catalog import ALIGN_MODELS, LANGUAGE_CODES, language_label
from subtitle_style import DEFAULT_FONT

PROFILE = "speech-whisperx-v1"
NO_SPACE_LANGUAGES = {"ja", "zh", "yue", "th", "lo", "my"}


def _valid_span(word: dict) -> bool:
    start, end = word.get("start"), word.get("end")
    return (isinstance(start, (int, float)) and isinstance(end, (int, float))
            and math.isfinite(start) and math.isfinite(end) and 0 <= start < end)


def _text_key(text: str) -> str:
    return re.sub(r"\s+", "", text)


def choose_words(original: dict, aligned: dict | None) -> tuple[list[dict], str]:
    """Only accept complete timed text; otherwise retain the original ASR timing."""
    if aligned is not None:
        words = aligned.get("words", [])
        if (words and all(_valid_span(w) for w in words)
                and all(a["start"] <= b["start"] for a, b in zip(words, words[1:]))
                and _text_key("".join(w["word"] for w in words)) == _text_key(original["text"])):
            return [dict(w, stage="aligned") for w in words], "aligned"
    words = original.get("words", [])
    if (words and all(_valid_span(w) for w in words)
            and _text_key("".join(w["word"] for w in words)) == _text_key(original["text"])):
        return [dict(w, stage="asr") for w in words], "asr"
    if original["text"].strip() and _valid_span(original):
        return [{"word": original["text"].strip(), "start": original["start"],
                 "end": original["end"], "stage": "asr_segment"}], "asr_segment"
    return [], "empty"


def make_cues(words: list[dict], language: str) -> list[dict]:
    """Group timed words; no time is inferred from text length."""
    cues, current = [], []
    compact = language in NO_SPACE_LANGUAGES
    max_chars = 28 if compact else 84

    def render(items):
        return ("" if compact else " ").join(w["word"].strip() for w in items).strip()

    def flush():
        if current:
            cues.append({"start": current[0]["start"], "end": max(w["end"] for w in current),
                         "text": render(current), "stage": "aligned" if all(
                             w["stage"] == "aligned" for w in current) else "asr", "words": list(current)})
            current.clear()

    for word in words:
        if not word.get("word", "").strip() or not _valid_span(word):
            continue
        if current and (word["start"] - current[-1]["end"] >= 0.65
                        or word["end"] - current[0]["start"] > 7.0
                        or len(render(current + [word])) > max_chars
                        or word["stage"] != current[-1]["stage"]):
            flush()
        current.append(word)
        if re.search(r"[。！？!?…][\"'’”）)]*$|\.[\"'’”）)]*$", word["word"].strip()):
            flush()
    flush()
    # Do not manufacture boundary positions when independent segments overlap.
    for first, second in zip(cues, cues[1:]):
        if first["end"] > second["start"]:
            first.setdefault("flags", []).append("overlapping_cues")
            second.setdefault("flags", []).append("overlapping_cues")
    return cues


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def transcribe(audio_path: Path, model_name: str, language: str, use_vad: bool) -> dict:
    from faster_whisper import WhisperModel
    import ctranslate2
    device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"    正在加载 Whisper 模型: {model_name} · {device}/{compute_type}", flush=True)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    try:
        segments, info = model.transcribe(
            str(audio_path), language=None if language == "auto" else language,
            word_timestamps=True, vad_filter=use_vad, beam_size=5,
            condition_on_previous_text=False,
        )
        rows = []
        for segment in segments:
            rows.append({"start": segment.start, "end": segment.end, "text": segment.text,
                         "words": [{"word": w.word, "start": w.start, "end": w.end,
                                    "score": w.probability} for w in (segment.words or [])]})
            progress = min(99, round(segment.end / max(info.duration, 0.001) * 100))
            print(f"视频字幕进度: 识别语音 {progress}/100", flush=True)
        return {"language": info.language, "language_probability": info.language_probability,
                "segments": rows}
    finally:
        del model
        gc.collect()


def refine(audio_path: Path, transcript: dict) -> tuple[list[dict], dict]:
    language = transcript["language"]
    model_id = ALIGN_MODELS.get(language)
    report = {"model": model_id, "aligned_segments": 0, "asr_segments": 0, "errors": []}
    segments = transcript["segments"]
    if not segments:
        return [], report
    if not model_id:
        print(f"    注意：{language_label(language)} 尚未配置对齐模型，使用 Whisper 原生词级时间。", flush=True)
        words = []
        for original in segments:
            selected, _ = choose_words(original, None)
            words.extend(selected)
        report["asr_segments"] = len(segments)
        return words, report
    import numpy as np
    import soundfile as sf
    import torch
    import whisperx
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError("语音对齐需要 16 kHz 单声道音频")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(6, torch.get_num_threads()))
    model = metadata = None
    print(f"    对齐模型: {model_id}", flush=True)
    model, metadata = whisperx.load_align_model(language, device, model_name=model_id)
    all_words = []
    try:
        for index, original in enumerate(segments):
            aligned = None
            if model is not None:
                # Include neighboring audio so a coarse ASR boundary can move.
                start = max(0.0, original["start"] - 0.5)
                end = min(len(audio) / 16000, original["end"] + 0.5)
                left, right = int(start * 16000), int(math.ceil(end * 16000))
                offset = left / 16000
                local = {"text": original["text"], "start": 0.0, "end": (right - left) / 16000}
                try:
                    if right <= left or right - left > 35 * 16000:
                        raise ValueError("识别片段超过精对齐窗口长度")
                    result = whisperx.align([local], model, metadata, np.ascontiguousarray(audio[left:right]),
                                            device, interpolate_method="ignore")
                    local_words = [dict(w) for row in result["segments"] for w in row.get("words", [])]
                    for word in local_words:
                        for key in ("start", "end"):
                            if isinstance(word.get(key), (int, float)):
                                word[key] += offset
                    aligned = {"words": local_words}
                except (ValueError, RuntimeError) as error:
                    report["errors"].append({"segment": index, "error": str(error)})
            words, stage = choose_words(original, aligned)
            if aligned is not None and stage != "aligned":
                report["errors"].append({"segment": index, "error": "incomplete_word_alignment"})
            report["aligned_segments" if stage == "aligned" else "asr_segments"] += 1
            all_words.extend(words)
            print(f"视频字幕进度: 对齐时间轴 {index + 1}/{len(segments)}", flush=True)
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    if report["asr_segments"]:
        print(f"    注意：{report['asr_segments']} 个片段保留 Whisper 原始时间，详情见对轴报告。", flush=True)
    return all_words, report


def process_video(video: Path, output: Path | None, *, helpers, model_name="large-v3", language="auto",
                  no_vad=False, no_burn=False, refresh=False, subtitle_font=DEFAULT_FONT, subtitle_size=None):
    if model_name not in WHISPER_MODELS:
        raise ValueError("视频自动字幕请选择 Whisper 模型")
    if language not in ("auto", *LANGUAGE_CODES):
        raise ValueError(f"不支持的识别语言: {language}")
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(video)
    output = Path(output) if output else video.with_name(f"{video.stem}_subtitled{video.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"profile": PROFILE, "video": helpers._file_fingerprint(video), "model": model_name,
                "language": language, "vad": not no_vad,
                "whisperx": version("whisperx"), "faster_whisper": version("faster-whisper")}
    cache = output.with_suffix(".speech.json")
    cached = cached_transcript = None
    if not refresh and cache.is_file():
        try:
            candidate = json.loads(cache.read_text(encoding="utf-8"))
            if candidate.get("metadata") == metadata:
                cached_transcript = candidate.get("transcript")
                if "words" in candidate and "alignment" in candidate:
                    cached = candidate
        except (OSError, ValueError):
            pass
    helpers._banner(f"处理: {video.name}")
    if cached is None:
        with tempfile.TemporaryDirectory(prefix="subtitle_speech_") as temp:
            audio = Path(temp) / "speech.wav"
            helpers._step(1, 5, "提取音频")
            helpers._run([helpers._get_ffmpeg_path(), "-y", "-i", str(video), "-vn", "-ar", "16000",
                          "-ac", "1", str(audio)], "提取音频")
            helpers._step(2, 5, "识别语音")
            transcript = cached_transcript or transcribe(audio, model_name, language, not no_vad)
            _write_json(cache, {"metadata": metadata, "transcript": transcript})
            print(f"    检测到语言: {language_label(transcript['language'])}", flush=True)
            helpers._step(3, 5, "对齐时间轴")
            words, alignment = refine(audio, transcript)
            cached = {"metadata": metadata, "transcript": transcript, "words": words, "alignment": alignment}
            _write_json(cache, cached)
    else:
        print("    复用视频字幕缓存", flush=True)
    helpers._step(4, 5, "生成字幕")
    cues = make_cues(cached["words"], cached["transcript"]["language"])
    if not cues:
        raise ValueError("未识别到有效语音，未生成字幕。")
    srt = output.with_suffix(".srt")
    with srt.open("w", encoding="utf-8-sig", newline="\n") as handle:
        for index, cue in enumerate(cues, 1):
            handle.write(f"{index}\n{helpers._fmt_srt_time(cue['start'])} --> "
                         f"{helpers._fmt_srt_time(cue['end'])}\n{cue['text']}\n\n")
    report = {"profile": PROFILE, "language": cached["transcript"]["language"],
              "models": {"recognizer": model_name, "aligner": cached["alignment"]["model"]},
              "summary": {"line_count": len(cues), **cached["alignment"]}, "lines": cues}
    _write_json(output.with_suffix(".alignment.json"), report)
    print(f"    SRT 文件: {srt}", flush=True)
    helpers._step(5, 5, "完成" if no_burn else "导出视频")
    if not no_burn:
        helpers.burn_subtitles(video, srt, output, subtitle_font=subtitle_font, subtitle_size=subtitle_size)
    print(f"    完成：{len(cues)} 条字幕", flush=True)
    return report


def process_batch(video_dir: Path, output_dir: Path | None, **kwargs):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir) if output_dir else video_dir / "output"
    extensions = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v"}
    videos = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not videos:
        raise ValueError("目录中没有视频文件")
    failures = []
    for video in videos:
        try:
            process_video(video, output_dir / f"{video.stem}_subtitled{video.suffix}", **kwargs)
        except Exception as error:
            failures.append(video.name)
            print(f"    错误：{video.name}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"批量处理失败 {len(failures)}/{len(videos)} 个：{', '.join(failures)}")
