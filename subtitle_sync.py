#!/usr/bin/env python3
"""
subtitle_sync.py
================
日语歌曲中文字幕自动对轴工具  v2.0

工作流程:
  1. FFmpeg 提取音频
  2. Demucs 人声分离（去掉背景音乐，提升 Whisper 准确率）
  3. Whisper 识别日语，或用中日双语歌词进行声学对齐
  4. 智能对轴：
     - 若有日文歌词：fuzzy matching（rapidfuzz）日→日匹配，再映射到中文
     - 若仅中文歌词：优先使用英文锚点局部对齐，否则低置信度时间回退
  5. FFmpeg 将硬字幕烧入原视频

歌词文件格式（UTF-8）:
  中日双语模式：
    世界の終わりで|世界的终结
    生まれた光|诞生的光芒
    注：分隔符支持 | 或 Tab

用法:
  单文件:
    python subtitle_sync.py --video song.mp4 --lyrics song.txt
    python subtitle_sync.py --video song.mp4 --lyrics song.txt --output out.mp4

  批量处理 (视频与歌词文件同名，扩展名不同):
    python subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics
    python subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics --output-dir ./output
"""

import argparse
import json
import os
import re
import sys
import subprocess
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from asr_models import LYRIC_MODEL, SUPPORTED_MODELS, model_label
from subtitle_style import DEFAULT_FONT, FONT_PRESETS, apply_ass_style, font_settings, subtitle_filter
# 强制禁止创建软链接，绕过 Windows 恶心的权限限制
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ── Windows 终端 UTF-8 修复 ────────────────────────────────────────────────────
# Windows 默认终端编码可能是 cp932（日文）或 cp936（中文），强制改为 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TRANSCRIPT_CACHE_SCHEMA = 2
OCR_CACHE_SCHEMA = 1
DEFAULT_MIN_MATCH_SCORE = 55.0
DEFAULT_OCR_MATCH_SCORE = 82.0
DEFAULT_OCR_SAMPLE_INTERVAL = 0.5
DEFAULT_WHISPER_MODEL = "large-v3"
DEMUCS_PROFILE = "htdemucs-shift1-seed0-v1"
# Demucs' test-time shifts use Python's random generator. Seed inside the
# child process (seeding the GUI/parent would not affect a fresh interpreter).
DEMUCS_LAUNCHER = (
    "import random, runpy; import numpy as np; import torch; "
    "random.seed(0); np.random.seed(0); torch.manual_seed(0); "
    "runpy.run_module('demucs', run_name='__main__')"
)
SUPPORTED_WHISPER_MODELS = (
    "large-v3",
    "large-v3-turbo",
    "medium",
    "small",
)


@dataclass(frozen=True)
class LyricLine:
    """一行源日文歌词及其对应的中文字幕。"""

    ja: str
    zh: str


@dataclass(frozen=True)
class WordTiming:
    """与 faster-whisper 解耦的逐词时间对象。"""

    start: float
    end: float
    text: str
    probability: Optional[float] = None


@dataclass
class SimpleSegment:
    """可序列化、可从缓存恢复的 Whisper 片段。"""

    start: float
    end: float
    text: str
    words: list[WordTiming]


@dataclass
class AlignmentItem:
    """一行字幕的可解释对轴结果。"""

    start: float
    end: float
    text: str
    source_text: str = ""
    recognized_text: str = ""
    score: Optional[float] = None
    status: str = "timing_fallback"
    word_probability: Optional[float] = None
    ocr_text: str = ""
    ocr_score: Optional[float] = None
    ocr_shift: Optional[float] = None
    ocr_target_coverage: Optional[float] = None
    review_reason: str = ""


@dataclass(frozen=True)
class OcrFrame:
    """一次画面采样中识别到的文字。"""

    time: float
    texts: tuple[str, ...]
    confidences: tuple[float, ...]


@dataclass(frozen=True)
class OcrLineAnchor:
    """OCR 文字与一行歌词之间的候选时间锚点。"""

    line_index: int
    time: float
    delta: float
    source_text: str
    recognized_text: str
    score: float
    target_fraction: float
    target_coverage: float = 1.0
    visual_start: Optional[float] = None
    visual_end: Optional[float] = None
    sample_count: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 1: 音频提取
# ──────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: Path, output_wav: Path) -> None:
    """用 FFmpeg 从视频中提取音频（44100Hz 立体声 WAV）。"""
    ffmpeg_exe = _get_ffmpeg_path()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", "44100",
        "-ac", "2",
        "-f", "wav",
        str(output_wav),
    ]
    _run(cmd, "FFmpeg 提取音频")


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 2: 人声分离
# ──────────────────────────────────────────────────────────────────────────────

def separate_vocals(audio_path: Path, work_dir: Path, *, deterministic: bool = False) -> Path:
    """
    用 Demucs htdemucs 模型分离人声。
    返回人声 WAV 文件路径。
    """
    # 确保 ffmpeg 目录在环境变量 PATH 中，否则 Demucs 内部会找不到 ffmpeg
    ffmpeg_exe = _get_ffmpeg_path()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    # Fixed shifts are opt-in for controlled experiments. The application keeps
    # its existing default until the changed audio passes the timing regressions.
    entrypoint = (["-c", DEMUCS_LAUNCHER, "-n", "htdemucs", "--shifts", "1"]
                  if deterministic else ["-m", "demucs"])
    cmd = [
        sys.executable, *entrypoint,
        "--two-stems=vocals",
        "--out", str(work_dir),
        str(audio_path),
    ]
    _run(cmd, "Demucs 人声分离")

    stem = audio_path.stem
    vocals_path = work_dir / "htdemucs" / stem / "vocals.wav"

    if not vocals_path.exists():
        # 尝试查找实际路径（模型名可能不同）
        candidates = list(work_dir.rglob("vocals.wav"))
        if candidates:
            vocals_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Demucs 完成但未找到 vocals.wav，请检查 {work_dir}"
            )

    return vocals_path


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 3: Whisper 日语识别
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    use_vad: bool = True,
) -> list:
    """
    用选定的语音模型识别日语，返回统一的词级时间片段。

    参数:
      audio_path:     人声 WAV 文件路径
      model_name:     语音模型标识
      use_vad:        是否启用 VAD 静音过滤

    返回:
      片段列表，每个元素有 .start, .end, .text, .words（词级时间戳）
    """
    if model_name not in SUPPORTED_WHISPER_MODELS:
        raise ValueError(f"不支持的 Whisper 模型: {model_name}")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        _die("未安装 faster-whisper，请运行: pip install faster-whisper")

    print(f"    正在加载 Whisper 模型: {model_name}...")

    device, compute_type = _detect_device()
    print(f"    使用设备: {device} / {compute_type}")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    # ── 构建转录参数 ──
    transcribe_kwargs = {
        "language": "ja",
        "beam_size": 5,
        "word_timestamps": True,              # 提供词级时间，供后续歌词定位使用
        "condition_on_previous_text": False,  # ★ 歌曲各行独立，不要用上文"脑补"
        "no_speech_threshold": 0.4,           # 降低无声判定的阈值（歌声可能被误判为无声）
        "compression_ratio_threshold": 2.4,   # 放宽压缩比阈值（音乐场景下容忍度更高）
    }

    if use_vad:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=500,       # 避免把短换气误切成新的歌词片段
            min_speech_duration_ms=100,        # 最短语音段 100ms（避免吞掉短促音节）
            threshold=0.35,                    # VAD 灵敏度（0.0-1.0，越低越敏感）
            speech_pad_ms=150,                 # 为歌声音头和尾音保留余量
        )
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_gen, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    segments = []
    for i, seg in enumerate(segments_gen, 1):
        segments.append(seg)
        word_info = ""
        if hasattr(seg, "words") and seg.words:
            word_info = f"  [words: {len(seg.words)}]"
        print(
            f"      [{i:03d}] {seg.start:.2f}s → {seg.end:.2f}s"
            f"{word_info}  {seg.text.strip()}",
            flush=True,
        )
    print(f"    Whisper 识别完成，共 {len(segments)} 个片段", flush=True)

    return segments


def _detect_device():
    """检测可用设备，返回 (device, compute_type)。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 4: 时间轴映射
# ──────────────────────────────────────────────────────────────────────────────
#
# ★ 核心原则：一行歌词 = 一个字幕轴
#   用户提供的歌词文件中每一行中文对应一个字幕条目。
#   有日文原文时，以 Whisper 识别文本做顺序受限匹配；无日文时才回退到
#   时间比例分配。识别数量变化不会再直接造成整首歌词全局错位。
# ──────────────────────────────────────────────────────────────────────────────

def load_lyrics(lyrics_path: Path, require_pairs: bool = False) -> tuple[str, list[LyricLine]]:
    """
    从 UTF-8 文本文件读取歌词，自动识别三种格式。

    返回:
      (ja_prompt_text, lyric_lines)
      - ja_prompt_text: 日文歌词拼接文本，无日文时为空串
      - lyric_lines:     保留日文与中文对应关系的歌词行

    格式 1 — 逐行双语（| 或 Tab 分隔）：
      世界の終わりで|世界的终结
      生まれた光|诞生的光芒

    格式 2 — 中/日分块（先中文段，后日文段）：
      世界第一的王子殿下
      这样的待遇你要铭记在心哦
      ...
      世界で一番おうじさま
      そういう扱い 心得てよね
      ...
      ★ 中文段行数 ≠ 日文段行数时：中文全保留，日文仅作 prompt

    格式 3 — 纯中文（向后兼容）：
      世界的终结
      诞生的光芒
    """
    with open(lyrics_path, "r", encoding="utf-8") as f:
        physical_lines = [ln.rstrip("\r\n") for ln in f]
    raw_lines = [ln.strip() for ln in physical_lines if ln.strip()]

    if not raw_lines:
        _die("歌词文件为空")

    # ── 格式 1: 逐行分隔符（| 或 Tab）──
    has_separator = any("|" in ln or "\t" in ln for ln in raw_lines)
    if has_separator:
        ja_list, zh_list = _parse_line_bilingual(raw_lines)
        if require_pairs and not all(ja.strip() and zh.strip() for ja, zh in zip(ja_list, zh_list)):
            raise ValueError("歌词声学对齐需要每行日文原词及其对应中文，请检查缺失的行")
        ja_prompt = "。".join(j for j in ja_list if j)
        print(f"    检测到逐行双语歌词：中文 {len(zh_list)} 行")
        return ja_prompt, [LyricLine(ja, zh) for ja, zh in zip(ja_list, zh_list)]

    # ── 格式 2: 双空行显式分隔中文块 / 日文块 ──
    # 单个空行常用于歌词分段，因此只把连续两个及以上空行当作块分隔符。
    explicit_blocks = _split_at_double_blank_line(physical_lines)
    if explicit_blocks is not None:
        zh_lines, ja_lines = explicit_blocks
        ja_prompt = "。".join(ja_lines)
        print(
            f"    检测到双空行分隔的中/日歌词："
            f"中文 {len(zh_lines)} 行 + 日文 {len(ja_lines)} 行"
        )
        if len(zh_lines) != len(ja_lines):
            if require_pairs:
                raise ValueError(f"中/日歌词行数不一致：中文 {len(zh_lines)} 行，日文 {len(ja_lines)} 行")
            print(f"    ⚠️  中/日行数不一致，请检查双空行分隔位置。")
        lyric_lines = [
            LyricLine(ja_lines[i] if i < len(ja_lines) else "", zh)
            for i, zh in enumerate(zh_lines)
        ]
        return ja_prompt, lyric_lines

    # ── 格式 2 vs 3: 假名检测 ──
    kana_flags = [_has_japanese_kana(ln) for ln in raw_lines]
    kana_count = sum(kana_flags)

    if kana_count == 0:
        if require_pairs:
            raise ValueError("歌词声学对齐需要中日双语歌词：先中文整块，空两行，再日文整块")
        # 纯中文
        print(f"    检测到纯中文歌词，共 {len(raw_lines)} 行")
        return "", [LyricLine("", line) for line in raw_lines]

    # 有假名 → 找分块分割点
    split_idx = kana_flags.index(True)

    # 验证分块格式的有效性
    if split_idx > 0 and any(kana_flags[:split_idx]):
        print(f"    检测到纯中文歌词（含少量假名标注），共 {len(raw_lines)} 行")
        return "", [LyricLine("", line) for line in raw_lines]

    after_kana = kana_flags[split_idx:]
    after_ratio = sum(after_kana) / len(after_kana) if after_kana else 0
    if after_ratio < 0.5:
        print(f"    检测到纯中文歌词，共 {len(raw_lines)} 行")
        return "", [LyricLine("", line) for line in raw_lines]

    # ── 确认是分块格式 ──
    zh_lines = raw_lines[:split_idx]   # ★ 全部中文行，不截断
    ja_lines = raw_lines[split_idx:]   # 日文行（仅用于 prompt）

    ja_prompt = "。".join(j for j in ja_lines if j)
    print(f"    检测到中/日分块歌词：中文 {len(zh_lines)} 行 + 日文 {len(ja_lines)} 行")

    if len(zh_lines) != len(ja_lines):
        if require_pairs:
            raise ValueError(f"中/日歌词行数不一致：中文 {len(zh_lines)} 行，日文 {len(ja_lines)} 行；请用双空行分隔")
        print(f"    💡 中/日行数不一致，日文仅用于辅助 Whisper 识别，"
              f"最终字幕 = {len(zh_lines)} 行中文")

    lyric_lines = [
        LyricLine(ja_lines[i] if i < len(ja_lines) else "", zh)
        for i, zh in enumerate(zh_lines)
    ]
    return ja_prompt, lyric_lines


def _split_at_double_blank_line(
    physical_lines: list[str],
) -> Optional[tuple[list[str], list[str]]]:
    """用连续两个及以上空行分隔中文整块与日文整块。"""
    index = 0
    while index < len(physical_lines):
        if physical_lines[index].strip():
            index += 1
            continue

        gap_start = index
        while index < len(physical_lines) and not physical_lines[index].strip():
            index += 1

        if index - gap_start < 2:
            continue

        left = [line.strip() for line in physical_lines[:gap_start] if line.strip()]
        right = [line.strip() for line in physical_lines[index:] if line.strip()]
        if left and right:
            return left, right

    return None


def _has_japanese_kana(text: str) -> bool:
    """检测文本中是否包含日文假名（平假名/片假名）。"""
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F:   # 平假名
            return True
        if 0x30A0 <= cp <= 0x30FF:   # 片假名
            return True
    return False


def _parse_line_bilingual(raw_lines: list[str]) -> tuple[list[str], list[str]]:
    """解析逐行双语格式（| 或 Tab 分隔）。"""
    ja_lyrics = []
    zh_lyrics = []
    for line in raw_lines:
        if "|" in line:
            parts = line.split("|", 1)
        elif "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = ["", line]  # 混合格式中的无分隔行按纯中文保留
        ja = parts[0].strip()
        zh = parts[1].strip() if len(parts) > 1 else ja
        ja_lyrics.append(ja)
        zh_lyrics.append(zh)
    return ja_lyrics, zh_lyrics


def align_lyrics_to_segments(
    segments: list,
    lyric_lines: list[LyricLine],
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    strict_anchors: bool = False,
) -> list[AlignmentItem]:
    """
    使用日文歌词与 Whisper 识别文本进行单调动态规划匹配。

    有日文原文时，匹配逐词文本并将高分结果作为时间锚点；漏识别行只在
    相邻锚点之间局部补间。没有日文原文时，优先提取歌词中的英文/数字
    作为弱锚点；可用锚点不足时才保留时间比例算法作为低置信度回退。
    """
    lyric_lines = [
        line if isinstance(line, LyricLine) else LyricLine("", str(line))
        for line in lyric_lines
    ]
    if not lyric_lines:
        _die("歌词文件没有可用的中文字幕行。")

    words = _flatten_words(segments)
    japanese_count = sum(bool(_normalize_japanese(line.ja)) for line in lyric_lines)

    if words and japanese_count:
        print(f"    对轴模式: 日文文本动态规划匹配（可匹配 {japanese_count}/{len(lyric_lines)} 行）")
        return _align_with_text(words, lyric_lines, min_match_score, strict_anchors=strict_anchors)

    if words:
        latin_aligned = _align_with_latin_anchors(
            words, lyric_lines, min_match_score, strict_anchors=strict_anchors,
        )
        if latin_aligned is not None:
            return latin_aligned

    print("    对轴模式: 仅时间比例回退（无可用于文本匹配的日文歌词）")
    return _align_by_timing(segments, lyric_lines)


def _flatten_words(segments: list) -> list[WordTiming]:
    """把 Whisper 片段展平为按时间排序的逐词列表。"""
    words: list[WordTiming] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None) or []
        if segment_words:
            for word in segment_words:
                text = getattr(word, "word", getattr(word, "text", "")).strip()
                if not text:
                    continue
                words.append(WordTiming(
                    float(word.start),
                    float(word.end),
                    text,
                    getattr(word, "probability", None),
                ))
        else:
            text = getattr(segment, "text", "").strip()
            if text:
                words.append(WordTiming(
                    float(segment.start),
                    float(segment.end),
                    text,
                    None,
                ))
    return sorted(words, key=lambda item: (item.start, item.end))


def _normalize_japanese(text: str) -> str:
    """统一全半角、片假名/平假名并移除标点空白。"""
    normalized = unicodedata.normalize("NFKC", text).lower()
    chars = []
    for char in normalized:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            char = chr(codepoint - 0x60)
        chars.append(char)
    return re.sub(r"[\W_]+", "", "".join(chars), flags=re.UNICODE)


def _similarity(left: str, right: str) -> float:
    """优先使用 RapidFuzz；测试环境未安装时回退到标准库。"""
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio
        return float(ratio(left, right))
    except ImportError:
        return SequenceMatcher(None, left, right).ratio() * 100.0


def _ocr_text_candidates(frame: OcrFrame) -> list[tuple[str, float]]:
    """组合同一画面中相邻 OCR 文本，兼容被拆成单字的竖排歌词。"""
    items = [
        (text.strip(), float(confidence))
        for text, confidence in zip(frame.texts, frame.confidences)
        if _normalize_japanese(text)
    ]
    candidates: dict[str, tuple[str, float]] = {}
    for start in range(len(items)):
        parts: list[str] = []
        confidences: list[float] = []
        for end in range(start, min(len(items), start + 5)):
            text, confidence = items[end]
            parts.append(text)
            confidences.append(confidence)
            combined = "".join(parts)
            normalized = _normalize_japanese(combined)
            if len(normalized) < 3:
                continue
            average_confidence = sum(confidences) / len(confidences)
            existing = candidates.get(normalized)
            if existing is None or average_confidence > existing[1]:
                candidates[normalized] = (combined, average_confidence)
    return list(candidates.values())


def _score_ocr_candidate(
    target: str,
    observed: str,
    confidence: float,
) -> tuple[float, float, float]:
    """返回 OCR 匹配分、片段中心位置和目标歌词覆盖率。"""
    normalized_target = _normalize_japanese(target)
    normalized_observed = _normalize_japanese(observed)
    if len(normalized_target) < 3 or len(normalized_observed) < 3:
        return 0.0, 0.5, 0.0

    matcher = SequenceMatcher(None, normalized_target, normalized_observed)
    block = max(matcher.get_matching_blocks(), key=lambda item: item.size)
    if block.size < 3:
        return 0.0, 0.5, 0.0

    try:
        from rapidfuzz.fuzz import partial_ratio
        partial_score = float(partial_ratio(normalized_target, normalized_observed))
    except ImportError:
        partial_score = matcher.ratio() * 100.0

    overlap_score = block.size / min(
        len(normalized_target),
        len(normalized_observed),
    ) * 100.0
    length_factor = 0.75 + 0.25 * min(1.0, block.size / 8.0)
    confidence_factor = 0.90 + 0.10 * max(0.0, min(1.0, confidence))
    score = (
        partial_score * 0.72 + overlap_score * 0.28
    ) * length_factor * confidence_factor
    target_fraction = (
        block.a + block.size / 2.0
    ) / len(normalized_target)
    target_coverage = block.size / len(normalized_target)
    return (
        round(score, 2),
        max(0.0, min(1.0, target_fraction)),
        max(0.0, min(1.0, target_coverage)),
    )


def _match_ocr_frame(
    target: str,
    frame: OcrFrame,
) -> tuple[float, str, float, float]:
    """找出一帧 OCR 结果中与目标歌词最接近的连续文本。"""
    best = (0.0, "", 0.5, 0.0)
    for candidate, confidence in _ocr_text_candidates(frame):
        score, target_fraction, target_coverage = _score_ocr_candidate(
            target,
            candidate,
            confidence,
        )
        if score > best[0]:
            best = (score, candidate, target_fraction, target_coverage)
    return best


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _duration_profile(
    baseline: list[AlignmentItem],
    line_index: int,
) -> tuple[float, float, float]:
    """用邻近行的中位数与 MAD 给出正常时长和异常值边界。"""
    durations = [
        max(0.08, item.end - item.start)
        for item in baseline
    ]
    if not durations:
        return 2.0, 0.5, 4.0

    peers = [
        duration
        for index, duration in enumerate(durations)
        if index != line_index and abs(index - line_index) <= 4
    ]
    if len(peers) < 3:
        peers = [
            duration
            for index, duration in enumerate(durations)
            if index != line_index
        ]
    if not peers:
        typical = durations[line_index]
        return typical, max(0.08, typical / 2.0), typical * 2.0

    typical = _median(peers)
    mad = _median([abs(duration - typical) for duration in peers])
    tolerance = max(
        typical * 0.5,
        3.0 * 1.4826 * mad,
        DEFAULT_OCR_SAMPLE_INTERVAL * 2.0,
    )
    return (
        typical,
        max(0.08, typical - tolerance),
        typical + tolerance,
    )


def _is_wide_duration_outlier(
    item: AlignmentItem,
    anchor: OcrLineAnchor,
    duration_profile: tuple[float, float, float],
) -> bool:
    """只接受统计与画面跨度同时证明的异常长区间。"""
    if anchor.visual_start is None or anchor.visual_end is None:
        return False
    typical_duration, _lower_duration, upper_duration = duration_profile
    baseline_duration = item.end - item.start
    visual_duration = anchor.visual_end - anchor.visual_start
    supported_duration = max(typical_duration, visual_duration)
    return (
        baseline_duration > upper_duration
        and baseline_duration >= supported_duration * 2.0
    )


def _classify_ocr_anchor_conflicts(
    anchors: list[OcrLineAnchor],
    sample_interval: float,
) -> tuple[set[int], set[int]]:
    """区分确定的时间倒序与可能合法的同屏重复句。"""
    hard_conflicts: set[int] = set()
    for left, right in zip(anchors, anchors[1:]):
        if right.time < left.time:
            hard_conflicts.update((left.line_index, right.line_index))

    repeated_frame_conflicts: set[int] = set()
    last_by_lyric: dict[str, OcrLineAnchor] = {}
    for anchor in anchors:
        normalized = _normalize_japanese(anchor.source_text)
        previous = last_by_lyric.get(normalized)
        if previous and anchor.time - previous.time < sample_interval:
            repeated_frame_conflicts.update((previous.line_index, anchor.line_index))
        last_by_lyric[normalized] = anchor
    return hard_conflicts, repeated_frame_conflicts


def build_ocr_line_anchors(
    frames: list[OcrFrame],
    lyric_lines: list[LyricLine],
    baseline: list[AlignmentItem],
    min_match_score: float = DEFAULT_OCR_MATCH_SCORE,
    max_shift_seconds: float = 6.0,
) -> list[OcrLineAnchor]:
    """在 Whisper 粗轴附近寻找顺序明确的画面歌词锚点。"""
    if len(lyric_lines) != len(baseline):
        raise ValueError("OCR 辅助要求歌词行数与基础时间轴一致")

    frame_intervals = [
        frames[index].time - frames[index - 1].time
        for index in range(1, len(frames))
        if frames[index].time > frames[index - 1].time
    ]
    sample_interval = (
        _median(frame_intervals)
        if frame_intervals
        else DEFAULT_OCR_SAMPLE_INTERVAL
    )

    anchors: list[OcrLineAnchor] = []
    for line_index, (line, item) in enumerate(zip(lyric_lines, baseline)):
        # 尖括号通常是和声或画面附注，不把它抢到的高分当成主句锚点。
        target = re.sub(r"<[^>]*>", "", line.ja or line.zh).strip()
        if len(_normalize_japanese(target)) < 3:
            continue

        duration = max(0.2, item.end - item.start)
        typical_duration, _lower_duration, upper_duration = _duration_profile(
            baseline,
            line_index,
        )
        # 只有远超统计上界的粗轴才放宽搜索；普通长句仍沿用局部时间先验。
        anomalously_long = duration > upper_duration + typical_duration
        candidates = []
        for frame in frames:
            if frame.time < item.start - max_shift_seconds:
                continue
            if frame.time > item.end + max_shift_seconds:
                break
            score, recognized, target_fraction, target_coverage = _match_ocr_frame(
                target,
                frame,
            )
            if score < min_match_score:
                continue
            expected_time = item.start + duration * target_fraction
            delta = frame.time - expected_time
            if abs(delta) > max_shift_seconds and not anomalously_long:
                continue
            rank = score - abs(delta) * 1.5
            candidates.append({
                "frame": frame,
                "score": score,
                "recognized": recognized,
                "target_fraction": target_fraction,
                "target_coverage": target_coverage,
                "delta": delta,
                "rank": rank,
            })

        if not candidates:
            continue

        groups: list[list[dict]] = []
        for candidate in candidates:
            if (
                not groups
                or candidate["frame"].time - groups[-1][-1]["frame"].time
                > sample_interval * 1.55
            ):
                groups.append([candidate])
            else:
                groups[-1].append(candidate)
        def group_rank(group: list[dict]) -> float:
            quality = (
                max(candidate["score"] for candidate in group)
                + min(len(group), 10) * 4.0
            )
            if anomalously_long:
                visual_start = group[0]["frame"].time - sample_interval / 2.0
                visual_end = group[-1]["frame"].time + sample_interval / 2.0
                edge_distance = min(
                    abs(visual_start - item.start),
                    abs(visual_end - item.end),
                )
                return quality - edge_distance
            return quality - min(abs(candidate["delta"]) for candidate in group)

        best_group = max(groups, key=group_rank)
        best = max(best_group, key=lambda candidate: candidate["rank"])
        nearby = [
            candidate
            for candidate in best_group
            if abs(candidate["frame"].time - best["frame"].time) <= 1.25
            and abs(candidate["delta"] - best["delta"]) <= 1.5
            and candidate["score"] >= best["score"] - 8.0
        ]
        weights = [
            max(1.0, candidate["score"] - min_match_score + 1.0) ** 2
            for candidate in nearby
        ]
        weight_sum = sum(weights)
        weighted_time = sum(
            candidate["frame"].time * weight
            for candidate, weight in zip(nearby, weights)
        ) / weight_sum
        weighted_delta = sum(
            candidate["delta"] * weight
            for candidate, weight in zip(nearby, weights)
        ) / weight_sum
        anchors.append(OcrLineAnchor(
            line_index=line_index,
            time=round(weighted_time, 3),
            delta=round(weighted_delta, 3),
            source_text=target,
            recognized_text=best["recognized"],
            score=round(best["score"], 2),
            target_fraction=round(best["target_fraction"], 4),
            target_coverage=round(best["target_coverage"], 4),
            visual_start=(
                round(best_group[0]["frame"].time - sample_interval / 2.0, 3)
                if len(best_group) >= 2
                else None
            ),
            visual_end=(
                round(best_group[-1]["frame"].time + sample_interval / 2.0, 3)
                if len(best_group) >= 2
                else None
            ),
            sample_count=len(best_group),
        ))
    return anchors


def apply_ocr_timing_assist(
    baseline: list[AlignmentItem],
    anchors: list[OcrLineAnchor],
) -> list[AlignmentItem]:
    """用 OCR 估计的局部漂移移动边界，同时保留原字幕间隔。"""
    line_count = len(baseline)
    minimum_anchors = min(line_count, max(3, (line_count + 9) // 10))
    if line_count == 0 or len(anchors) < minimum_anchors:
        print(f"    OCR 可靠锚点不足：{len(anchors)}/{line_count}。")
        return baseline

    anchor_map = {anchor.line_index: anchor for anchor in anchors}
    hard_conflicts, repeated_frame_conflicts = _classify_ocr_anchor_conflicts(
        anchors,
        DEFAULT_OCR_SAMPLE_INTERVAL,
    )
    if hard_conflicts:
        print("    OCR 锚点时间倒序。")
        return [
            replace(
                item,
                ocr_text=anchor_map[index].recognized_text if index in anchor_map else "",
                ocr_score=anchor_map[index].score if index in anchor_map else None,
                ocr_shift=0.0 if index in anchor_map else None,
                ocr_target_coverage=(
                    anchor_map[index].target_coverage if index in anchor_map else None
                ),
                review_reason=(
                    "ocr_sequence_conflict" if index in hard_conflicts else ""
                ),
            )
            for index, item in enumerate(baseline)
        ]
    review_reasons: dict[int, str] = {
        index: "ocr_repeated_frame"
        for index in repeated_frame_conflicts
    }
    duration_profiles = [
        _duration_profile(baseline, index)
        for index in range(line_count)
    ]
    partial_directions: dict[int, str] = {}
    anchor_shifts: dict[int, float] = {}
    for index, anchor in anchor_map.items():
        item = baseline[index]
        _typical_duration, lower_duration, _upper_duration = duration_profiles[index]
        baseline_duration = item.end - item.start
        has_visual_span = (
            anchor.visual_start is not None and anchor.visual_end is not None
        )
        if _is_wide_duration_outlier(
            item,
            anchor,
            duration_profiles[index],
        ):
            start_distance = abs(anchor.visual_start - item.start)
            end_distance = abs(anchor.visual_end - item.end)
            anchor_shifts[index] = (
                anchor.visual_start - item.start
                if start_distance <= end_distance
                else anchor.visual_end - item.end
            )
            continue
        if (
            has_visual_span
            and baseline_duration < lower_duration
            and anchor.target_coverage < 0.75
        ):
            if (
                anchor.visual_start > item.end + DEFAULT_OCR_SAMPLE_INTERVAL
                and anchor.target_fraction > 0.5
            ):
                partial_directions[index] = "tail"
                continue
            if (
                anchor.visual_end < item.start - DEFAULT_OCR_SAMPLE_INTERVAL
                and anchor.target_fraction < 0.5
            ):
                partial_directions[index] = "head"
                continue
        anchor_shifts[index] = anchor.delta

    anchor_indices = sorted(anchor_shifts)
    shifts: list[float] = []
    for index in range(line_count):
        if index in anchor_shifts:
            shifts.append(anchor_shifts[index])
            continue
        left = next((value for value in reversed(anchor_indices) if value < index), None)
        right = next((value for value in anchor_indices if value > index), None)
        if left is not None and right is not None:
            left_time = (baseline[left].start + baseline[left].end) / 2.0
            right_time = (baseline[right].start + baseline[right].end) / 2.0
            current_time = (baseline[index].start + baseline[index].end) / 2.0
            span = max(0.001, right_time - left_time)
            ratio = max(0.0, min(1.0, (current_time - left_time) / span))
            shifts.append(
                anchor_shifts[left] * (1.0 - ratio)
                + anchor_shifts[right] * ratio
            )
        elif left is not None and index - left <= 2:
            shifts.append(anchor_shifts[left])
        elif right is not None and right - index <= 2:
            shifts.append(anchor_shifts[right])
        else:
            shifts.append(0.0)

    # 相邻歌词不允许出现突兀的大幅反向跳动；这类情况通常是重复副歌误匹配。
    for index in range(1, line_count):
        shifts[index] = max(
            shifts[index - 1] - 2.0,
            min(shifts[index - 1] + 2.0, shifts[index]),
        )
    for index in range(line_count - 2, -1, -1):
        shifts[index] = max(
            shifts[index + 1] - 2.0,
            min(shifts[index + 1] + 2.0, shifts[index]),
        )

    starts: list[float] = [0.0] * line_count
    ends: list[float] = [0.0] * line_count
    starts[0] = max(0.0, baseline[0].start + shifts[0])
    ends[-1] = baseline[-1].end + shifts[-1]

    for index in range(line_count - 1):
        original_gap = max(
            0.0,
            baseline[index + 1].start - baseline[index].end,
        )
        original_center = (
            baseline[index].end + baseline[index + 1].start
        ) / 2.0
        boundary_shift = (shifts[index] + shifts[index + 1]) / 2.0
        corrected_center = original_center + boundary_shift
        ends[index] = corrected_center - original_gap / 2.0
        starts[index + 1] = corrected_center + original_gap / 2.0

    visual_boundary_corrections = 0
    for index in range(line_count - 1):
        left_anchor = anchor_map.get(index)
        right_anchor = anchor_map.get(index + 1)
        if not left_anchor or not right_anchor:
            continue
        if index in partial_directions or index + 1 in partial_directions:
            continue
        if (
            left_anchor.visual_end is None
            or right_anchor.visual_start is None
            or right_anchor.visual_start < left_anchor.visual_end
            or right_anchor.visual_start - left_anchor.visual_end > 1.5
        ):
            continue
        current_center = (ends[index] + starts[index + 1]) / 2.0
        visual_center = (
            left_anchor.visual_end + right_anchor.visual_start
        ) / 2.0
        if abs(current_center - visual_center) <= 1.5:
            continue
        ends[index] = left_anchor.visual_end
        starts[index + 1] = right_anchor.visual_start
        visual_boundary_corrections += 1

    duration_corrections = 0
    for index, item in enumerate(baseline):
        anchor = anchor_map.get(index)
        if (
            not anchor
            or anchor.visual_start is None
            or anchor.visual_end is None
            or anchor.sample_count < 2
        ):
            continue
        typical_duration, _lower_duration, _upper_duration = duration_profiles[index]
        baseline_duration = item.end - item.start
        if not _is_wide_duration_outlier(
            item,
            anchor,
            duration_profiles[index],
        ):
            continue

        visual_span = anchor.visual_end - anchor.visual_start
        estimated_duration = max(typical_duration, visual_span)
        if baseline_duration <= estimated_duration + DEFAULT_OCR_SAMPLE_INTERVAL:
            continue

        start_distance = abs(anchor.visual_start - item.start)
        end_distance = abs(anchor.visual_end - item.end)
        if start_distance <= end_distance:
            estimated_start = anchor.visual_start
            estimated_end = max(
                anchor.visual_end,
                estimated_start + estimated_duration,
            )
        else:
            estimated_end = anchor.visual_end
            estimated_start = min(
                anchor.visual_start,
                estimated_end - estimated_duration,
            )
        minimum_duration = min(0.5, max(0.08, baseline_duration))
        if index > 0:
            estimated_start = max(estimated_start, ends[index - 1])
        if index + 1 < line_count:
            estimated_end = min(estimated_end, starts[index + 1])
        if estimated_end - estimated_start < minimum_duration:
            continue

        starts[index] = estimated_start
        ends[index] = estimated_end
        review_reasons[index] = "ocr_duration_outlier"
        duration_corrections += 1

    partial_span_corrections = 0
    for index, direction in partial_directions.items():
        anchor = anchor_map[index]
        typical_duration = duration_profiles[index][0]
        if direction == "tail":
            estimated_end = max(
                anchor.visual_end,
                anchor.time + typical_duration * (1.0 - anchor.target_fraction),
            )
            if index + 1 < line_count:
                estimated_end = min(estimated_end, starts[index + 1])
            if estimated_end > ends[index]:
                ends[index] = estimated_end
            else:
                continue
        else:
            estimated_start = min(
                anchor.visual_start,
                anchor.time - typical_duration * anchor.target_fraction,
            )
            if index > 0:
                estimated_start = max(estimated_start, ends[index - 1])
            if estimated_start < starts[index]:
                starts[index] = estimated_start
            else:
                continue
        review_reasons[index] = "ocr_partial_span"
        partial_span_corrections += 1

    # 极短的 Okay / C'mon 等行容易被两侧不同的修正量挤扁。
    # 局部移动两侧边界，至少保留原长度或 0.5 秒中的较小值。
    for _ in range(line_count * 2):
        changed = False
        for index in range(line_count):
            minimum_duration = min(
                0.5,
                max(0.08, baseline[index].end - baseline[index].start),
            )
            deficit = minimum_duration - (ends[index] - starts[index])
            if deficit <= 1e-6:
                continue
            changed = True
            left_adjustment = deficit / 2.0 if index > 0 else 0.0
            right_adjustment = deficit - left_adjustment if index < line_count - 1 else 0.0
            if index > 0:
                starts[index] -= left_adjustment
                ends[index - 1] -= left_adjustment
            if index < line_count - 1:
                ends[index] += right_adjustment
                starts[index + 1] += right_adjustment
        if not changed:
            break

    if any(end - start < 0.079 for start, end in zip(starts, ends)):
        print("    OCR 修正后出现无效字幕区间。")
        return baseline

    corrected = []
    for index, item in enumerate(baseline):
        anchor = anchor_map.get(index)
        applied_shift = (
            starts[index] + ends[index] - item.start - item.end
        ) / 2.0
        corrected.append(replace(
            item,
            start=round(starts[index], 3),
            end=round(ends[index], 3),
            ocr_text=anchor.recognized_text if anchor else "",
            ocr_score=anchor.score if anchor else None,
            ocr_shift=round(applied_shift, 3),
            ocr_target_coverage=anchor.target_coverage if anchor else None,
            review_reason=review_reasons.get(index, ""),
        ))
    print(
        f"    OCR 画面锚点: {len(anchors)}/{line_count} 行，"
        "已校正局部漂移。"
    )
    if visual_boundary_corrections:
        print(f"    OCR 修正明显错位边界: {visual_boundary_corrections} 处。")
    if duration_corrections:
        print(f"    OCR 修正异常时长: {duration_corrections} 行。")
    if partial_span_corrections:
        print(f"    OCR 合并单侧片段: {partial_span_corrections} 行。")
    return corrected


def scan_video_ocr(
    video_path: Path,
    work_dir: Path,
    sample_interval: float = DEFAULT_OCR_SAMPLE_INTERVAL,
) -> list[OcrFrame]:
    """按固定间隔扫描全画面文字，返回可缓存的 OCR 帧。"""
    if sample_interval <= 0:
        raise ValueError("OCR 采样间隔必须大于 0")
    try:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    except ImportError:
        _die("未安装 OCR 组件，请运行: pip install rapidocr onnxruntime")

    frames_dir = work_dir / "ocr_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / sample_interval
    frame_pattern = frames_dir / "frame_%06d.jpg"
    _run(
        [
            _get_ffmpeg_path(), "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps:.6f},scale='min(1920,iw)':-2",
            "-q:v", "3",
            str(frame_pattern),
        ],
        "提取 OCR 采样画面",
    )

    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        return []
    print("    正在加载日文 OCR 模型...")
    engine = RapidOCR(params={
        "Det.model_type": ModelType.SMALL,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.lang_type": LangRec.JAPAN,
        "Rec.model_type": ModelType.SMALL,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
    })

    frames: list[OcrFrame] = []
    total = len(paths)
    for index, path in enumerate(paths):
        result = engine(str(path))
        texts = []
        confidences = []
        for text, confidence in zip(result.txts or (), result.scores or ()):
            cleaned = str(text).strip()
            if not cleaned or float(confidence) < 0.35:
                continue
            texts.append(cleaned)
            confidences.append(round(float(confidence), 4))
        frames.append(OcrFrame(
            time=round(index * sample_interval, 3),
            texts=tuple(texts),
            confidences=tuple(confidences),
        ))
        if (index + 1) % 10 == 0 or index + 1 == total:
            print(f"    OCR 扫描进度: {index + 1}/{total}", flush=True)
    return frames


def _normalize_latin_anchor(text: str) -> str:
    """提取中日混合歌词中可与 Whisper 直接比较的英文和数字。"""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(re.findall(r"[a-z0-9]+", normalized))


def _align_with_latin_anchors(
    words: list[WordTiming],
    lyric_lines: list[LyricLine],
    min_match_score: float,
    strict_anchors: bool = False,
) -> Optional[list[AlignmentItem]]:
    """纯中文歌词使用其中的英文/数字短语作为低置信度顺序锚点。"""
    anchor_lines = [
        LyricLine(
            anchor if len(anchor) >= 4 else "",
            line.zh,
        )
        for line in lyric_lines
        for anchor in [_normalize_latin_anchor(line.zh)]
    ]
    anchorable_count = sum(bool(line.ja) for line in anchor_lines)
    latin_words = [
        WordTiming(word.start, word.end, anchor, word.probability)
        for word in words
        for anchor in [_normalize_latin_anchor(word.text)]
        if anchor
    ]
    if anchorable_count < 2 or len(latin_words) < 2:
        return None

    anchor_threshold = max(45.0, min_match_score - 10.0)
    print(
        f"    对轴模式: 纯中文英文锚点匹配"
        f"（候选 {anchorable_count}/{len(lyric_lines)} 行）"
    )
    aligned = _align_with_text(
        latin_words,
        anchor_lines,
        anchor_threshold,
        max_span_seconds=5.0,
        strict_anchors=strict_anchors,
    )
    matched_count = 0
    for item in aligned:
        if item.status in {"matched", "low_confidence"}:
            item.status = "latin_anchor"
            matched_count += 1

    if matched_count < 2:
        print("    英文文本锚点不足 2 行，改用时间比例回退。")
        return None
    return aligned


def _align_with_text(
    words: list[WordTiming],
    lyric_lines: list[LyricLine],
    min_match_score: float,
    max_span_seconds: Optional[float] = None,
    strict_anchors: bool = False,
    fill_missing: bool = True,
) -> list[AlignmentItem]:
    """按歌词顺序将每行日文匹配到连续的 Whisper word span。"""
    line_count = len(lyric_lines)
    word_count = len(words)
    infinity = float("inf")
    skip_word_cost = 0.16
    missing_line_cost = 1.05
    anchor_threshold = max(75.0, min_match_score) if strict_anchors else min_match_score
    candidate_floor = anchor_threshold if strict_anchors else max(30.0, min_match_score - 20.0)

    costs = [[infinity] * (word_count + 1) for _ in range(line_count + 1)]
    previous: list[list[Optional[tuple]]] = [
        [None] * (word_count + 1) for _ in range(line_count + 1)
    ]
    costs[0][0] = 0.0

    def update(next_i: int, next_j: int, value: float, back: tuple) -> None:
        if value + 1e-9 < costs[next_i][next_j]:
            costs[next_i][next_j] = value
            previous[next_i][next_j] = back

    for line_index in range(line_count + 1):
        for word_index in range(word_count + 1):
            current = costs[line_index][word_index]
            if current == infinity:
                continue

            if word_index < word_count:
                update(
                    line_index,
                    word_index + 1,
                    current + skip_word_cost,
                    (line_index, word_index, "skip_word", None),
                )

            if line_index >= line_count:
                continue

            line = lyric_lines[line_index]
            target = _normalize_japanese(line.ja)
            update(
                line_index + 1,
                word_index,
                current + missing_line_cost,
                (line_index, word_index, "missing_line", None),
            )
            if not target or word_index >= word_count:
                continue

            max_words = min(word_count - word_index, max(4, min(30, len(target) * 2 + 4)))
            raw_parts: list[str] = []
            for span_size in range(1, max_words + 1):
                end_index = word_index + span_size
                if (
                    max_span_seconds is not None
                    and words[end_index - 1].end - words[word_index].start
                    > max_span_seconds
                ):
                    break
                raw_parts.append(words[end_index - 1].text)
                recognized = "".join(raw_parts)
                normalized_recognized = _normalize_japanese(recognized)
                if not normalized_recognized:
                    continue
                if (
                    strict_anchors and len(target) <= 16
                    and re.fullmatch(r"[a-z0-9]+", target)
                    and target != normalized_recognized
                ):
                    target_tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", line.ja).casefold())
                    observed_tokens = re.findall(r"[a-z0-9]+", " ".join(raw_parts).casefold())
                    exact_words = sum(block.size for block in SequenceMatcher(
                        None, target_tokens, observed_tokens, autojunk=False,
                    ).get_matching_blocks())
                    if exact_words / max(len(target_tokens), 1) < 0.75:
                        continue
                if strict_anchors:
                    timed_words = [word for word in words[word_index:end_index] if word.end > word.start]
                    if not timed_words or len(timed_words) * 2 < span_size:
                        continue
                if len(normalized_recognized) > len(target) * 3 + 10:
                    break

                text_score = _similarity(target, normalized_recognized)
                length_score = (
                    min(len(target), len(normalized_recognized))
                    / max(len(target), len(normalized_recognized))
                    * 100.0
                )
                score = text_score * 0.88 + length_score * 0.12
                if score < candidate_floor:
                    continue

                match_cost = (100.0 - score) / 50.0
                update(
                    line_index + 1,
                    end_index,
                    current + match_cost,
                    (line_index, word_index, "match", (end_index, score, recognized)),
                )

    line_index, word_index = line_count, word_count
    matches: dict[int, tuple[int, int, float, str]] = {}
    while line_index or word_index:
        back = previous[line_index][word_index]
        if back is None:
            break
        prev_line, prev_word, action, payload = back
        if action == "match":
            end_index, score, recognized = payload
            matches[prev_line] = (prev_word, end_index, score, recognized)
        line_index, word_index = prev_line, prev_word

    anchored: dict[int, AlignmentItem] = {}
    for index, (start_word, end_word, score, recognized) in matches.items():
        if score < anchor_threshold:
            continue
        span_words = words[start_word:end_word]
        if strict_anchors:
            span_words = [word for word in span_words if word.end > word.start]
            if not span_words:
                continue
        probabilities = [
            word.probability
            for word in words[start_word:end_word]
            if word.probability is not None
        ]
        word_probability = (
            sum(probabilities) / len(probabilities) if probabilities else None
        )
        status = (
            "matched"
            if score >= 75.0 and (word_probability is None or word_probability >= 0.50)
            else "low_confidence"
        )
        anchored[index] = AlignmentItem(
            start=span_words[0].start,
            end=span_words[-1].end,
            text=lyric_lines[index].zh,
            source_text=lyric_lines[index].ja,
            recognized_text=recognized,
            score=round(score, 2),
            status=status,
            word_probability=(
                round(word_probability, 4) if word_probability is not None else None
            ),
        )

    print(f"    文本锚点: {len(anchored)}/{line_count} 行（阈值 {anchor_threshold:.0f}）")
    if not fill_missing:
        return [anchored.get(index) or AlignmentItem(None, None, line.zh, source_text=line.ja, status="unresolved")
                for index, line in enumerate(lyric_lines)]
    return _fill_unmatched_lines(lyric_lines, anchored, words[0].start, words[-1].end)


def _fill_unmatched_lines(
    lyric_lines: list[LyricLine],
    anchored: dict[int, AlignmentItem],
    speech_start: float,
    speech_end: float,
) -> list[AlignmentItem]:
    """只在相邻文本锚点之间补齐未识别歌词，避免误差全局扩散。"""
    results: list[Optional[AlignmentItem]] = [None] * len(lyric_lines)
    for index, item in anchored.items():
        results[index] = item

    cursor = 0
    while cursor < len(results):
        if results[cursor] is not None:
            cursor += 1
            continue
        run_start = cursor
        while cursor < len(results) and results[cursor] is None:
            cursor += 1
        run_end = cursor

        has_left_anchor = run_start > 0 and results[run_start - 1] is not None
        has_right_anchor = run_end < len(results) and results[run_end] is not None
        left = speech_start
        if has_left_anchor:
            left = results[run_start - 1].end
        right = speech_end
        if has_right_anchor:
            right = results[run_end].start

        count = run_end - run_start
        if right <= left:
            if has_right_anchor and not has_left_anchor:
                left = max(0.0, right - 2.0 * count)
            elif has_left_anchor and not has_right_anchor:
                right = left + 2.0 * count
            else:
                right = left + max(0.10 * count, 0.10)

        weights = []
        for index in range(run_start, run_end):
            line = lyric_lines[index]
            basis = _normalize_japanese(line.ja) or re.sub(r"\s+", "", line.zh)
            weights.append(max(len(basis), 1))
        weight_sum = sum(weights)
        position = left
        for offset, weight in enumerate(weights):
            index = run_start + offset
            end = right if offset == count - 1 else position + (right - left) * weight / weight_sum
            results[index] = AlignmentItem(
                start=position,
                end=max(end, position + 0.05),
                text=lyric_lines[index].zh,
                source_text=lyric_lines[index].ja,
                status="interpolated",
                review_reason=("连续多行未可靠定位，时间为估计值" if count >= 3 else ""),
            )
            position = end
            weight_sum -= weight
            left = position

    return [item for item in results if item is not None]


def _align_by_timing(
    segments: list,
    lyric_lines: list[LyricLine],
) -> list[AlignmentItem]:
    """无日文原文时的兼容回退；结果统一标记为 timing_fallback。"""
    sub_segments = _split_segments_at_gaps(segments, gap_threshold=0.3)
    n_subs = len(sub_segments)
    n_lyrs = len(lyric_lines)

    print(f"    Whisper 片段: {len(segments)} → word-gap 拆分后: {n_subs} 子段")
    print(f"    歌词行数: {n_lyrs}")

    if n_subs == 0:
        _die("未找到有效音频片段。")

    if n_subs == n_lyrs:
        print(f"    子模式: 1:1 直接映射")
        return [
            AlignmentItem(_get_word_start(s), _get_word_end(s), line.zh)
            for s, line in zip(sub_segments, lyric_lines)
        ]

    elif n_subs > n_lyrs:
        print(f"    子模式: 合并子段（{n_subs}→{n_lyrs}）")
        result = []
        for i, line in enumerate(lyric_lines):
            start_idx = round(i * n_subs / n_lyrs)
            end_idx = round((i + 1) * n_subs / n_lyrs)
            end_idx = min(end_idx, n_subs)
            start_idx = min(start_idx, end_idx - 1)

            start_t = _get_word_start(sub_segments[start_idx])
            end_t = _get_word_end(sub_segments[end_idx - 1])
            # 单条字幕不超过 15 秒
            if end_t - start_t > 15:
                end_t = start_t + 15
            result.append(AlignmentItem(start_t, end_t, line.zh))
        return result

    else:
        print(f"    子模式: 拆分子段（{n_subs}→{n_lyrs}）")
        result = []
        for i, seg in enumerate(sub_segments):
            start_lyr = round(i * n_lyrs / n_subs)
            end_lyr = round((i + 1) * n_lyrs / n_subs)
            end_lyr = min(end_lyr, n_lyrs)
            if start_lyr >= end_lyr:
                continue

            seg_dur = seg.end - seg.start
            chunk_dur = seg_dur / (end_lyr - start_lyr)

            for j, lyr_idx in enumerate(range(start_lyr, end_lyr)):
                t_start = seg.start + j * chunk_dur
                t_end = t_start + chunk_dur
                result.append(AlignmentItem(t_start, t_end, lyric_lines[lyr_idx].zh))
        return result


# ── Word-gap 拆分 ─────────────────────────────────────────────────────────────

class _SubSegment:
    """轻量子段对象，有 start/end/words 属性。"""
    __slots__ = ("start", "end", "words")
    def __init__(self, start, end, words=None):
        self.start = start
        self.end = end
        self.words = words


def _split_segments_at_gaps(segments: list, gap_threshold: float = 0.3) -> list:
    """
    将 Whisper 的 segment 按 word 间的时间间隙拆分为子段。

    歌曲连续唱时，Whisper VAD 检测不到换气停顿，会把多句歌词
    合并成一个长 segment（如 11 秒、18 秒）。但 segment 内部的
    word 级时间戳之间存在微小间隙，这些间隙往往对应歌词边界。

    策略：找出相邻 word 之间 > gap_threshold 秒的间隙，在此处拆分。
    """
    result = []
    for seg in segments:
        words = getattr(seg, "words", None)
        if not words or len(words) <= 1:
            result.append(_SubSegment(seg.start, seg.end, words))
            continue

        # 扫描相邻 word 间隙
        group_start = 0
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap > gap_threshold:
                # 在 i 和 i+1 之间拆分
                sub = _SubSegment(
                    words[group_start].start,
                    words[i].end,
                    words[group_start:i + 1],
                )
                result.append(sub)
                group_start = i + 1

        # 最后一组
        sub = _SubSegment(
            words[group_start].start,
            words[-1].end,
            words[group_start:],
        )
        result.append(sub)

    return result


def _get_word_start(segment) -> float:
    """获取 segment 的首个词起始时间（词级精度 ±0.05s）。"""
    if hasattr(segment, "words") and segment.words:
        return segment.words[0].start
    return segment.start


def _get_word_end(segment) -> float:
    """获取 segment 的末词结束时间（词级精度 ±0.05s）。"""
    if hasattr(segment, "words") and segment.words:
        return segment.words[-1].end
    return segment.end


# ──────────────────────────────────────────────────────────────────────────────
# 步骤 5: 生成 SRT & 烧入字幕
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_srt_time(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳 HH:MM:SS,mmm。"""
    total_ms = max(0, round(seconds * 1000))
    total_seconds, ms = divmod(total_ms, 1000)
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(aligned: list[AlignmentItem], output_path: Path) -> None:
    """生成标准 SRT 字幕文件。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(aligned, 1):
            f.write(f"{idx}\n")
            f.write(f"{_fmt_srt_time(item.start)} --> {_fmt_srt_time(item.end)}\n")
            f.write(f"{item.text}\n\n")
    print(f"    SRT 文件已生成: {output_path}")


def save_alignment_report(
    aligned: list[AlignmentItem],
    output_path: Path,
    min_match_score: float,
    strict_anchors: bool = False,
    whisperx_backend: dict | None = None,
) -> dict:
    """保存逐行匹配依据与置信度，返回摘要。"""
    matched = [
        item
        for item in aligned
        if item.status in {"matched", "low_confidence", "latin_anchor", "whisperx", "asr_anchor"}
    ]
    scores = [item.score for item in matched if item.score is not None]
    word_probabilities = [
        item.word_probability
        for item in matched
        if item.word_probability is not None
    ]
    ocr_anchors = [item for item in aligned if item.ocr_score is not None]
    ocr_scores = [item.ocr_score for item in ocr_anchors if item.ocr_score is not None]
    if whisperx_backend is not None:
        mode = "whisperx"
    elif any(item.status == "latin_anchor" for item in aligned):
        mode = "latin_anchor"
    elif any(item.source_text for item in aligned):
        mode = "text_dp"
    else:
        mode = "timing_fallback"
    effective_minimum = (
        max(45.0, min_match_score - 10.0)
        if mode == "latin_anchor"
        else min_match_score
    )
    if strict_anchors:
        effective_minimum = max(75.0, effective_minimum)
    summary = {
        "mode": mode,
        "line_count": len(aligned),
        "matched_lines": len(matched),
        "matched_coverage": round(len(matched) / len(aligned), 4) if aligned else 0.0,
        "estimated_lines": sum(item.status == "interpolated" for item in aligned),
        "average_match_score": round(sum(scores) / len(scores), 2) if scores else None,
        "average_word_probability": (
            round(sum(word_probabilities) / len(word_probabilities), 4)
            if word_probabilities else None
        ),
        "minimum_match_score": effective_minimum,
        "ocr_assisted": bool(ocr_anchors),
        "ocr_anchor_lines": len(ocr_anchors),
        "ocr_anchor_coverage": (
            round(len(ocr_anchors) / len(aligned), 4) if aligned else 0.0
        ),
        "average_ocr_score": (
            round(sum(ocr_scores) / len(ocr_scores), 2) if ocr_scores else None
        ),
        "needs_review": [
            index
            for index, item in enumerate(aligned, 1)
            if item.status not in {"matched", "whisperx"} or item.review_reason
        ],
        "review_reasons": {
            str(index): item.review_reason
            for index, item in enumerate(aligned, 1)
            if item.review_reason
        },
    }
    payload = {
        "schema_version": 2,
        "summary": summary,
        "items": [
            {"index": index, **asdict(item)}
            for index, item in enumerate(aligned, 1)
        ],
    }
    if whisperx_backend is not None:
        payload["whisperx"] = whisperx_backend
        summary["whisperx_lines"] = whisperx_backend["aligned_lines"]
        summary["asr_anchor_lines"] = whisperx_backend["asr_anchor_lines"]
        summary["unresolved_lines"] = whisperx_backend["unresolved_lines"]
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"    对轴报告已生成: {output_path}")
    if mode == "whisperx":
        print(f"    WhisperX 歌词对齐: {summary['whisperx_lines']}/{len(aligned)} 行，"
              f"ASR 锚点 {summary['asr_anchor_lines']} 行，未定位 {len(summary['unresolved_lines'])} 行")
    elif mode in {"text_dp", "latin_anchor"}:
        label = "文本匹配" if mode == "text_dp" else "英文锚点"
        print(
            f"    {label}覆盖率: {summary['matched_coverage'] * 100:.1f}%"
            f"，平均分: {summary['average_match_score'] if scores else 'N/A'}"
        )
    else:
        print("    ⚠️  未提供可匹配的日文歌词，全部时间轴均为低置信度回退。")
    if ocr_anchors:
        print(
            f"    OCR 辅助覆盖率: {summary['ocr_anchor_coverage'] * 100:.1f}%"
            f"，平均分: {summary['average_ocr_score']}"
        )
    return summary


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path, *,
                   subtitle_font: str = DEFAULT_FONT, subtitle_size: float | None = None) -> None:
    """用所选字体和字号压制硬字幕，版式沿用 FFmpeg 工具箱的 SRT 默认样式。"""
    ffmpeg_exe = _get_ffmpeg_path()
    family, size = font_settings(subtitle_font, subtitle_size)
    with tempfile.TemporaryDirectory(prefix="subtitle_style_") as temporary:
        ass = Path(temporary) / "subtitle.ass"
        _run([ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
              "-sub_charenc", "UTF-8", "-i", str(srt_path), str(ass)], "转换字幕样式")
        content = ass.read_text(encoding="utf-8-sig")
        ass.write_text(apply_ass_style(content, subtitle_font, size), encoding="utf-8")
        print(f"    字幕字体: {family}，字号: {size:g}", flush=True)
        cmd = [
            ffmpeg_exe, "-y", "-i", str(video_path),
            "-vf", subtitle_filter(ass, subtitle_font),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy", str(output_path),
        ]
        _run(cmd, "FFmpeg 烧入字幕")


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return result


def _ocr_metadata(video_path: Path, sample_interval: float) -> dict:
    return {
        "schema_version": OCR_CACHE_SCHEMA,
        "profile": "rapidocr-ppocrv6-japan-full-frame-v1",
        "sample_interval": sample_interval,
        "video": _file_fingerprint(video_path),
    }


def save_ocr_cache(path: Path, frames: list[OcrFrame], metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "frames": [
            {
                "time": frame.time,
                "texts": list(frame.texts),
                "confidences": list(frame.confidences),
            }
            for frame in frames
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"    OCR 扫描缓存已保存: {path}")


def load_ocr_cache(path: Path, expected_metadata: dict) -> Optional[list[OcrFrame]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("metadata") != expected_metadata:
            print("    已有 OCR 缓存与当前视频或设置不一致，将重新扫描。")
            return None
        frames = [
            OcrFrame(
                time=float(frame["time"]),
                texts=tuple(str(text) for text in frame.get("texts", [])),
                confidences=tuple(
                    float(value) for value in frame.get("confidences", [])
                ),
            )
            for frame in payload.get("frames", [])
        ]
        if frames:
            print(f"    复用 OCR 扫描缓存，共 {len(frames)} 帧")
            return frames
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"    OCR 缓存读取失败，将重新扫描: {error}")
    return None


def _transcript_metadata(
    video_path: Path,
    use_vad: bool,
    audio_source: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
) -> dict:
    # Whisper 原始转录不依赖歌词内容。
    metadata = {
        "schema_version": TRANSCRIPT_CACHE_SCHEMA,
        "model": model_name,
        "language": "ja",
        "transcription_profile": "audio-ja-v2-no-lyric-prompt",
        "use_vad": use_vad,
        "audio_source": audio_source,
        "video": _file_fingerprint(video_path),
    }
    return metadata


def _transcript_records(segments: list) -> list[dict]:
    records = []
    for segment in segments:
        words = []
        for word in getattr(segment, "words", None) or []:
            probability = getattr(word, "probability", None)
            words.append({
                "start": float(word.start),
                "end": float(word.end),
                "text": getattr(word, "word", getattr(word, "text", "")),
                "probability": float(probability) if probability is not None else None,
            })
        record = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": getattr(segment, "text", ""),
            "words": words,
        }
        records.append(record)
    return records


def _transcript_segments(records: list[dict]) -> list[SimpleSegment]:
    return [SimpleSegment(
        start=float(raw["start"]), end=float(raw["end"]), text=raw.get("text", ""),
        words=[WordTiming(**word) for word in raw.get("words", [])],
    ) for raw in records]


def save_transcript_cache(path: Path, segments: list, metadata: dict) -> None:
    payload = {"metadata": metadata, "segments": _transcript_records(segments)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"    转录缓存已保存: {path}")


def load_transcript_cache(path: Path, expected_metadata: dict) -> Optional[list[SimpleSegment]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("metadata") != expected_metadata:
            print("    已有转录缓存与当前输入或设置不一致，将重新识别。")
            return None
        segments = _transcript_segments(payload.get("segments", []))
        if segments:
            print(f"    ♻️  复用转录缓存，共 {len(segments)} 个片段")
            return segments
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"    ⚠️  转录缓存读取失败，将重新识别: {error}")
    return None


def process_single(
    video_path: Path,
    lyrics_path: Path,
    output_path: Path | None = None,
    no_vad: bool = False,
    no_burn: bool = False,
    refresh_transcript: bool = False,
    audio_source: str = "vocals",
    model_name: str = DEFAULT_WHISPER_MODEL,
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    ocr_assist: bool = False,
    refresh_ocr: bool = False,
    ocr_sample_interval: float = DEFAULT_OCR_SAMPLE_INTERVAL,
    subtitle_font: str = DEFAULT_FONT,
    subtitle_size: float | None = None,
) -> None:
    """处理单个视频文件。"""
    video_path  = Path(video_path)
    lyrics_path = Path(lyrics_path)

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_subtitled{video_path.suffix}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_source not in {"vocals", "original"}:
        raise ValueError(f"不支持的音源模式: {audio_source}")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的语音模型: {model_name}")

    if model_name == LYRIC_MODEL:
        if ocr_assist:
            raise ValueError("歌词声学对齐模式不使用画面歌词辅助，请关闭该选项")
        from lyric_alignment.workflow import process_known_lyrics
        return process_known_lyrics(
            video_path, lyrics_path, output_path, audio_source=audio_source,
            no_burn=no_burn, refresh=refresh_transcript, helpers=sys.modules[__name__],
            subtitle_font=subtitle_font, subtitle_size=subtitle_size,
        )

    _banner(f"处理: {video_path.name}")
    total_steps = 8 if ocr_assist else 7

    with tempfile.TemporaryDirectory(prefix="subtitle_sync_") as _tmpdir:
        tmpdir = Path(_tmpdir)

        _step(1, total_steps, "加载歌词")
        _ja_prompt, lyric_lines = load_lyrics(lyrics_path, require_pairs=True)
        from mv_whisperx import align_lyrics as align_mv, metadata as mv_metadata, load_cache as load_mv_cache
        cache_path = output_path.with_suffix(".whisper.json")
        metadata = _transcript_metadata(
            video_path,
            use_vad=not no_vad,
            audio_source=audio_source,
            model_name=model_name,
        )
        segments = None if refresh_transcript else load_transcript_cache(cache_path, metadata)
        report_path = output_path.with_suffix(".alignment.json")
        alignment_metadata = mv_metadata(metadata, lyric_lines)
        cached_alignment = None if refresh_transcript else load_mv_cache(report_path, alignment_metadata, lyric_lines,
                                                                        sys.modules[__name__])
        if segments is None or cached_alignment is None:
            _step(2, total_steps, "提取音频")
            audio_wav = tmpdir / "audio.wav"
            extract_audio(video_path, audio_wav)

            if audio_source == "vocals":
                _step(3, total_steps, "Demucs 人声分离")
                transcription_audio = separate_vocals(audio_wav, tmpdir)
            else:
                _step(3, total_steps, "使用原始混合音频（跳过 Demucs）")
                transcription_audio = audio_wav

        else:
            _step(2, total_steps, "复用 WhisperX 歌词对齐缓存，跳过音频提取")
            _step(3, total_steps, "复用 WhisperX 歌词对齐缓存，跳过人声分离")

        if segments is None:
            _step(4, total_steps, f"{model_label(model_name)} 日语识别")
            segments = transcribe_audio(
                transcription_audio,
                model_name=model_name,
                use_vad=not no_vad,
            )
            save_transcript_cache(cache_path, segments, metadata)
        else:
            _step(4, total_steps, "复用转录缓存，跳过识别")

        _step(5, total_steps, "WhisperX 日文歌词声学对齐")
        if cached_alignment is not None:
            aligned, whisperx_backend = cached_alignment
            print("    复用 WhisperX 歌词对齐缓存")
        else:
            alignment_audio = tmpdir / "alignment.wav"
            _run([_get_ffmpeg_path(), "-y", "-i", str(transcription_audio), "-ar", "16000", "-ac", "1",
                  str(alignment_audio)], "准备 WhisperX 对齐音频")
            aligned, whisperx_backend = align_mv(alignment_audio, segments, lyric_lines, min_match_score,
                                                sys.modules[__name__])
            whisperx_backend["metadata"] = alignment_metadata
            whisperx_backend["items"] = [asdict(item) for item in aligned]
        if whisperx_backend["unresolved_lines"]:
            save_alignment_report(aligned, report_path, min_match_score, whisperx_backend=whisperx_backend)
            missing = ", ".join(map(str, whisperx_backend["unresolved_lines"]))
            raise RuntimeError(f"WhisperX 未定位到歌词第 {missing} 行，详情见 {report_path}")

        ocr_cache_path = None
        if ocr_assist:
            ocr_cache_path = output_path.with_suffix(".ocr.json")
            ocr_metadata = _ocr_metadata(video_path, ocr_sample_interval)
            ocr_frames = None if refresh_ocr else load_ocr_cache(
                ocr_cache_path,
                ocr_metadata,
            )
            if ocr_frames is None:
                _step(6, total_steps, "扫描画面歌词")
                ocr_frames = scan_video_ocr(
                    video_path,
                    tmpdir,
                    sample_interval=ocr_sample_interval,
                )
                save_ocr_cache(ocr_cache_path, ocr_frames, ocr_metadata)
            else:
                _step(6, total_steps, "复用 OCR 缓存，跳过画面扫描")
            ocr_anchors = build_ocr_line_anchors(
                ocr_frames,
                lyric_lines,
                aligned,
            )
            aligned = apply_ocr_timing_assist(aligned, ocr_anchors)

        # 生成 SRT（始终保存到输出视频同目录）
        srt_path = output_path.with_suffix(".srt")
        generate_srt(aligned, srt_path)

        report_step = 7 if ocr_assist else 6
        _step(report_step, total_steps, "生成对轴置信度报告")
        summary = save_alignment_report(aligned, report_path, min_match_score, whisperx_backend=whisperx_backend)

        if no_burn:
            _step(total_steps, total_steps, "仅生成 SRT，跳过视频烧录")
        else:
            _step(total_steps, total_steps, "FFmpeg 烧入硬字幕")
            burn_subtitles(video_path, srt_path, output_path,
                           subtitle_font=subtitle_font, subtitle_size=subtitle_size)

    print()
    if not no_burn:
        print(f"  ✅ 完成！输出文件: {output_path}")
    else:
        print("  ✅ 对轴完成（未烧录视频）")
    print(f"  📄 SRT 文件: {srt_path}")
    print(f"  📊 对轴报告: {report_path}")
    print(f"  ♻️  转录缓存: {cache_path}")
    if ocr_cache_path is not None:
        print(f"  👁️  画面文字缓存: {ocr_cache_path}")
    print()


def process_batch(
    video_dir: Path,
    lyrics_dir: Path,
    output_dir: Path | None = None,
    no_vad: bool = False,
    no_burn: bool = False,
    refresh_transcript: bool = False,
    audio_source: str = "vocals",
    model_name: str = DEFAULT_WHISPER_MODEL,
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
    ocr_assist: bool = False,
    refresh_ocr: bool = False,
    ocr_sample_interval: float = DEFAULT_OCR_SAMPLE_INTERVAL,
    subtitle_font: str = DEFAULT_FONT,
    subtitle_size: float | None = None,
) -> None:
    """批量处理目录下的所有视频文件。"""
    video_dir  = Path(video_dir)
    lyrics_dir = Path(lyrics_dir)

    if output_dir is None:
        output_dir = video_dir / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v"}
    videos = sorted(f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)

    if not videos:
        print(f"❌ 未在 {video_dir} 找到视频文件。")
        return

    print(f"📁 找到 {len(videos)} 个视频文件")
    skipped = 0

    for video_path in videos:
        lyrics_path = lyrics_dir / f"{video_path.stem}.txt"
        if not lyrics_path.exists():
            print(f"  ⚠️  跳过 {video_path.name}（未找到歌词文件 {lyrics_path.name}）")
            skipped += 1
            continue

        output_path = output_dir / f"{video_path.stem}_subtitled{video_path.suffix}"
        try:
            process_single(
                video_path,
                lyrics_path,
                output_path,
                no_vad=no_vad,
                no_burn=no_burn,
                refresh_transcript=refresh_transcript,
                audio_source=audio_source,
                model_name=model_name,
                min_match_score=min_match_score,
                ocr_assist=ocr_assist,
                refresh_ocr=refresh_ocr,
                ocr_sample_interval=ocr_sample_interval,
                subtitle_font=subtitle_font,
                subtitle_size=subtitle_size,
            )
        except Exception as e:
            print(f"  ❌ 处理 {video_path.name} 时出错: {e}")

    print(f"\n✅ 批量处理完成（跳过 {skipped} 个，输出目录: {output_dir}）")


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> None:
    """运行子进程，失败时打印详细错误信息。"""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [{label}] 命令失败:")
        print(f"   命令: {' '.join(cmd)}")
        if e.stderr:
            print(f"   错误信息 (最后2000字符):\n{e.stderr[-2000:]}")
        sys.exit(1)
    except FileNotFoundError:
        _die(f"命令未找到: {cmd[0]}（请确认已安装并在 PATH 中）")


def _get_ffmpeg_path() -> str:
    """获取 FFmpeg 可执行文件路径，支持回退到 imageio-ffmpeg 捆绑的版本。"""
    # 1. 尝试从系统 PATH 查找
    path = shutil.which("ffmpeg")
    if path:
        return path
    
    # 2. 尝试使用 imageio-ffmpeg 捆绑的 ffmpeg
    try:
        import imageio_ffmpeg
        base_path = imageio_ffmpeg.get_ffmpeg_exe()
        if base_path:
            # imageio-ffmpeg 的文件名可能叫 ffmpeg-win32-v...exe
            # Demucs 强依赖于名字叫做 ffmpeg.exe
            # 我们在相同目录下拷贝一份 ffmpeg.exe，或者在临时目录下创建一个
            ffmpeg_dir = os.path.dirname(base_path)
            target_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
            if not os.path.exists(target_exe):
                try:
                    shutil.copy2(base_path, target_exe)
                except PermissionError:
                    # 如果没有权限，就复制到临时目录
                    target_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_alias")
                    os.makedirs(target_dir, exist_ok=True)
                    target_exe = os.path.join(target_dir, "ffmpeg.exe")
                    if not os.path.exists(target_exe):
                        shutil.copy2(base_path, target_exe)
            return target_exe
    except ImportError:
        pass
        
    _die(
        "未找到 ffmpeg。\n"
        "  已尝试从系统 PATH 和 imageio-ffmpeg 查找均失败。\n"
        "  请运行: pip install imageio-ffmpeg\n"
        "  或手动下载 ffmpeg (https://ffmpeg.org/download.html) 并加入 PATH。"
    )
    return ""


def _die(msg: str) -> None:
    print(f"\n❌ 错误: {msg}")
    sys.exit(1)


def _banner(msg: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {msg}")
    print(sep)


def _step(current: int, total: int, label: str) -> None:
    print(f"\n  [{current}/{total}] {label}...")


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="subtitle_sync",
        description="MV 歌词打轴与视频自动字幕工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # MV 歌词打轴（中日双语歌词）
  py subtitle_sync.py --video song.mp4 --lyrics song.txt

  # 视频自动字幕（无需歌词）
  py subtitle_sync.py --task speech --video clip.mp4 --language auto

  # 指定输出
  py subtitle_sync.py --video song.mp4 --lyrics song.txt --output out.mp4

  # 批量处理
  py subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics
  py subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics --output-dir ./output

歌词文件格式:
  整块格式：中文歌词在前、日文原词在后，两块行数一致
  逐行格式：日文|中文  或  日文<Tab>中文
        """,
    )

    # ── 单文件模式 ────────────────────────────────────────────────
    parser.add_argument("--video",   help="输入视频文件路径")
    parser.add_argument("--lyrics",  help="中日双语歌词文本（UTF-8，支持整块或日文|中文逐行格式）")
    parser.add_argument("--output",  help="输出视频路径（默认: 原文件名_subtitled + 原扩展名）")
    parser.add_argument("--task", choices=("lyrics", "speech"), default="lyrics",
                        help="lyrics：歌词打轴；speech：视频自动字幕（WhisperX）")
    parser.add_argument("--language", default="auto", help="视频自动字幕识别语言：auto 或语言代码（如 ja/en/zh）")
    parser.add_argument("--no-vad",  action="store_true", help="禁用 VAD 静音过滤（不推荐）")
    parser.add_argument("--no-burn", action="store_true", help="只生成 SRT/报告，不烧录视频")
    parser.add_argument("--subtitle-font", choices=tuple(FONT_PRESETS), default=DEFAULT_FONT,
                        help="硬字幕字体：noto-sc（中文）、noto-jp（日文）、yahei（微软雅黑）")
    parser.add_argument("--subtitle-size", type=float,
                        help="硬字幕字号（6–96；默认 Noto 17.6、微软雅黑 16）")
    parser.add_argument(
        "--refresh-transcript",
        action="store_true",
        help="忽略已有转录缓存并重新识别",
    )
    parser.add_argument(
        "--audio-source",
        choices=["vocals", "original"],
        default="vocals",
        help="识别音源：Demucs 人声或原始混合音频（默认: vocals）",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=DEFAULT_WHISPER_MODEL,
        help=f"语音模型（默认: {DEFAULT_WHISPER_MODEL}）",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=DEFAULT_MIN_MATCH_SCORE,
        help=f"日文文本锚点最低相似度（默认: {DEFAULT_MIN_MATCH_SCORE:.0f}）",
    )
    parser.add_argument(
        "--ocr-assist",
        action="store_true",
        help="用视频画面中的歌词或硬字幕辅助校正时间轴",
    )
    parser.add_argument(
        "--refresh-ocr",
        action="store_true",
        help="忽略已有 OCR 缓存并重新扫描画面",
    )

    # ── 批量模式 ──────────────────────────────────────────────────
    parser.add_argument("--batch",      action="store_true", help="启用批量处理模式")
    parser.add_argument("--video-dir",  help="视频文件目录（批量模式）")
    parser.add_argument("--lyrics-dir", help="歌词文件目录（批量模式）")
    parser.add_argument("--output-dir", help="输出目录（批量模式，默认: video-dir/output）")

    args = parser.parse_args()

    try:
        font_settings(args.subtitle_font, args.subtitle_size)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.min_match_score <= 100:
        parser.error("--min-match-score 必须在 0 到 100 之间")

    if args.task == "speech":
        from speech_catalog import LANGUAGE_CODES
        from speech_subtitles import process_video, process_batch as process_speech_batch
        if args.model not in SUPPORTED_WHISPER_MODELS:
            parser.error("视频自动字幕请选择 Whisper 模型")
        if args.language not in ("auto", *LANGUAGE_CODES):
            parser.error("--language 需要填写 auto 或 Whisper 语言代码")
        if args.ocr_assist:
            parser.error("视频自动字幕不使用画面歌词辅助")
        options = dict(helpers=sys.modules[__name__], model_name=args.model, language=args.language,
                       no_vad=args.no_vad, no_burn=args.no_burn, refresh=args.refresh_transcript,
                       subtitle_font=args.subtitle_font, subtitle_size=args.subtitle_size)
        if args.batch:
            if not args.video_dir:
                parser.error("批量模式需要指定 --video-dir")
            process_speech_batch(Path(args.video_dir), Path(args.output_dir) if args.output_dir else None, **options)
        else:
            if not args.video:
                parser.error("单文件模式需要指定 --video")
            process_video(Path(args.video), Path(args.output) if args.output else None, **options)
        return

    if args.batch:
        if not args.video_dir or not args.lyrics_dir:
            parser.error("批量模式需要同时指定 --video-dir 和 --lyrics-dir")
        process_batch(
            Path(args.video_dir),
            Path(args.lyrics_dir),
            Path(args.output_dir) if args.output_dir else None,
            no_vad=args.no_vad,
            no_burn=args.no_burn,
            refresh_transcript=args.refresh_transcript,
            audio_source=args.audio_source,
            model_name=args.model,
            min_match_score=args.min_match_score,
            ocr_assist=args.ocr_assist,
            refresh_ocr=args.refresh_ocr,
            subtitle_font=args.subtitle_font,
            subtitle_size=args.subtitle_size,
        )
    else:
        if not args.video or not args.lyrics:
            parser.error("单文件模式需要同时指定 --video 和 --lyrics")
        process_single(
            Path(args.video),
            Path(args.lyrics),
            Path(args.output) if args.output else None,
            no_vad=args.no_vad,
            no_burn=args.no_burn,
            refresh_transcript=args.refresh_transcript,
            audio_source=args.audio_source,
            model_name=args.model,
            min_match_score=args.min_match_score,
            ocr_assist=args.ocr_assist,
            refresh_ocr=args.refresh_ocr,
            subtitle_font=args.subtitle_font,
            subtitle_size=args.subtitle_size,
        )


if __name__ == "__main__":
    main()
