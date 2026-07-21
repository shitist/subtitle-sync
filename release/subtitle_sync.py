#!/usr/bin/env python3
"""
subtitle_sync.py
================
日语歌曲中文字幕自动对轴工具  v2.0

工作流程:
  1. FFmpeg 提取音频
  2. Demucs 人声分离（去掉背景音乐，提升 Whisper 准确率）
  3. faster-whisper large-v3 独立识别日语并输出 word 级时间戳
  4. 智能对轴：
     - 若有日文歌词：fuzzy matching（rapidfuzz）日→日匹配，再映射到中文
     - 若仅中文歌词：优先使用英文锚点局部对齐，否则低置信度时间回退
  5. FFmpeg 将硬字幕烧入原视频

歌词文件格式（UTF-8）:
  纯中文模式（向后兼容）：
    世界的终结
    诞生的光芒
  中日双语模式（推荐，精度最高）：
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
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
# 强制禁止创建软链接，绕过 Windows 恶心的权限限制
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ── Windows 终端 UTF-8 修复 ────────────────────────────────────────────────────
# Windows 默认终端编码可能是 cp932（日文）或 cp936（中文），强制改为 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TRANSCRIPT_CACHE_SCHEMA = 2
DEFAULT_MIN_MATCH_SCORE = 55.0


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

def separate_vocals(audio_path: Path, work_dir: Path) -> Path:
    """
    用 Demucs htdemucs 模型分离人声。
    返回人声 WAV 文件路径。
    """
    # 确保 ffmpeg 目录在环境变量 PATH 中，否则 Demucs 内部会找不到 ffmpeg
    ffmpeg_exe = _get_ffmpeg_path()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    cmd = [
        sys.executable, "-m", "demucs",
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
    use_vad: bool = True,
) -> list:
    """
    用 faster-whisper large-v3 识别日语，返回片段列表。

    参数:
      audio_path:     人声 WAV 文件路径
      use_vad:        是否启用 VAD 静音过滤

    返回:
      片段列表，每个元素有 .start, .end, .text, .words（词级时间戳）
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        _die("未安装 faster-whisper，请运行: pip install faster-whisper")

    print("    正在加载 Whisper large-v3 模型（首次运行会下载，约 3GB）...")

    device, compute_type = _detect_device()
    print(f"    使用设备: {device} / {compute_type}")

    model = WhisperModel("large-v3", device=device, compute_type=compute_type)

    # ── 构建转录参数 ──
    transcribe_kwargs = {
        "language": "ja",
        "beam_size": 5,
        "word_timestamps": True,              # ★ 词级时间戳，边界精度从 ±0.5s → ±0.05s
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

def load_lyrics(lyrics_path: Path) -> tuple[str, list[LyricLine]]:
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
        return _align_with_text(words, lyric_lines, min_match_score)

    if words:
        latin_aligned = _align_with_latin_anchors(words, lyric_lines, min_match_score)
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


def _normalize_latin_anchor(text: str) -> str:
    """提取中日混合歌词中可与 Whisper 直接比较的英文和数字。"""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(re.findall(r"[a-z0-9]+", normalized))


def _align_with_latin_anchors(
    words: list[WordTiming],
    lyric_lines: list[LyricLine],
    min_match_score: float,
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
) -> list[AlignmentItem]:
    """按歌词顺序将每行日文匹配到连续的 Whisper word span。"""
    line_count = len(lyric_lines)
    word_count = len(words)
    infinity = float("inf")
    skip_word_cost = 0.16
    missing_line_cost = 1.05
    candidate_floor = max(30.0, min_match_score - 20.0)

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
        if score < min_match_score:
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
            start=words[start_word].start,
            end=words[end_word - 1].end,
            text=lyric_lines[index].zh,
            source_text=lyric_lines[index].ja,
            recognized_text=recognized,
            score=round(score, 2),
            status=status,
            word_probability=(
                round(word_probability, 4) if word_probability is not None else None
            ),
        )

    print(f"    文本锚点: {len(anchored)}/{line_count} 行（阈值 {min_match_score:.0f}）")
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
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int(round((seconds % 1) * 1000))
    # 毫秒溢出修正
    if ms >= 1000:
        ms -= 1000
        s  += 1
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
) -> dict:
    """保存逐行匹配依据与置信度，返回摘要。"""
    matched = [
        item
        for item in aligned
        if item.status in {"matched", "low_confidence", "latin_anchor"}
    ]
    scores = [item.score for item in matched if item.score is not None]
    word_probabilities = [
        item.word_probability
        for item in matched
        if item.word_probability is not None
    ]
    if any(item.status == "latin_anchor" for item in aligned):
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
    summary = {
        "mode": mode,
        "line_count": len(aligned),
        "matched_lines": len(matched),
        "matched_coverage": round(len(matched) / len(aligned), 4) if aligned else 0.0,
        "average_match_score": round(sum(scores) / len(scores), 2) if scores else None,
        "average_word_probability": (
            round(sum(word_probabilities) / len(word_probabilities), 4)
            if word_probabilities else None
        ),
        "minimum_match_score": effective_minimum,
        "needs_review": [
            index
            for index, item in enumerate(aligned, 1)
            if item.status != "matched"
        ],
    }
    payload = {
        "schema_version": 1,
        "summary": summary,
        "items": [
            {"index": index, **asdict(item)}
            for index, item in enumerate(aligned, 1)
        ],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"    对轴报告已生成: {output_path}")
    if mode in {"text_dp", "latin_anchor"}:
        label = "文本匹配" if mode == "text_dp" else "英文锚点"
        print(
            f"    {label}覆盖率: {summary['matched_coverage'] * 100:.1f}%"
            f"，平均分: {summary['average_match_score'] if scores else 'N/A'}"
        )
    else:
        print("    ⚠️  未提供可匹配的日文歌词，全部时间轴均为低置信度回退。")
    return summary


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    """
    用 FFmpeg 将 SRT 字幕硬烧入视频。
    字体: 微软雅黑 24pt，白字黑边，底部居中。
    """
    ffmpeg_exe = _get_ffmpeg_path()

    # Windows 路径需要转义冒号和反斜杠
    srt_str = str(srt_path).replace("\\", "/").replace(":", "\\:")

    style = (
        "FontName=Microsoft YaHei,"
        "FontSize=24,"
        "PrimaryColour=&H00FFFFFF,"   # 白色
        "OutlineColour=&H00000000,"   # 黑色描边
        "BackColour=&H80000000,"      # 半透明黑色背景
        "Bold=1,"
        "Outline=2,"
        "Shadow=1,"
        "MarginV=35,"
        "Alignment=2"                 # 底部居中
    )

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_str}':force_style='{style}'",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
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


def _transcript_metadata(
    video_path: Path,
    use_vad: bool,
    audio_source: str,
) -> dict:
    # 识别不再依赖歌词内容；同一音视频设置可在纯中文/双语歌词间复用缓存。
    return {
        "schema_version": TRANSCRIPT_CACHE_SCHEMA,
        "model": "large-v3",
        "language": "ja",
        "transcription_profile": "audio-ja-v2-no-lyric-prompt",
        "use_vad": use_vad,
        "audio_source": audio_source,
        "video": _file_fingerprint(video_path),
    }


def save_transcript_cache(path: Path, segments: list, metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "segments": [],
    }
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
        payload["segments"].append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": getattr(segment, "text", ""),
            "words": words,
        })
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"    Whisper 转录缓存已保存: {path}")


def load_transcript_cache(path: Path, expected_metadata: dict) -> Optional[list[SimpleSegment]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("metadata") != expected_metadata:
            print("    已有 Whisper 缓存与当前输入或设置不一致，将重新识别。")
            return None
        segments = []
        for raw_segment in payload.get("segments", []):
            segments.append(SimpleSegment(
                start=float(raw_segment["start"]),
                end=float(raw_segment["end"]),
                text=raw_segment.get("text", ""),
                words=[WordTiming(**word) for word in raw_segment.get("words", [])],
            ))
        if segments:
            print(f"    ♻️  复用 Whisper 转录缓存，共 {len(segments)} 个片段")
            return segments
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"    ⚠️  Whisper 缓存读取失败，将重新识别: {error}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def process_single(
    video_path: Path,
    lyrics_path: Path,
    output_path: Path | None = None,
    no_vad: bool = False,
    no_burn: bool = False,
    refresh_transcript: bool = False,
    audio_source: str = "vocals",
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
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

    _banner(f"处理: {video_path.name}")

    with tempfile.TemporaryDirectory(prefix="subtitle_sync_") as _tmpdir:
        tmpdir = Path(_tmpdir)

        _step(1, 7, "加载歌词")
        _ja_prompt, lyric_lines = load_lyrics(lyrics_path)

        cache_path = output_path.with_suffix(".whisper.json")
        metadata = _transcript_metadata(
            video_path,
            use_vad=not no_vad,
            audio_source=audio_source,
        )
        segments = None if refresh_transcript else load_transcript_cache(cache_path, metadata)

        if segments is None:
            _step(2, 7, "提取音频")
            audio_wav = tmpdir / "audio.wav"
            extract_audio(video_path, audio_wav)

            if audio_source == "vocals":
                _step(3, 7, "Demucs 人声分离")
                transcription_audio = separate_vocals(audio_wav, tmpdir)
            else:
                _step(3, 7, "使用原始混合音频（跳过 Demucs）")
                transcription_audio = audio_wav

            _step(4, 7, "Whisper large-v3 日语识别")
            segments = transcribe_audio(
                transcription_audio,
                use_vad=not no_vad,
            )
            save_transcript_cache(cache_path, segments, metadata)
        else:
            _step(2, 7, "复用 Whisper 缓存，跳过音频提取")
            _step(3, 7, "复用 Whisper 缓存，跳过人声分离")
            _step(4, 7, "复用 Whisper 缓存，跳过识别")

        _step(5, 7, "文本驱动时间轴映射")
        aligned = align_lyrics_to_segments(
            segments,
            lyric_lines,
            min_match_score=min_match_score,
        )

        # 生成 SRT（始终保存到输出视频同目录）
        srt_path = output_path.with_suffix(".srt")
        generate_srt(aligned, srt_path)

        _step(6, 7, "生成对轴置信度报告")
        report_path = output_path.with_suffix(".alignment.json")
        summary = save_alignment_report(aligned, report_path, min_match_score)

        if summary["mode"] == "text_dp" and summary["matched_coverage"] < 0.5:
            print("    ⚠️  文本锚点覆盖率低于 50%，建议先检查 alignment.json 和 SRT。")

        if no_burn:
            _step(7, 7, "仅生成 SRT，跳过视频烧录")
        else:
            _step(7, 7, "FFmpeg 烧入硬字幕")
            burn_subtitles(video_path, srt_path, output_path)

    print()
    if not no_burn:
        print(f"  ✅ 完成！输出文件: {output_path}")
    else:
        print("  ✅ 对轴完成（未烧录视频）")
    print(f"  📄 SRT 文件: {srt_path}")
    print(f"  📊 对轴报告: {report_path}")
    print(f"  ♻️  转录缓存: {cache_path}")
    print()


def process_batch(
    video_dir: Path,
    lyrics_dir: Path,
    output_dir: Path | None = None,
    no_vad: bool = False,
    no_burn: bool = False,
    refresh_transcript: bool = False,
    audio_source: str = "vocals",
    min_match_score: float = DEFAULT_MIN_MATCH_SCORE,
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
                min_match_score=min_match_score,
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
        description="日语歌曲中文字幕自动对轴工具 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件处理（纯中文歌词）
  py subtitle_sync.py --video song.mp4 --lyrics song.txt

  # 单文件处理（中日双语歌词，推荐！精度最高）
  py subtitle_sync.py --video song.mp4 --lyrics song_bilingual.txt

  # 指定输出
  py subtitle_sync.py --video song.mp4 --lyrics song.txt --output out.mp4

  # 批量处理
  py subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics
  py subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics --output-dir ./output

歌词文件格式:
  纯中文（向后兼容）：  每行一句中文
  中日双语（推荐）：    日文|中文  或  日文<Tab>中文
        """,
    )

    # ── 单文件模式 ────────────────────────────────────────────────
    parser.add_argument("--video",   help="输入视频文件路径")
    parser.add_argument("--lyrics",  help="歌词文本文件（UTF-8，支持纯中文或日文|中文双语格式）")
    parser.add_argument("--output",  help="输出视频路径（默认: 原文件名_subtitled + 原扩展名）")
    parser.add_argument("--no-vad",  action="store_true", help="禁用 VAD 静音过滤（不推荐）")
    parser.add_argument("--no-burn", action="store_true", help="只生成 SRT/报告，不烧录视频")
    parser.add_argument(
        "--refresh-transcript",
        action="store_true",
        help="忽略已有 Whisper 缓存并重新识别",
    )
    parser.add_argument(
        "--audio-source",
        choices=["vocals", "original"],
        default="vocals",
        help="Whisper 音源：Demucs 人声或原始混合音频（默认: vocals）",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=DEFAULT_MIN_MATCH_SCORE,
        help=f"日文文本锚点最低相似度（默认: {DEFAULT_MIN_MATCH_SCORE:.0f}）",
    )

    # ── 批量模式 ──────────────────────────────────────────────────
    parser.add_argument("--batch",      action="store_true", help="启用批量处理模式")
    parser.add_argument("--video-dir",  help="视频文件目录（批量模式）")
    parser.add_argument("--lyrics-dir", help="歌词文件目录（批量模式）")
    parser.add_argument("--output-dir", help="输出目录（批量模式，默认: video-dir/output）")

    args = parser.parse_args()

    if not 0 <= args.min_match_score <= 100:
        parser.error("--min-match-score 必须在 0 到 100 之间")

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
            min_match_score=args.min_match_score,
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
            min_match_score=args.min_match_score,
        )


if __name__ == "__main__":
    main()
