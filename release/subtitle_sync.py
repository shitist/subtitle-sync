#!/usr/bin/env python3
"""
subtitle_sync.py
================
日语歌曲中文字幕自动对轴工具  v2.0

工作流程:
  1. FFmpeg 提取音频
  2. Demucs 人声分离（去掉背景音乐，提升 Whisper 准确率）
  3. faster-whisper large-v3 识别日语（支持日文歌词 prompt 引导 + word 级时间戳）
  4. 智能对轴：
     - 若有日文歌词：fuzzy matching（rapidfuzz）日→日匹配，再映射到中文
     - 若仅中文歌词：动态规划序列对齐 + 按比例分配
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
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional
# 强制禁止创建软链接，绕过 Windows 恶心的权限限制
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ── Windows 终端 UTF-8 修复 ────────────────────────────────────────────────────
# Windows 默认终端编码可能是 cp932（日文）或 cp936（中文），强制改为 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
    initial_prompt: Optional[str] = None,
    use_vad: bool = True,
) -> list:
    """
    用 faster-whisper large-v3 识别日语，返回片段列表。

    参数:
      audio_path:     人声 WAV 文件路径
      initial_prompt: 日文歌词文本，引导 Whisper 识别（大幅提升准确率）
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

    if initial_prompt:
        # 截断 prompt（Whisper 限制约 224 token，日文约 100 字）
        prompt = initial_prompt[:300]
        transcribe_kwargs["initial_prompt"] = prompt
        print(f"    使用日文歌词 prompt（{len(prompt)} 字符）引导识别")

    if use_vad:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=200,       # ★ 200ms（歌曲换气约 150-300ms，比默认 500ms 更敏感）
            min_speech_duration_ms=100,        # 最短语音段 100ms（避免吞掉短促音节）
            threshold=0.35,                    # VAD 灵敏度（0.0-1.0，越低越敏感）
            speech_pad_ms=100,                 # 语音段前后各留 100ms 余量
        )
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_gen, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    segments = list(segments_gen)
    print(f"    Whisper 识别完成，共 {len(segments)} 个片段")
    for i, seg in enumerate(segments):
        word_info = ""
        if hasattr(seg, "words") and seg.words:
            word_info = f"  [words: {len(seg.words)}]"
        print(f"      [{i+1:03d}] {seg.start:.2f}s → {seg.end:.2f}s{word_info}  {seg.text.strip()}")

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
#   用户提供的歌词文件中每一行中文对应一个字幕条目，
#   不会因 Whisper 识别结果多/少而合并或拆分歌词行。
#   日文歌词仅用于引导 Whisper 识别（initial_prompt），不参与对轴计算。
# ──────────────────────────────────────────────────────────────────────────────

def load_lyrics(lyrics_path: Path) -> tuple[str, list[str]]:
    """
    从 UTF-8 文本文件读取歌词，自动识别三种格式。

    返回:
      (ja_prompt_text, zh_lines)
      - ja_prompt_text: 日文歌词拼接文本（仅用于 Whisper prompt），无日文时为空串
      - zh_lines:        中文歌词行列表（★ 每行一个字幕，不截断不合并）

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
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    if not raw_lines:
        _die("歌词文件为空")

    # ── 格式 1: 逐行分隔符（| 或 Tab）──
    has_separator = any("|" in ln or "\t" in ln for ln in raw_lines)
    if has_separator:
        ja_list, zh_list = _parse_line_bilingual(raw_lines)
        ja_prompt = "。".join(j for j in ja_list if j)
        print(f"    检测到逐行双语歌词：中文 {len(zh_list)} 行")
        return ja_prompt, zh_list

    # ── 格式 2 vs 3: 假名检测 ──
    kana_flags = [_has_japanese_kana(ln) for ln in raw_lines]
    kana_count = sum(kana_flags)

    if kana_count == 0:
        # 纯中文
        print(f"    检测到纯中文歌词，共 {len(raw_lines)} 行")
        return "", raw_lines

    # 有假名 → 找分块分割点
    split_idx = kana_flags.index(True)

    # 验证分块格式的有效性
    if split_idx > 0 and any(kana_flags[:split_idx]):
        print(f"    检测到纯中文歌词（含少量假名标注），共 {len(raw_lines)} 行")
        return "", raw_lines

    after_kana = kana_flags[split_idx:]
    after_ratio = sum(after_kana) / len(after_kana) if after_kana else 0
    if after_ratio < 0.5:
        print(f"    检测到纯中文歌词，共 {len(raw_lines)} 行")
        return "", raw_lines

    # ── 确认是分块格式 ──
    zh_lines = raw_lines[:split_idx]   # ★ 全部中文行，不截断
    ja_lines = raw_lines[split_idx:]   # 日文行（仅用于 prompt）

    ja_prompt = "。".join(j for j in ja_lines if j)
    print(f"    检测到中/日分块歌词：中文 {len(zh_lines)} 行 + 日文 {len(ja_lines)} 行")

    if len(zh_lines) != len(ja_lines):
        print(f"    💡 中/日行数不一致，日文仅用于辅助 Whisper 识别，"
              f"最终字幕 = {len(zh_lines)} 行中文")

    return ja_prompt, zh_lines


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
            parts = [line, line]  # 无分隔符，日文=中文
        ja = parts[0].strip()
        zh = parts[1].strip() if len(parts) > 1 else ja
        ja_lyrics.append(ja)
        zh_lyrics.append(zh)
    return ja_lyrics, zh_lyrics


def align_lyrics_to_segments(
    segments: list,
    zh_lines: list[str],
) -> list[tuple]:
    """
    将 Whisper 时间轴按比例分配给中文歌词行。

    ★ 核心原则：一行歌词 = 一个字幕轴，不合并不拆分。
    
    精度提升策略：
      1. 先按 word 间时间间隙（>300ms）拆分过长的 segment
         （歌曲连续唱时 Whisper 会把多句合并，需要拆开）
      2. 再对子段做比例分配

    返回: [(start_sec, end_sec, chinese_text), ...]
    """
    # ── 步骤 0: 按 word 间隙拆分长 segment ──
    sub_segments = _split_segments_at_gaps(segments, gap_threshold=0.3)
    n_subs = len(sub_segments)
    n_lyrs = len(zh_lines)

    print(f"    Whisper 片段: {len(segments)} → word-gap 拆分后: {n_subs} 子段")
    print(f"    歌词行数: {n_lyrs}")
    print(f"    对轴模式: 比例分配（每行中文 = 一个字幕）")

    if n_subs == 0:
        _die("未找到有效音频片段。")

    if n_subs == n_lyrs:
        print(f"    子模式: 1:1 直接映射")
        return [
            (_get_word_start(s), _get_word_end(s), lyr)
            for s, lyr in zip(sub_segments, zh_lines)
        ]

    elif n_subs > n_lyrs:
        print(f"    子模式: 合并子段（{n_subs}→{n_lyrs}）")
        result = []
        for i, lyr in enumerate(zh_lines):
            start_idx = round(i * n_subs / n_lyrs)
            end_idx = round((i + 1) * n_subs / n_lyrs)
            end_idx = min(end_idx, n_subs)
            start_idx = min(start_idx, end_idx - 1)

            start_t = _get_word_start(sub_segments[start_idx])
            end_t = _get_word_end(sub_segments[end_idx - 1])
            # 单条字幕不超过 15 秒
            if end_t - start_t > 15:
                end_t = start_t + 15
            result.append((start_t, end_t, lyr))
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
                result.append((t_start, t_end, zh_lines[lyr_idx]))
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


def generate_srt(aligned: list[tuple], output_path: Path) -> None:
    """生成标准 SRT 字幕文件。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(aligned, 1):
            f.write(f"{idx}\n")
            f.write(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n")
            f.write(f"{text}\n\n")
    print(f"    SRT 文件已生成: {output_path}")


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


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def process_single(
    video_path: Path,
    lyrics_path: Path,
    output_path: Path | None = None,
    no_vad: bool = False,
) -> None:
    """处理单个视频文件。"""
    video_path  = Path(video_path)
    lyrics_path = Path(lyrics_path)

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_subtitled{video_path.suffix}"
    output_path = Path(output_path)

    _banner(f"处理: {video_path.name}")

    with tempfile.TemporaryDirectory(prefix="subtitle_sync_") as _tmpdir:
        tmpdir = Path(_tmpdir)

        # ── 0. 预加载歌词（获取日文 prompt）──────────────────────────────────
        _step(1, 6, "加载歌词")
        ja_prompt, zh_lines = load_lyrics(lyrics_path)

        # 构建日文 prompt（用于引导 Whisper 识别）
        initial_prompt: Optional[str] = ja_prompt if ja_prompt else None
        if initial_prompt:
            print(f"    日文歌词 prompt 共 {len(initial_prompt)} 字符")

        # ── 1. 提取音频 ──────────────────────────────────────────────────────
        _step(2, 6, "提取音频")
        audio_wav = tmpdir / "audio.wav"
        extract_audio(video_path, audio_wav)

        # ── 2. 人声分离 ──────────────────────────────────────────────────────
        _step(3, 6, "Demucs 人声分离")
        vocals_wav = separate_vocals(audio_wav, tmpdir)

        # ── 3. Whisper 识别 ──────────────────────────────────────────────────
        _step(4, 6, "Whisper large-v3 日语识别")
        segments = transcribe_audio(
            vocals_wav,
            initial_prompt=initial_prompt,
            use_vad=not no_vad,
        )

        # ── 4. 对轴（一行歌词 = 一个字幕轴）────────────────────────────────
        _step(5, 6, "时间轴映射")
        aligned = align_lyrics_to_segments(segments, zh_lines)

        # 生成 SRT（始终保存到输出视频同目录）
        srt_path = output_path.with_suffix(".srt")
        generate_srt(aligned, srt_path)

        # ── 5. 烧入字幕 ──────────────────────────────────────────────────────
        _step(6, 6, "FFmpeg 烧入硬字幕")
        burn_subtitles(video_path, srt_path, output_path)

    print()
    print(f"  ✅ 完成！输出文件: {output_path}")
    print(f"  📄 SRT 文件: {srt_path}")
    print()


def process_batch(
    video_dir: Path,
    lyrics_dir: Path,
    output_dir: Path | None = None,
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
            process_single(video_path, lyrics_path, output_path)
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
    parser.add_argument("--output",  help="输出视频路径（默认: 原文件名_subtitled.mp4）")
    parser.add_argument("--no-vad",  action="store_true", help="禁用 VAD 静音过滤（不推荐）")

    # ── 批量模式 ──────────────────────────────────────────────────
    parser.add_argument("--batch",      action="store_true", help="启用批量处理模式")
    parser.add_argument("--video-dir",  help="视频文件目录（批量模式）")
    parser.add_argument("--lyrics-dir", help="歌词文件目录（批量模式）")
    parser.add_argument("--output-dir", help="输出目录（批量模式，默认: video-dir/output）")

    args = parser.parse_args()

    if args.batch:
        if not args.video_dir or not args.lyrics_dir:
            parser.error("批量模式需要同时指定 --video-dir 和 --lyrics-dir")
        process_batch(
            Path(args.video_dir),
            Path(args.lyrics_dir),
            Path(args.output_dir) if args.output_dir else None,
        )
    else:
        if not args.video or not args.lyrics:
            parser.error("单文件模式需要同时指定 --video 和 --lyrics")
        process_single(
            Path(args.video),
            Path(args.lyrics),
            Path(args.output) if args.output else None,
            no_vad=args.no_vad,
        )


if __name__ == "__main__":
    main()
