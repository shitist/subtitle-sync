"""Application integration: paired lyrics in, Chinese SRT and one report out."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from . import PROFILE, COARSE_PROFILE
from .resources import MMS_REVISION, TIFA_COMMIT

SCHEMA = 1


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_metadata(video: Path, source_lines: list[str], audio_source: str, fingerprint) -> dict:
    return {
        "schema": SCHEMA, "profile": PROFILE,
        "video": fingerprint(video), "audio_source": audio_source,
        "source_sha256": hashlib.sha256(json.dumps(source_lines, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "mms_revision": MMS_REVISION, "tifa_code": TIFA_COMMIT,
        "precision": "float32", "timeline": "copyts-start-at-zero-aresample-v1",
    }


def valid_result(result: dict, count: int) -> bool:
    try:
        lines = result["lines"]
        duration = result["duration"]
        return len(lines) == count and all(
            item["line_id"] == index
            and isinstance(item["start"], (int, float)) and isinstance(item["end"], (int, float))
            and math.isfinite(item["start"]) and math.isfinite(item["end"])
            and 0 <= item["start"] < item["end"] <= duration + 0.05
            and item["stage"] in {"refined", "coarse_only"}
            for index, item in enumerate(lines)
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_cache(path: Path, expected: dict, count: int) -> dict | None:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        result = saved["acoustic_alignment"]
        if saved["metadata"] == expected and result["profile"] == PROFILE and valid_result(result, count):
            return result
        # v4 acoustic and phoneme evidence is compatible with v5 assembly.
        if result.get("profile") == "ja-mms-frame-tifa-local24-bounded-fp32-v4" and "refinement" in result:
            old = {k: v for k, v in saved["metadata"].items() if k != "profile"}
            new = {k: v for k, v in expected.items() if k != "profile"}
            if old == new:
                from .core import LineSpan, merge_refinement
                assembled = merge_refinement([LineSpan(**row) for row in result["coarse"]],
                    [LineSpan(**row) for row in result["refinement"]], result["duration"])
                result = {**result, "profile": PROFILE, "assembly_replay_from": result["profile"],
                          "lines": [row.record() for row in assembled]}
                if valid_result(result, count):
                    return result
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def load_coarse_cache(path: Path, expected: dict) -> dict | None:
    """Reuse the same acoustic stage across local-refiner revisions."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        old = {k: v for k, v in saved["metadata"].items() if k != "profile"}
        new = {k: v for k, v in expected.items() if k != "profile"}
        result = saved["acoustic_alignment"]
        compatible = result.get("coarse_profile") == COARSE_PROFILE or result.get("profile") in {
            "ja-mms-frame-tifa-local-fp32-v2", "ja-mms-frame-tifa-local-bounded-fp32-v3",
        }
        if old == new and compatible and result["models"]["coarse_revision"] == MMS_REVISION:
            return result
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def extract_timeline_audio(video: Path, wav: Path, helpers) -> None:
    # Preserve audio offset relative to the input container's origin. Delayed
    # audio gets initial silence; timestamps are not cut by VAD or renumbered.
    helpers._run([
        helpers._get_ffmpeg_path(), "-y", "-copyts", "-start_at_zero", "-i", str(video),
        "-vn", "-af", "aresample=async=1:first_pts=0", "-ar", "44100", "-ac", "2",
        "-f", "wav", str(wav),
    ], "FFmpeg 提取音频")


def chinese_items(result: dict, lyrics, item_type) -> list:
    if not valid_result(result, len(lyrics)):
        raise ValueError("歌词对齐未能为每一行给出有效时间，请查看 alignment.json")
    return [item_type(
        start=item["start"], end=item["end"], text=line.zh, source_text=line.ja,
        status="acoustic_refined" if item["stage"] == "refined" else "acoustic_coarse",
        review_reason=",".join(item.get("flags", [])),
    ) for item, line in zip(result["lines"], lyrics)]


def process_known_lyrics(video, lyrics_path, output, *, audio_source, no_burn, refresh, helpers) -> None:
    from .backend import align_audio, check_runtime
    helpers._banner(f"处理: {video.name}")
    helpers._step(1, 7, "加载中日歌词")
    _, lyrics = helpers.load_lyrics(lyrics_path, require_pairs=True)
    if not lyrics or not all(line.ja.strip() and line.zh.strip() for line in lyrics):
        raise ValueError("歌词声学对齐需要逐行对应的中日双语歌词")
    source_lines = [line.ja for line in lyrics]
    metadata = cache_metadata(video, source_lines, audio_source, helpers._file_fingerprint)
    report_path, srt_path = output.with_suffix(".alignment.json"), output.with_suffix(".srt")
    result = None if refresh else load_cache(report_path, metadata, len(lyrics))
    if result is None:
        coarse_cache = None if refresh else load_coarse_cache(report_path, metadata)
        check_runtime()
        with tempfile.TemporaryDirectory(prefix="subtitle_lyrics_") as temp:
            folder = Path(temp)
            wav = folder / "audio.wav"
            helpers._step(2, 7, "提取音频")
            extract_timeline_audio(video, wav, helpers)
            helpers._step(3, 7, "Demucs 人声分离" if audio_source == "vocals" else "使用原始音频")
            audio = helpers.separate_vocals(wav, folder) if audio_source == "vocals" else wav
            helpers._step(4, 7, "歌词声学定位与逐句精对齐")
            def checkpoint(partial):
                _write_json(report_path, {"metadata": metadata, "summary": {"mode": "lyrics_acoustic", "stage": "refinement_pending"},
                                          "acoustic_alignment": partial})
            result = align_audio(audio, source_lines, coarse_cache=coarse_cache, checkpoint=checkpoint)
    else:
        helpers._step(4, 7, "复用歌词对齐缓存")
    helpers._step(5, 7, "映射中文字幕")
    lines = result.get("lines", [])
    summary = {
        "mode": "lyrics_acoustic", "lines": len(lyrics),
        "refined_lines": sum(item.get("stage") == "refined" for item in lines),
        "coarse_only_lines": sum(item.get("stage") == "coarse_only" for item in lines),
        "review_lines": [item["line_id"] + 1 for item in lines if item.get("flags")],
        "accuracy_verified": False,
    }
    _write_json(report_path, {"metadata": metadata, "summary": summary, "acoustic_alignment": result,
                              "subtitles": [{"line": i + 1, "source": line.ja, "text": line.zh} for i, line in enumerate(lyrics)]})
    aligned = chinese_items(result, lyrics, helpers.AlignmentItem)
    helpers._step(6, 7, "生成 SRT 和对轴报告")
    helpers.generate_srt(aligned, srt_path)
    print(f"    歌词对齐完成: {summary['refined_lines']}/{len(lyrics)} 行完成精对齐", flush=True)
    if summary["review_lines"]:
        print(f"    注意：{len(summary['review_lines'])} 行需要复核，详见对轴报告", flush=True)
    if no_burn:
        helpers._step(7, 7, "仅生成 SRT")
    else:
        helpers._step(7, 7, "FFmpeg 烧入硬字幕")
        helpers.burn_subtitles(video, srt_path, output)
    print(f"  SRT 文件: {srt_path}\n  对轴报告: {report_path}", flush=True)

