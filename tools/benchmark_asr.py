"""在同一音频上对比 ASR 文字与字幕时间；参考歌词仅用于完成转录后的评分。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subtitle_sync as sync
from asr_models import WHISPER_MODELS


def read_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        match = re.fullmatch(
            r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)", lines[1],
        )
        if not match:
            raise ValueError(f"无效的参考 SRT 时间：{lines[1]}")
        values = list(map(int, match.groups()))
        cues.append({
            "start": values[0]*3600 + values[1]*60 + values[2] + values[3]/1000,
            "end": values[4]*3600 + values[5]*60 + values[6] + values[7]/1000,
            "text": "\n".join(lines[2:]),
        })
    return cues


def score(segments, lyrics, reference_cues) -> dict:
    from rapidfuzz.distance import Levenshtein

    expected = sync._normalize_japanese("".join(line.ja for line in lyrics))
    observed = sync._normalize_japanese("".join(segment.text for segment in segments))
    if not expected:
        raise ValueError("文字评分必须提供日文原词")
    operations = Levenshtein.editops(expected, observed)
    counts = {name: sum(op.tag == name for op in operations) for name in ("replace", "delete", "insert")}
    aligned = sync.align_lyrics_to_segments(segments, lyrics)
    if len(aligned) != len(reference_cues):
        raise ValueError("歌词与参考 SRT 行数不一致")
    errors = [
        abs(getattr(item, side) - reference[side])
        for item, reference in zip(aligned, reference_cues)
        for side in ("start", "end")
    ]
    ordered = sorted(errors)
    return {
        "reference_characters": len(expected),
        "recognized_characters": len(observed),
        "character_errors": len(operations),
        "cer": len(operations)/len(expected),
        "substitutions": counts["replace"], "deletions": counts["delete"], "insertions": counts["insert"],
        "cues": len(aligned), "boundaries": len(errors),
        "boundary_mae_seconds": sum(errors)/len(errors),
        "boundary_p90_seconds": ordered[min(len(ordered)-1, int((len(ordered)-1)*0.9))],
        "boundary_max_seconds": max(errors),
        "boundaries_within_0_5": sum(error <= 0.5 for error in errors),
        "boundaries_within_1": sum(error <= 1 for error in errors),
        "matched_lines": sum(item.status == "matched" for item in aligned),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lyrics", type=Path)
    source.add_argument("--source-lines", type=Path, help="与参考 SRT 逐行对应的日文原词")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=WHISPER_MODELS, required=True)
    parser.add_argument("--no-vad", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 转录函数只接收音频与通用设置；评分参考在它返回后才打开。
    started = time.monotonic()
    segments = sync.transcribe_audio(args.audio, args.model, not args.no_vad)
    elapsed = time.monotonic() - started
    cache = args.output_dir / f"{args.model}.json"
    sync.save_transcript_cache(cache, segments, {
        "model": args.model, "audio": sync._file_fingerprint(args.audio),
        "use_vad": not args.no_vad, "elapsed_seconds": elapsed,
    })
    reference = read_srt(args.reference)
    if args.lyrics:
        _, lyrics = sync.load_lyrics(args.lyrics)
    else:
        source_lines = [line.strip() for line in args.source_lines.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(source_lines) != len(reference):
            raise ValueError("日文原词与参考 SRT 行数不一致")
        lyrics = [sync.LyricLine(line, cue["text"]) for line, cue in zip(source_lines, reference)]
    result = score(segments, lyrics, reference)
    result.update(model=args.model, elapsed_seconds=elapsed, use_vad=not args.no_vad, ocr=False)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    (args.output_dir / f"{args.model}.metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
