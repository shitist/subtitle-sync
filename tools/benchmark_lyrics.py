"""Known-lyrics engine regression. Reference SRT is read ONLY after inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subtitle_sync as sync
from lyric_alignment.backend import align_audio
from lyric_alignment.workflow import _write_json, cache_metadata, load_cache, load_coarse_cache, chinese_items, extract_timeline_audio
from tools.regression_subtitles import evaluate
from tools.benchmark_asr import read_srt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--save-emissions", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        folder = args.work_dir / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        print(f"CASE {case['id']}", flush=True)
        try:
            _, lyrics = sync.load_lyrics(Path(case["lyrics"]), require_pairs=True)
            video = Path(case["video"])
            metadata = cache_metadata(video, [line.ja for line in lyrics], case.get("audio_source", "vocals"), sync._file_fingerprint)
            if case.get("deterministic_separation") and case.get("audio_source", "vocals") == "vocals":
                metadata["audio_preparation"] = sync.DEMUCS_PROFILE
            cache = folder / "engine.json"
            result = load_cache(cache, metadata, len(lyrics))
            run_mode = "full_cache" if result is not None else "cold"
            if result is None and case.get("inference_cache"):
                result = load_cache(Path(case["inference_cache"]), metadata, len(lyrics))
                if result is not None:
                    run_mode = "assembly_replay"
                    _write_json(cache, {"metadata": metadata, "acoustic_alignment": result})
                    print("REPLAY acoustic + phoneme evidence; assemble current line policy", flush=True)
            if result is None:
                coarse_cache = load_coarse_cache(cache, metadata)
                if coarse_cache is None and case.get("coarse_cache"):
                    coarse_cache = load_coarse_cache(Path(case["coarse_cache"]), metadata)
                if coarse_cache is not None:
                    run_mode = "coarse_cache_then_refinement"
                # Optional audio is an independently extracted/separated copy, never a reference subtitle.
                if case.get("audio"):
                    audio = Path(case["audio"])
                else:
                    audio = folder / "audio.wav"
                    extract_timeline_audio(video, audio, sync)
                    if case.get("audio_source", "vocals") == "vocals":
                        audio = sync.separate_vocals(audio, folder, deterministic=case.get("deterministic_separation", False))
                def save_frames(values, times, step):
                    import numpy as np
                    from lyric_alignment.resources import sha256
                    np.savez_compressed(folder / "emissions.npz", values=values, times=times,
                                        step=step, audio_sha256=sha256(audio))
                result = align_audio(audio, [line.ja for line in lyrics], coarse_cache=coarse_cache,
                    checkpoint=lambda partial: _write_json(cache, {"metadata": metadata, "acoustic_alignment": partial}),
                    frame_checkpoint=save_frames if args.save_emissions else None)
                _write_json(cache, {"metadata": metadata, "acoustic_alignment": result})
            aligned = chinese_items(result, lyrics, sync.AlignmentItem)
            actual_path = folder / "generated.srt"
            sync.generate_srt(aligned, actual_path)
            # Ground truth is deliberately outside align_audio's argument list.
            metrics = evaluate(read_srt(actual_path), read_srt(Path(case["reference"])))
            metrics.update(case=case["id"], seconds=time.monotonic() - start,
                           run_mode=run_mode, assembly_replay_from=result.get("assembly_replay_from"),
                           inputs={key: str(Path(case[key]).resolve()) for key in ("video", "lyrics", "reference")},
                           coarse_only=sum(item["stage"] == "coarse_only" for item in result["lines"]),
                           flagged=sum(bool(item["flags"]) for item in result["lines"]))
            _write_json(folder / "metrics.json", metrics)
            results.append(metrics)
            print("RESULT", json.dumps({k: v for k, v in metrics.items() if k != "rows"}, ensure_ascii=False), flush=True)
        except Exception as error:
            traceback.print_exc()
            results.append({"case": case["id"], "initial_pass": False, "error": str(error)})
        _write_json(args.work_dir / "report.json", {"results": results, "all_pass": all(r["initial_pass"] for r in results)})
    return 0 if results and all(r["initial_pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
