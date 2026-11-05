"""Audio-only experiment: compare equal-weight mixed/vocal MMS evidence.

This runner never accepts or opens reference subtitles. Its numeric evidence
can be scored separately, after every case has completed inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subtitle_sync as sync
from lyric_alignment.backend import (
    _load_tifa_config, _mms_probabilities, _pipeline, prepare_pronunciations, read_audio, roman_units,
)
from lyric_alignment.core import acoustic_gap_costs, coarse_lines, decode_frames, make_targets, make_grouped_targets, word_gap_mask
from lyric_alignment.resources import prepare_mms, prepare_tifa, sha256
from lyric_alignment.workflow import _write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--redecode", action="store_true", help="Replay this experiment's saved emissions and readings")
    parser.add_argument("--word-contiguous", action="store_true")
    parser.add_argument("--pronunciation-contiguous", action="store_true")
    parser.add_argument("--internal-gap-prior", action="store_true",
                        help="One-second exponential prior on intra-word silent waits, not a hard cap")
    parser.add_argument("--adaptive-gap-prior", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    if any("reference" in case for case in cases):
        parser.error("Reference subtitles belong in a separate evaluation manifest")
    import torch
    torch.set_num_threads(4)
    source, model_dir = prepare_tifa()
    _, config = _load_tifa_config(source, model_dir)
    pipeline = _pipeline(config, model_dir)
    if not args.redecode:
        mms_path = prepare_mms()
    else:
        # No download/check at HF is needed to decode existing numeric frames.
        from huggingface_hub import snapshot_download
        from lyric_alignment.resources import MMS_REPO, MMS_REVISION
        mms_path = Path(snapshot_download(MMS_REPO, revision=MMS_REVISION,
                                         allow_patterns=["vocab.json"], local_files_only=True))
    vocab = json.loads((mms_path / "vocab.json").read_text(encoding="utf-8"))
    for case in cases:
        folder = args.work_dir / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        _, lyrics = sync.load_lyrics(Path(case["lyrics"]), require_pairs=True)
        if args.redecode:
            previous = json.loads((folder / "audio-evidence.json").read_text(encoding="utf-8"))
            roman = previous["romanized_lines"]
            if len(roman) != len(lyrics):
                raise ValueError("Saved experiment readings do not match lyric line count")
            _, current_roman = prepare_pronunciations(pipeline, [line.ja for line in lyrics])
            if current_roman != roman:
                raise ValueError("Saved experiment readings do not match the current source lyrics")
        else:
            _, roman = prepare_pronunciations(pipeline, [line.ja for line in lyrics])
        gap_allowed = None
        if args.pronunciation_contiguous:
            words, fresh_roman = prepare_pronunciations(pipeline, [line.ja for line in lyrics])
            if fresh_roman != roman:
                raise ValueError("Saved experiment readings no longer match G2P")
            units = [[unit for word in line for unit in roman_units(word)] for line in words]
            target, owners, gap_allowed = make_grouped_targets(roman, units, vocab)
        else:
            target, owners = make_targets(roman, vocab)
        if args.word_contiguous:
            gap_allowed = word_gap_mask(target, vocab["|"])
        evidence = {}
        provenance = {}
        for name in ("original", "vocals"):
            audio_path = Path(case[name])
            if args.redecode and not audio_path.is_file():
                old_input = previous["inputs"][name]
                if str(audio_path.resolve()) != old_input["path"]:
                    raise ValueError("Cached numeric evidence belongs to a different audio path")
                digest = old_input["sha256"]
            else:
                digest = sha256(audio_path)
            cache = folder / f"{name}.npz"
            if cache.is_file():
                with np.load(cache, allow_pickle=False) as saved:
                    if str(saved["sha256"]) == digest:
                        evidence[name] = (saved["values"], saved["times"], float(saved["step"]))
            if name not in evidence:
                if args.redecode:
                    raise ValueError("Numeric evidence is missing or its input audio changed")
                print(f"MMS {case['id']} {name}", flush=True)
                audio = read_audio(audio_path, 16000)
                evidence[name] = _mms_probabilities(audio, mms_path,
                    lambda phase, current, total: print(f"{phase} {current}/{total}", flush=True))
                values, times, step = evidence[name]
                np.savez_compressed(cache, values=values, times=times, step=step, sha256=digest)
            provenance[name] = {"path": str(audio_path.resolve()), "sha256": digest}
        original, times, step = evidence["original"]
        vocals, vocal_times, vocal_step = evidence["vocals"]
        if original.shape != vocals.shape or step != vocal_step or not np.allclose(times, vocal_times):
            raise ValueError("Audio views must preserve identical sample counts and frame times")
        # Symmetric product of experts. No per-case weights or reference data.
        combined = (original + vocals) * np.float32(0.5)
        variants = {}
        suffix = "-mora" if args.pronunciation_contiguous else "-word" if args.word_contiguous else ""
        if args.internal_gap_prior:
            suffix += "-prior"
        if args.adaptive_gap_prior:
            if not args.pronunciation_contiguous:
                raise ValueError("Adaptive prior requires pronunciation groups")
            suffix += "-adaptive"
        for name, values in (("original", original), ("vocals", vocals), ("equal_fusion", combined)):
            # A duration prior, not a maximum: a sung word can still contain a
            # genuine pause. Instrumental gaps BETWEEN words are not penalized.
            costs = (~word_gap_mask(target, vocab["|"])).astype(np.float32) * step if args.internal_gap_prior else None
            path = decode_frames(values, target, vocab["[PAD]"], gap_allowed, costs)
            if args.adaptive_gap_prior:
                costs = acoustic_gap_costs(path, target, gap_allowed, vocab["|"])
                path = decode_frames(values, target, vocab["[PAD]"], gap_allowed, costs)
            lines = coarse_lines(path, owners, values, target, times, step, len(lyrics))
            variants[name] = [line.record() for line in lines]
            sync.generate_srt([sync.AlignmentItem(start=row.start, end=row.end, text=lyric.zh)
                               for row, lyric in zip(lines, lyrics)], folder / f"{name}{suffix}.srt")
        _write_json(folder / (f"{suffix[1:]}-evidence.json" if suffix else "audio-evidence.json"), {
            "inputs": provenance, "romanized_lines": roman, "variants": variants,
            "policy": "fixed-equal-log-probabilities-v1", "precision": "float32",
            "word_contiguous": args.word_contiguous,
            "pronunciation_contiguous": args.pronunciation_contiguous,
            "internal_gap_prior_seconds": 1.0 if args.internal_gap_prior else None,
            "adaptive_gap_prior": args.adaptive_gap_prior,
        })
        print(f"DONE {case['id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
