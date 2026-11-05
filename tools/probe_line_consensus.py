"""Audio-only local-line experiment. References are not accepted by this runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import math

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subtitle_sync as sync
from lyric_alignment.backend import (
    _load_tifa, _pipeline, _refine_window, prepare_pronunciations, read_audio,
)
from lyric_alignment.core import LineSpan, Window
from lyric_alignment.resources import prepare_tifa, sha256
from lyric_alignment.workflow import _write_json


def consensus(views, line_id):
    if len(views) != 2:
        return None
    left, right = views
    if not left or len(left) != len(right):
        return None
    common = []
    for a, b in zip(left, right):
        if any(a[k] != b[k] for k in ("line_id", "word", "phone")):
            return None
        start, end = max(a["start"], b["start"]), min(a["end"], b["end"])
        if not math.isfinite(start + end) or start >= end:
            return None
        common.append({**a, "start": start, "end": end})
    return LineSpan(line_id, common[0]["start"], common[-1]["end"], "refined",
                    coverage=1, flags=["phoneme_context_consensus"]), common


def voiced_expansion(base, candidate, times, margins):
    """Keep an endpoint if its expansion is not supported by non-gap audio."""
    decisions = {}
    for side in ("start", "end"):
        old, new = getattr(base, side), getattr(candidate, side)
        expands = new < old if side == "start" else new > old
        evidence = None
        if expands:
            selected = (times >= min(old, new)) & (times < max(old, new))
            evidence = float(np.mean(margins[selected])) if selected.any() else 0.0
            if evidence <= 0:
                setattr(candidate, side, old)
                candidate.flags.append(f"{side}_expansion_without_voice_support")
        decisions[side] = {"old": old, "proposed": new, "selected": getattr(candidate, side),
                           "vocal_log_odds": evidence}
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    source, model_dir = prepare_tifa()
    backend, vocabulary, config = _load_tifa(source, model_dir)
    pipeline = _pipeline(config, model_dir)
    for case in json.loads(args.manifest.read_text(encoding="utf-8")):
        if "reference" in case:
            parser.error("Reference subtitles belong in a separate evaluation manifest")
        folder = args.work_dir / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        evidence_dir = Path(case["evidence_dir"])
        saved = json.loads((evidence_dir / "engine.json").read_text(encoding="utf-8"))["acoustic_alignment"]
        audio_path = Path(case["audio"])
        audio = read_audio(audio_path, backend.sample_rate)
        duration = len(audio) / backend.sample_rate
        _, lyrics = sync.load_lyrics(Path(case["lyrics"]), require_pairs=True)
        words, roman = prepare_pronunciations(pipeline, [line.ja for line in lyrics])
        if roman != saved["romanized_lines"] or abs(duration - saved["duration"]) > .001:
            raise ValueError("Numeric evidence does not match this input")
        coarse = [LineSpan(**r) for r in saved["coarse"]]
        current = [LineSpan(**r) for r in saved["lines"]]
        frames = np.load(evidence_dir / "emissions.npz", allow_pickle=False)
        expected_hash = (str(frames["audio_sha256"]) if "audio_sha256" in frames else
                         json.loads((evidence_dir / "input-audio.json").read_text(encoding="utf-8"))["sha256"])
        actual_hash = sha256(audio_path)
        if actual_hash != expected_hash:
            raise ValueError("Audio waveform does not match cached MMS frames")
        # P(all roman letters) versus P(silence). Vocabulary is fixed by the
        # pinned MMS revision: |, a-z, apostrophe, UNK, PAD.
        values, times = frames["values"], frames["times"]
        margins = np.logaddexp.reduce(values[:, 1:28], axis=1) - values[:, 29]
        records = []
        for i, base in enumerate(coarse):
            low = coarse[i - 1].start if i else 0.0
            high = coarse[i + 1].end if i + 1 < len(coarse) else duration
            views, attempts = [], []
            for padding in (2.0, 4.0):
                left, right = max(low, base.start - padding), min(high, base.end + padding)
                if right - left > 24:
                    continue
                window = Window(left, right, (i,), (i,))
                rows, phones, agreement = _refine_window(backend, vocabulary, audio, words, window, coarse,
                                                        ownership_bounds=(low, high))
                attempts.append({"window": [left, right], "phones": phones, "agreement": agreement})
                if len(rows) == 1 and rows[0].start is not None and rows[0].end is not None:
                    views.append(phones)
            # Equal crops are one view, not independent context evidence.
            if len(attempts) == 2 and attempts[0]["window"] == attempts[1]["window"]:
                views = []
            accepted = consensus(views, i)
            decision = None
            if accepted is not None:
                candidate, common = accepted
                decision = voiced_expansion(base, candidate, times, margins)
                if candidate.start < candidate.end:
                    current[i] = candidate
            records.append({"line_id": i, "accepted": accepted is not None,
                            "decision": decision, "attempts": attempts})
            print(f"CONSENSUS {case['id']} {i+1}/{len(coarse)}", flush=True)
        output = {**saved, "lines": [r.record() for r in current], "local_line_checks": records,
                  "audio_sha256": actual_hash, "experiment": "two-context-phoneme-intersection-v1"}
        _write_json(folder / "engine.json", {"acoustic_alignment": output})
        sync.generate_srt([sync.AlignmentItem(r.start, r.end, l.zh) for r, l in zip(current, lyrics)], folder / "generated.srt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
