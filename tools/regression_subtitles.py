"""比较生成 SRT 与标准 SRT：每条起止误差均不超过一秒才通过。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_asr import read_srt


def evaluate(actual: list[dict], reference: list[dict]) -> dict:
    if len(actual) != len(reference) or not reference:
        return {"initial_pass": False, "error": "字幕条数与标准不一致", "cues": len(actual), "reference_cues": len(reference)}
    rows = []
    for index, (cue, target) in enumerate(zip(actual, reference), 1):
        start_error = round((cue["start"] - target["start"]) * 1000)
        end_error = round((cue["end"] - target["end"]) * 1000)
        maximum = max(abs(start_error), abs(end_error))
        rows.append({
            "line": index, "text": target["text"],
            "reference": [target["start"], target["end"]], "actual": [cue["start"], cue["end"]],
            "start_error_ms": start_error, "end_error_ms": end_error,
            "pass": maximum <= 1000, "maximum_error_ms": maximum,
        })
    errors = [abs(row[key]) for row in rows for key in ("start_error_ms", "end_error_ms")]
    return {
        "initial_pass": all(row["pass"] for row in rows), "cues": len(rows),
        "within_one_second": sum(row["pass"] for row in rows),
        "within_half_second": sum(row["maximum_error_ms"] <= 500 for row in rows),
        "boundary_mae_seconds": sum(errors) / len(errors) / 1000,
        "boundary_max_seconds": max(errors) / 1000,
        "failed_lines": [row["line"] for row in rows if not row["pass"]], "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(read_srt(args.actual), read_srt(args.reference))
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0 if result["initial_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
