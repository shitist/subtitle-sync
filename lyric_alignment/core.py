"""Timeline, decoding and line ownership. No ASR or reference subtitles here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
import unicodedata
import numpy as np


@dataclass
class LineSpan:
    line_id: int
    start: float | None
    end: float | None
    stage: str
    coverage: float = 0.0
    acoustic_score: float | None = None
    flags: list[str] = field(default_factory=list)

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    context_ids: tuple[int, ...]
    keep_ids: tuple[int, ...]


def normalize_roman(text: str) -> str:
    # Same alphabet/normalization as the MMS checkpoint's training notebook.
    text = unicodedata.normalize("NFKD", text.casefold().replace("’", "'")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub("[^a-z']", " ", text)).strip()


def make_targets(roman_lines: list[str], vocab: dict[str, int], *, allow_empty: bool = False) -> tuple[np.ndarray, np.ndarray]:
    tokens, owners = [], []
    for line_id, text in enumerate(roman_lines):
        normalized = normalize_roman(text)
        if not normalized:
            if allow_empty:
                continue
            raise ValueError(f"第 {line_id + 1} 行无法转换为模型支持的读音")
        if tokens:
            tokens.append(vocab["|"])
            owners.append(-1)  # A between-line separator belongs to neither subtitle.
        for char in normalized:
            tokens.append(vocab["|" if char == " " else char])
            owners.append(line_id)
    return np.asarray(tokens, dtype=np.int32), np.asarray(owners, dtype=np.int32)


def word_gap_mask(target: np.ndarray, separator_id: int) -> np.ndarray:
    """Silence may separate words, but cannot split a word across an interlude.

    Repeated letters remain separate required label states. Label self-loops
    still support sustained singing; no duration is assigned from text length.
    """
    allowed = np.ones(len(target) + 1, dtype=np.bool_)
    allowed[1:-1] = (target[:-1] == separator_id) | (target[1:] == separator_id)
    return allowed


def make_grouped_targets(roman_lines: list[str], units_by_line: list[list[str]],
                         vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep pronunciation groups contiguous, while permitting sung syllable gaps.

    Japanese groups come from G2P mora boundaries, not guessed durations or
    character counts. English uses the supplied G2P written-word groups.
    """
    if len(roman_lines) != len(units_by_line):
        raise ValueError("读音分组行数不一致")
    target, owners = make_targets(roman_lines, vocab)
    allowed = word_gap_mask(target, vocab["|"])
    cursor = 0
    for roman, units in zip(roman_lines, units_by_line):
        normalized = [normalize_roman(unit).replace(" ", "") for unit in units]
        if "".join(normalized) != normalize_roman(roman).replace(" ", ""):
            raise ValueError("读音分组与声学目标不一致")
        for unit in normalized:
            if not unit:
                continue
            while cursor < len(target) and target[cursor] == vocab["|"]:
                cursor += 1
            for char in unit:
                while cursor < len(target) and target[cursor] == vocab["|"]:
                    cursor += 1
                if cursor >= len(target) or target[cursor] != vocab[char]:
                    raise ValueError("读音分组与词表不一致")
                cursor += 1
            allowed[cursor] = True
    return target, owners, allowed


def acoustic_gap_costs(path: np.ndarray, target: np.ndarray, group_gaps: np.ndarray,
                       separator_id: int) -> np.ndarray:
    """A soft word-internal wait prior scaled by observed pronunciation units.

    One unit of cost per median unit duration. Its scale comes from allocated
    AUDIO frames, not lyric length, tempo settings, or reference subtitles.
    Long real pauses remain possible; word/line boundaries have zero cost.
    """
    counts = np.bincount(path[path >= 0], minlength=len(target))
    durations, current = [], 0
    for i, token in enumerate(target):
        if token != separator_id:
            current += counts[i]
        if group_gaps[i + 1]:
            if current:
                durations.append(current)
            current = 0
    if not durations:
        raise ValueError("声学路径没有有效读音分组")
    median_frames = float(np.median(durations))
    return (~word_gap_mask(target, separator_id)).astype(np.float32) / median_frames


def decode_frames(log_probs: np.ndarray, target: np.ndarray, gap_id: int,
                  gap_allowed: np.ndarray | None = None,
                  gap_costs: np.ndarray | None = None) -> np.ndarray:
    """Frame-supervised monotonic FA, not CTC collapse.

    Label states may follow identical labels directly. Optional gap states keep
    intro/interlude/silence independent of caption duration. Every target label
    is visited; no implicit text deletion or duration-based interpolation.
    """
    from numba import njit
    values = np.ascontiguousarray(log_probs, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.int32)
    allowed = (np.ones(len(target) + 1, dtype=np.bool_) if gap_allowed is None
               else np.ascontiguousarray(gap_allowed, dtype=np.bool_))
    costs = (np.zeros(len(target) + 1, dtype=np.float32) if gap_costs is None
             else np.ascontiguousarray(gap_costs, dtype=np.float32))
    if values.ndim != 2 or not len(target) or values.shape[0] < len(target):
        raise ValueError("音频帧数不足以容纳歌词")
    if not np.isfinite(values).all() or min(target.min(), gap_id) < 0 or max(target.max(), gap_id) >= values.shape[1]:
        raise ValueError("无效的声学概率或词表")
    if allowed.shape != (len(target) + 1,) or not allowed[0] or not allowed[-1]:
        raise ValueError("无效的词间空档约束")
    if costs.shape != allowed.shape or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("无效的词内空档代价")
    return njit(cache=True)(_frame_viterbi)(values, target, gap_id, allowed, costs)


def _frame_viterbi(values, target, gap_id, gap_allowed, gap_costs):
    frames, _ = values.shape
    count = len(target)
    states = count * 2 + 1  # gap_0, label_0, gap_1, label_1, ... gap_N
    previous = np.full(states, -np.inf, dtype=np.float32)
    previous[0] = 0.0
    trace = np.zeros((frames, states), dtype=np.uint8)
    for t in range(frames):
        current = np.full(states, -np.inf, dtype=np.float32)
        for s in range(states):
            if s % 2 == 0 and not gap_allowed[s // 2]:
                continue
            best = previous[s]
            step = 0
            if s > 0 and previous[s - 1] > best:
                best, step = previous[s - 1], 1
            # Gap states cannot skip required label states; label states may skip optional gaps.
            if s % 2 and s > 1 and previous[s - 2] > best:
                best, step = previous[s - 2], 2
            label = target[s // 2] if s % 2 else gap_id
            current[s] = best + values[t, label] - (gap_costs[s // 2] if s % 2 == 0 else 0.0)
            trace[t, s] = step
        previous = current
    state = states - 1 if previous[-1] > previous[-2] else states - 2
    if not np.isfinite(previous[state]):
        raise ValueError("整曲歌词声学路径不存在")
    path = np.full(frames, -1, dtype=np.int32)
    for t in range(frames - 1, -1, -1):
        if state % 2:
            path[t] = state // 2
        state -= trace[t, state]
    return path


def coarse_lines(path, owners, probabilities, target, frame_times, frame_step, count) -> list[LineSpan]:
    result = []
    for line_id in range(count):
        indices = np.flatnonzero(owners == line_id)
        present = np.flatnonzero(np.isin(path, indices))
        if not len(present):
            result.append(LineSpan(line_id, None, None, "unresolved", flags=["no_acoustic_path"]))
            continue
        selected = path[present]
        score = float(np.exp(probabilities[present, target[selected]]).mean())
        result.append(LineSpan(
            line_id, float(frame_times[present[0]]), float(frame_times[present[-1]] + frame_step),
            "coarse", coverage=len(np.unique(selected)) / len(indices), acoustic_score=score,
        ))
    return result


def plan_windows(lines: list[LineSpan], duration: float, max_seconds: float = 24.0, padding: float = 2.0) -> list[Window]:
    """Keep each line once; add whole neighboring lines only when they fit.

    Never split a line's text using guessed time proportions. An overlong line
    stays coarse-only and is explicitly flagged by the caller.
    """
    if not 0 < max_seconds <= 55 or padding < 0:
        raise ValueError("精对齐窗口必须不超过 55 秒")
    valid = [line for line in lines if line.start is not None and line.end is not None]
    windows = []
    position = 0
    while position < len(valid):
        first = position
        last = first
        left = max(0.0, valid[first].start - padding)
        right = min(duration, valid[first].end + padding)
        if right - left > max_seconds:
            position += 1
            continue
        while last + 1 < len(valid):
            proposed = min(duration, valid[last + 1].end + padding)
            if proposed - left > max_seconds - 8.0:
                break
            last += 1
            right = proposed
        keep = tuple(item.line_id for item in valid[first:last + 1])
        context_first, context_last = first, last
        if first and right - max(0.0, valid[first - 1].start - padding) <= max_seconds:
            context_first -= 1
            left = max(0.0, valid[context_first].start - padding)
        if last + 1 < len(valid) and min(duration, valid[last + 1].end + padding) - left <= max_seconds:
            context_last += 1
            right = min(duration, valid[context_last].end + padding)
        windows.append(Window(left, right, tuple(item.line_id for item in valid[context_first:context_last + 1]), keep))
        position = last + 1
    return windows


def aggregate_phones(line_id: int, spans: list[tuple[float, float]], scores: list[float], offset: float, duration: float) -> LineSpan:
    valid = [(a, b) for a, b in spans if math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration + 0.02]
    coverage = len(valid) / len(spans) if spans else 0.0
    flags = []
    if len(valid) != len(spans):
        flags.append("skipped_phonemes")
    # Missing edge phones must not silently shrink a subtitle's boundaries.
    if not valid or len(valid) != len(spans) and (spans[0] not in valid or spans[-1] not in valid):
        return LineSpan(line_id, None, None, "unresolved", coverage, flags=flags + ["missing_edge_phoneme"])
    return LineSpan(line_id, offset + valid[0][0], offset + valid[-1][1], "refined", coverage,
                    float(np.mean(scores)) if scores else None, flags)


def line_bounds(lines: list[LineSpan], line_id: int, duration: float) -> tuple[float, float]:
    """A refiner cannot consume frames acoustically owned by another line.

    Gaps remain available for phoneme boundary refinement. These constraints
    come exclusively from the whole-song audio path, never from a reference SRT.
    """
    previous = next((item for item in reversed(lines[:line_id]) if item.end is not None), None)
    following = next((item for item in lines[line_id + 1:] if item.start is not None), None)
    return previous.end if previous else 0.0, following.start if following else duration


def merge_refinement(coarse: list[LineSpan], refined: list[LineSpan], duration: float) -> list[LineSpan]:
    by_id = {item.line_id: item for item in refined}
    if len(by_id) != len(refined):
        raise ValueError("精对齐返回了重复行号")
    merged = []
    for base in coarse:
        item = by_id.get(base.line_id)
        outside_support = (
            item is not None and item.start is not None and item.end is not None
            and base.start is not None and base.end is not None
            and (item.start < base.start - 0.02 or item.end > base.end + 0.02)
        )
        if item is None or item.start is None or item.end is None or outside_support:
            # One MMS frame is 20ms. Refinement cannot create vocal support
            # outside the independently decoded coarse interval. This prevents
            # a phoneme tail from consuming an otherwise silent interlude.
            reason = "refinement_outside_acoustic_support" if outside_support else "refinement_unresolved"
            merged.append(LineSpan(base.line_id, base.start, base.end, "coarse_only", base.coverage,
                                   base.acoustic_score, [reason] + (item.flags if item else [])))
        else:
            merged.append(item)
    for index, item in enumerate(merged):
        if item.start is None or item.end is None or not 0 <= item.start < item.end <= duration + 0.05:
            item.flags.append("invalid_span")
        if index and item.start is not None and merged[index - 1].end is not None and item.start < merged[index - 1].end - 0.02:
            item.flags.append("overlapping_lines")
            merged[index - 1].flags.append("overlapping_lines")
    return merged

