"""FP32 MMS frame alignment followed by context-limited TIFA phoneme alignment."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np

from . import PROFILE, COARSE_PROFILE
from .core import (
    LineSpan, aggregate_phones, coarse_lines, decode_frames, make_targets,
    line_bounds, merge_refinement, normalize_roman, plan_windows,
)
from .resources import MMS_REPO, MMS_REVISION, TIFA_COMMIT, prepare_mms, prepare_tifa


def check_runtime() -> None:
    import importlib
    modules = ("torch", "transformers", "g2pflow", "fugashi", "unidic", "numba", "num2words", "librosa", "soundfile", "omegaconf", "pydantic", "einops", "onnxruntime")
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError("歌词对齐环境缺少: " + ", ".join(missing) + "；请在 .venv-align 中安装 requirements-align.txt")


def _load_tifa_config(source: Path, model_dir: Path):
    # Import the pinned upstream numerical backend; avoid Lightning/Trainer and
    # its global device selection. This subprocess owns these generic module names.
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from lib.config.io import load_raw_config
    from lib.config.schema import InferenceConfig, ConfigurationScope
    raw = load_raw_config(model_dir / "config.yaml")
    config = InferenceConfig.model_validate(raw["inference"], scope=ConfigurationScope.FA)
    return raw, config


def _load_tifa(source: Path, model_dir: Path):
    import torch
    raw, config = _load_tifa_config(source, model_dir)
    from lib.config.schema import ModelConfig, ConfigurationScope
    from lib.vocabulary import Vocabulary
    from inference.backend import ForcedAlignmentInferenceModel
    model_config = ModelConfig.model_validate(raw["model"], scope=ConfigurationScope.FA)
    backend = ForcedAlignmentInferenceModel(model_config, config)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    backend.load_state_dict(checkpoint["state_dict"], strict=True)
    backend.float().eval()
    vocabulary = Vocabulary.from_file(model_dir / "vocabulary.json")
    return backend, vocabulary, config


def _pipeline(config, model_dir):
    from g2pflow import G2PPipelineConfig, build_pipeline_from_config
    # Only build the languages needed for Japanese lyrics mixed with English.
    raw = config.g2p.model_dump()
    raw["converters"] = [item for item in raw["converters"] if
        item["id"] in {"japanese-mecab", "lstm"}
        or item["id"] == "dictionary" and "ja" in (item.get("language") or "")]
    return build_pipeline_from_config(G2PPipelineConfig.model_validate(raw), root_path=model_dir)


def roman_units(word) -> list[str]:
    from .numbers import NumberWord
    if isinstance(word, NumberWord):
        return normalize_roman(word.romans[word.chosen]).split()
    # English labels are the original alphabetic words used by this karaoke
    # checkpoint. Japanese pronunciation groups come from the MeCab reading.
    if re.fullmatch(r"[A-Za-z0-9'’\- ]+", word.text):
        return normalize_roman(word.text).split()
    groups = word.readings[0].paths[0]
    scripts = [group.script.casefold() for group in groups]
    result = []
    closure = ""
    for index, script in enumerate(scripts):
        if script in {"cl", "q"}:
            following = scripts[index + 1] if index + 1 < len(scripts) else ""
            closure += following[:1] if following[:1] not in "aeiou" else ""
        else:
            result.append(normalize_roman(closure + ("n" if script in {"um", "nn"} else script)))
            closure = ""
    if closure:
        result.append(normalize_roman(closure))
    return result


def _roman_word(word) -> str:
    from .numbers import NumberWord
    if isinstance(word, NumberWord):
        return normalize_roman(word.romans[word.chosen])
    if re.fullmatch(r"[A-Za-z0-9'’\- ]+", word.text):
        return normalize_roman(word.text)
    groups = word.readings[0].paths[0]
    scripts = [group.script.casefold() for group in groups]
    result = []
    for index, script in enumerate(scripts):
        if script in {"cl", "q"}:
            following = scripts[index + 1] if index + 1 < len(scripts) else ""
            result.append(following[:1] if following[:1] not in "aeiou" else "")
        else:
            result.append("n" if script in {"um", "nn"} else script)
    return normalize_roman("".join(result))


def spoken_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    # Explicit kana reading, e.g. 声明(こえ), replaces that word's reading only.
    # This normalization is internal; source/translation output is kept verbatim.
    text = re.sub(r"[一-龯々]+\(([ぁ-ゖァ-ヺー]+)\)", r"\1", text)
    from .numbers import number_spans
    numeric = {i for start, end in number_spans(text) for i in range(start, end)}
    result = []
    for index, char in enumerate(text):
        internal_apostrophe = char == "'" and 0 < index < len(text) - 1 and text[index - 1].isascii() and text[index - 1].isalpha() and text[index + 1].isascii() and text[index + 1].isalpha()
        result.append(" " if unicodedata.category(char).startswith(("S", "P"))
                      and not internal_apostrophe and index not in numeric else char)
    return "".join(result)


def _convert_available_words(pipeline, text: str, *, causes: list | None = None):
    """Retry only unclaimed input spans; retain every convertible word in order."""
    from g2pflow.converters.base import G2PConversionError
    from .numbers import convert_numbers
    remaining = spoken_text(text)
    skipped = []
    while remaining.strip():
        try:
            return convert_numbers(remaining, pipeline, _roman_word, spoken_text), remaining, skipped
        except G2PConversionError as error:
            if causes is not None and error.reason:
                causes.append(error.reason)
            previous = remaining
            for token in error.unconverted_tokens:
                pattern = r"\s+".join(re.escape(part) for part in token.split())
                if not pattern:
                    continue
                pattern = r"(?<![A-Za-z0-9])" + pattern + r"(?![A-Za-z0-9])"
                matches = list(re.finditer(pattern, remaining, flags=re.IGNORECASE))
                skipped.extend(match.group().strip() for match in matches)
                remaining = re.sub(pattern, lambda match: " " * len(match.group()), remaining,
                                   flags=re.IGNORECASE)
            if remaining == previous:
                # The converter may return a transformed token with no source
                # offsets. Mark this line unresolved rather than dropping other lines.
                skipped.append(remaining.strip())
                return [], "", skipped
    return [], "", skipped


def prepare_pronunciations(pipeline, source_lines: list[str], *, issues: list | None = None):
    words_by_line, roman_lines = [], []
    for line_id, text in enumerate(source_lines):
        if not text.strip():
            raise ValueError(f"第 {line_id + 1} 行缺少日文原词")
        try:
            causes = []
            words, alignment_text, skipped = _convert_available_words(pipeline, text, causes=causes)
            roman = normalize_roman(" ".join(_roman_word(word) for word in words))
        except Exception as error:
            raise ValueError(f"第 {line_id + 1} 行读音转换失败: {text}\n{error}") from error
        if skipped or not words or not roman:
            spoken = spoken_text(text)
            missing_start = (len(alignment_text) - len(alignment_text.lstrip())
                             > len(spoken) - len(spoken.lstrip())) if alignment_text.strip() else True
            missing_end = len(alignment_text.rstrip()) < len(spoken.rstrip()) if alignment_text.strip() else True
            issue = {"line_id": line_id, "source": text, "unconverted_tokens": skipped,
                     "alignment_text": alignment_text, "missing_start": missing_start, "missing_end": missing_end,
                     "status": "partial" if words and roman else "unresolved"}
            if issues is not None:
                issues.append(issue)
            detail = "、".join(skipped) or text
            if causes:
                reason = " ".join(dict.fromkeys(causes))
            elif any(re.search(r"[A-Za-z]", token) and re.search(r"\d", token) for token in skipped):
                reason = f"模型不支持 {detail} 中的数字。"
            elif skipped and all(re.fullmatch(r"[\d\s,.+-]+", token) for token in skipped):
                reason = "缺少数字读音规则。"
            else:
                reason = f"找不到 {detail} 的读音。"
            issue["reason"] = reason
            print(f"    第 {line_id + 1} 行：{reason}", flush=True)
            if not words or not roman:
                words, roman = [], ""
        words_by_line.append(words)
        roman_lines.append(roman)
    return words_by_line, roman_lines


def _mark_pronunciation_issues(lines, issues):
    by_id = {issue["line_id"]: issue for issue in issues}
    for line in lines:
        issue = by_id.get(line.line_id)
        if issue is None:
            continue
        if issue["status"] == "unresolved":
            if "g2p_unresolved" not in line.flags:
                line.flags.append("g2p_unresolved")
            line.start = line.end = None
            line.stage = "unresolved"
        else:
            for boundary in ("start", "end"):
                flag = "g2p_missing_" + boundary
                if issue["missing_" + boundary] and flag not in line.flags:
                    line.flags.append(flag)


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf
    import librosa
    waveform, original_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if original_rate != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=original_rate, target_sr=sample_rate)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("音频为空或包含无效采样")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _mms_probabilities(audio: np.ndarray, model_path: Path, progress):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    model = Wav2Vec2ForCTC.from_pretrained(model_path, dtype=torch.float32).eval()
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    stride = math.prod(model.config.conv_stride)
    receptive, scale = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel - 1) * scale
        scale *= step
    frame_step = stride / 16000
    # Core intervals partition the timeline; context frames are discarded once.
    core_samples, context = 24 * 16000, 2 * 16000
    all_values, all_times = [], []
    count = math.ceil(len(audio) / core_samples)
    for index, core_start in enumerate(range(0, len(audio), core_samples)):
        core_end = min(len(audio), core_start + core_samples)
        left, right = max(0, core_start - context), min(len(audio), core_end + context)
        # Extremely short final tails are covered by the previous window.
        if right - left < receptive:
            continue
        inputs = extractor(audio[left:right], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            values = model(**inputs).logits[0].float().log_softmax(-1).cpu().numpy()
        centers = left + np.arange(len(values)) * stride + (receptive - 1) / 2
        selected = (centers >= core_start) & (centers < core_end)
        all_values.append(values[selected])
        all_times.append(np.maximum(0, centers[selected] / 16000 - frame_step / 2))
        progress("整曲定位", index + 1, count)
    del model
    gc.collect()
    values, times = np.concatenate(all_values), np.concatenate(all_times)
    if len(times) > 1 and not np.allclose(np.diff(times), frame_step, atol=1e-6):
        raise RuntimeError("音频窗口拼接存在缺帧或重复帧")
    return values, times, frame_step


def _refine_window(backend, vocabulary, audio, words_by_line, window, coarse, *, ownership_bounds=None):
    import torch
    from lib.g2p_encoding import encode_paths
    from lib.path_traversal import materialize_paths
    from modules.decoding import decode_alignment_flat
    words, word_lines = [], []
    for line_id in window.context_ids:
        words.extend(words_by_line[line_id])
        word_lines.extend([line_id] * len(words_by_line[line_id]))
    data, lexicon, _ = encode_paths(words, vocabulary, languages=["ja", "en"], default_language="ja")
    packed = {name: torch.from_numpy(value).unsqueeze(0) for name, value in data.items()}
    left, right = round(window.start * backend.sample_rate), round(window.end * backend.sample_rate)
    excerpt = torch.from_numpy(audio[left:right]).unsqueeze(0)
    actual_duration = excerpt.shape[-1] / backend.sample_rate
    with torch.inference_mode():
        spec = backend.spectrogram(excerpt, torch.tensor([actual_duration]))
        scored = backend.score(spec, paths=packed["paths"], words=packed["words"], candidates=packed["candidates"], unit="levenshtein")
        tokens, owners, groups = materialize_paths(packed["paths"], packed["words"], packed["groups"], scored.choices)
        aligned = backend.align(spec, tokens=tokens, groups=groups, unit="frame", skip_penalty=0.5)
        # Local phonetic decoding must respect the global line ownership. Without
        # this constraint a 40s phoneme path can slide into neighboring lyrics.
        constrained = aligned.similarity.clone()
        times = torch.arange(constrained.shape[1]) * backend.timestep + left / backend.sample_rate
        for token_index, owner in enumerate(owners[0].tolist()):
            if not owner:
                continue
            low, high = (ownership_bounds if ownership_bounds is not None else
                         line_bounds(coarse, word_lines[owner - 1], len(audio) / backend.sample_rate))
            outside = (times < low - backend.timestep) | (times >= high + backend.timestep)
            constrained[0, outside, token_index] = -10000.0
        bounded = decode_alignment_flat(constrained, spec.mask.sum(-1), (tokens != 0).sum(-1), groups=groups, skip_penalty=0.5)
    spans = bounded[0].cpu().numpy()
    similarity = aligned.similarity[0].cpu().numpy()
    selected_phones = [phoneme for candidates, choice in zip(lexicon, scored.choices[0].tolist())
                       if choice > 0 for phoneme in candidates[choice - 1]["phonemes"]]
    phone_records = []
    by_line, scores = {}, {}
    for token_index, owner in enumerate(owners[0].tolist()):
        if not owner or not tokens[0, token_index]:
            continue
        line_id = word_lines[owner - 1]
        a, b = spans[token_index]
        by_line.setdefault(line_id, []).append((float(a * backend.timestep), float(b * backend.timestep)))
        if b > a:
            scores.setdefault(line_id, []).append(float(similarity[a:b, token_index].mean()))
        phone_records.append({
            "line_id": line_id, "word": words[owner - 1].text, "phone": selected_phones[token_index],
            "start": left / backend.sample_rate + float(a * backend.timestep),
            "end": left / backend.sample_rate + float(b * backend.timestep),
        })
    return ([aggregate_phones(line_id, by_line.get(line_id, []), scores.get(line_id, []),
                              left / backend.sample_rate, actual_duration) for line_id in window.keep_ids],
            phone_records, float(aligned.agreement[0]))


def _reading_likelihood(values, roman_lines, vocab):
    targets, _ = make_targets(roman_lines, vocab, allow_empty=True)
    try:
        path = decode_frames(values, targets, vocab["[PAD]"])
    except ValueError:
        return -np.inf
    emitted = np.where(path >= 0, targets[np.maximum(path, 0)], vocab["[PAD]"])
    return float(values[np.arange(len(values)), emitted].mean())


def select_number_readings(values, times, vocab, words_by_line, coarse, progress):
    from .numbers import NumberWord
    duration = float(times[-1] + (times[1] - times[0] if len(times) > 1 else .02))
    numbered = [(line_id, words) for line_id, words in enumerate(words_by_line)
                if any(isinstance(word, NumberWord) for word in words)]
    records = []
    for position, (line_id, words) in enumerate(numbered):
        previous = next((line for line in reversed(coarse[:line_id]) if line.start is not None), None)
        following = next((line for line in coarse[line_id + 1:] if line.end is not None), None)
        first, last = previous.line_id if previous else line_id, following.line_id if following else line_id
        low = previous.start if previous else 0.0
        high = following.end if following else duration
        low, high = max(0.0, low), min(duration, high)
        if high <= low:
            raise ValueError(f"第 {line_id + 1} 行：数字音频区间为空")
        excerpt = values[np.searchsorted(times, low):np.searchsorted(times, high, side="right")]
        context = words_by_line[first:last + 1]
        numeric_indices = [index for index, word in enumerate(words) if isinstance(word, NumberWord)]
        def score():
            romans = [normalize_roman(" ".join(_roman_word(word) for word in row)) for row in context]
            return _reading_likelihood(excerpt, romans, vocab)
        # Beam search preserves joint numeric choices within a line. Each
        # candidate is scored on the same audio frames, including blank frames.
        beam = [(0.0, ())]
        for offset, index in enumerate(numeric_indices):
            expanded = []
            for _, prefix in beam:
                for previous_index, choice in zip(numeric_indices, prefix):
                    words[previous_index].chosen = choice
                for choice in range(len(words[index].readings)):
                    words[index].chosen = choice
                    expanded.append((score(), prefix + (choice,)))
            beam = sorted(expanded, key=lambda item: (-item[0], item[1]))[:8]
        best_score, choices = beam[0]
        if not np.isfinite(best_score):
            raise ValueError(f"第 {line_id + 1} 行：数字音频区间没有匹配读音")
        for index, choice in zip(numeric_indices, choices):
            words[index].chosen = choice
        for index, choice in zip(numeric_indices, choices):
            word = words[index]
            scores = []
            for candidate in range(len(word.readings)):
                word.chosen = candidate
                value = score()
                scores.append(value if np.isfinite(value) else None)
            word.chosen = choice
            records.append({"line_id": line_id, "word_index": index, "text": word.text,
                            "reading": word.labels[choice], "roman": word.romans[choice],
                            "window": [low, high], "method": "mms_frame_likelihood",
                            "candidates": [{"reading": label, "roman": roman, "score": value}
                                           for label, roman, value in zip(word.labels, word.romans, scores)]})
            word.select(choice)
        progress("数字读音", position + 1, len(numbered))
    return records


def align_audio(audio_path: Path, source_lines: list[str], progress=None, *, coarse_cache=None, checkpoint=None,
                frame_checkpoint=None) -> dict:
    """Only source lyrics and audio enter inference; Chinese/reference SRT never do."""
    import torch
    from importlib.metadata import version
    if progress is None:
        progress = lambda phase, current, total: print(f"歌词对齐进度: {phase} {current}/{total}", flush=True)
    check_runtime()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    print("    正在加载歌词对齐模型（FP32）", flush=True)
    source, model_dir = prepare_tifa()
    _, config = _load_tifa_config(source, model_dir)
    pipeline = _pipeline(config, model_dir)
    pronunciation_issues = []
    words_by_line, roman_lines = prepare_pronunciations(pipeline, source_lines, issues=pronunciation_issues)
    from .numbers import NumberWord
    has_numbers = any(isinstance(word, NumberWord) for words in words_by_line for word in words)
    if has_numbers:
        coarse_cache = None
    audio16 = read_audio(audio_path, 16000)
    duration = len(audio16) / 16000
    if not any(roman_lines):
        lines = [LineSpan(index, None, None, "unresolved", flags=["g2p_unresolved"])
                 for index in range(len(source_lines))]
        return {"profile": PROFILE, "duration": duration, "romanized_lines": roman_lines,
                "pronunciation_issues": pronunciation_issues, "lines": [line.record() for line in lines]}
    mms_path = prepare_mms()
    coarse = None
    if coarse_cache and coarse_cache.get("romanized_lines") == roman_lines and abs(coarse_cache.get("duration", -1) - duration) < 0.001:
        try:
            candidate = [LineSpan(**item) for item in coarse_cache["coarse"]]
            if len(candidate) == len(source_lines) and all(item.line_id == i and item.start is not None and item.end is not None
                    and 0 <= item.start < item.end <= duration + 0.05 for i, item in enumerate(candidate)):
                coarse = candidate
                print("    复用整曲声学定位缓存", flush=True)
        except (TypeError, KeyError):
            pass
    if coarse is None:
        values, times, frame_step = _mms_probabilities(audio16, mms_path, progress)
        if frame_checkpoint:
            frame_checkpoint(values, times, frame_step)
        import json
        vocab = json.loads((mms_path / "vocab.json").read_text(encoding="utf-8"))
        targets, owners = make_targets(roman_lines, vocab, allow_empty=True)
        path = decode_frames(values, targets, vocab["[PAD]"])
        coarse = coarse_lines(path, owners, values, targets, times, frame_step, len(source_lines))
        del path
        if not has_numbers:
            del values
    backend = vocabulary = audio48 = None
    number_readings = []
    if has_numbers:
        number_readings = select_number_readings(values, times, vocab, words_by_line, coarse, progress)
        roman_lines = [normalize_roman(" ".join(_roman_word(word) for word in words)) for words in words_by_line]
        targets, owners = make_targets(roman_lines, vocab, allow_empty=True)
        path = decode_frames(values, targets, vocab["[PAD]"])
        coarse = coarse_lines(path, owners, values, targets, times, frame_step, len(source_lines))
        del values, path
    _mark_pronunciation_issues(coarse, pronunciation_issues)
    base_result = {
        "profile": PROFILE, "coarse_profile": COARSE_PROFILE, "precision": "float32", "duration": duration,
        "runtime": {name: version(name) for name in ("torch", "transformers", "g2pflow", "numpy", "numba")},
        "models": {"coarse": MMS_REPO, "coarse_revision": MMS_REVISION, "refiner": "TIFA-1.0", "tifa_code": TIFA_COMMIT},
        "romanized_lines": roman_lines, "coarse": [item.record() for item in coarse],
        "pronunciation_issues": pronunciation_issues,
        "number_readings": number_readings,
    }
    if checkpoint:
        checkpoint(base_result)
    windows = plan_windows(coarse, duration)
    del audio16
    gc.collect()
    if backend is None:
        backend, vocabulary, _ = _load_tifa(source, model_dir)
        audio48 = read_audio(audio_path, backend.sample_rate)
    refined = []
    window_records = []
    for index, window in enumerate(windows):
        try:
            rows, phones, agreement = _refine_window(backend, vocabulary, audio48, words_by_line, window, coarse)
            refined.extend(rows)
            window_records.append({"start": window.start, "end": window.end, "lines": list(window.keep_ids),
                                   "agreement": agreement, "phones": phones})
        except (ValueError, RuntimeError) as error:
            window_records.append({"start": window.start, "end": window.end, "lines": list(window.keep_ids), "error": str(error)})
            print(f"    精对齐窗口 {index + 1} 失败: {error}", flush=True)
        progress("逐句精对齐", index + 1, len(windows))
    result = merge_refinement(coarse, refined, duration)
    _mark_pronunciation_issues(result, pronunciation_issues)
    return {
        **base_result,
        "refinement": [item.record() for item in refined],
        "lines": [item.record() for item in result], "windows": window_records,
    }

