import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import zipfile

import numpy as np

from lyric_alignment import PROFILE
from lyric_alignment.core import (
    LineSpan, aggregate_phones, coarse_lines, decode_frames, make_targets,
    line_bounds, merge_refinement, normalize_roman, plan_windows, word_gap_mask, make_grouped_targets, acoustic_gap_costs,
)
from lyric_alignment.resources import safe_extract
from lyric_alignment.workflow import cache_metadata, chinese_items, load_cache, load_coarse_cache, valid_result
import subtitle_sync as sync
import subtitle_sync_gui as gui
from asr_models import LYRIC_MODEL


class FrameAlignmentTests(unittest.TestCase):
    def test_repeated_identical_letters_are_distinct_targets_without_ctc_blank(self):
        values = np.log(np.array([[.99, .01]] * 4, dtype=np.float32))
        path = decode_frames(values, np.array([0, 0], dtype=np.int32), 1)
        self.assertEqual(set(path), {0, 1})
        self.assertTrue(np.all(np.diff(path) >= 0))

    def test_instrumental_gap_does_not_extend_lyric_to_next_line(self):
        values = np.log(np.array([
            [.01, .01, .98], [.98, .01, .01], [.98, .01, .01],
            [.01, .01, .98], [.01, .01, .98], [.01, .01, .98],
            [.01, .98, .01], [.01, .98, .01], [.01, .01, .98],
        ], dtype=np.float32))
        targets, owners = np.array([0, 1]), np.array([0, 1])
        path = decode_frames(values, targets, 2)
        np.testing.assert_array_equal(path, [-1, 0, 0, -1, -1, -1, 1, 1, -1])
        lines = coarse_lines(path, owners, values, targets, np.arange(9), 1, 2)
        self.assertEqual((lines[0].start, lines[0].end), (1, 3))
        self.assertEqual((lines[1].start, lines[1].end), (6, 8))

    def test_repeat_lyrics_keep_original_line_ids(self):
        targets, owners = make_targets(["Yeah", "Yeah"], {c: i for i, c in enumerate("|abcdefghijklmnopqrstuvwxyz'")})
        self.assertEqual(owners.tolist(), [0] * 4 + [-1] + [1] * 4)
        np.testing.assert_array_equal(targets[:4], targets[-4:])

    def test_bad_emissions_and_too_short_audio_fail(self):
        with self.assertRaises(ValueError):
            decode_frames(np.zeros((1, 2)), np.array([0, 1]), 1)
        with self.assertRaises(ValueError):
            decode_frames(np.array([[np.nan, 0]]), np.array([0]), 1)

    def test_training_normalization_preserves_apostrophes(self):
        self.assertEqual(normalize_roman("  Dón’t! Yeah～  "), "don't yeah")

    def test_word_cannot_start_in_noise_then_resume_after_long_silence(self):
        values = np.log(np.array([
            [.70, .01, .29], [.01, .01, .98], [.01, .01, .98],
            [.01, .01, .98], [.60, .01, .39], [.01, .98, .01],
        ], dtype=np.float32))
        target = np.array([0, 1])
        unconstrained = decode_frames(values, target, 2)
        constrained = decode_frames(values, target, 2, word_gap_mask(target, 3))
        self.assertEqual(np.flatnonzero(unconstrained == 0)[0], 0)
        self.assertEqual(np.flatnonzero(constrained == 0)[0], 4)

    def test_word_gaps_preserve_interludes_and_repeated_characters(self):
        target = np.array([0, 0, 2, 1])
        np.testing.assert_array_equal(word_gap_mask(target, 2), [True, False, True, True, True])
        values = np.log(np.array([
            [.98, .01, .005, .005], [.98, .01, .005, .005],
            [.005, .005, .98, .01], [.005, .005, .01, .98],
            [.005, .005, .01, .98], [.01, .98, .005, .005],
        ], dtype=np.float32))
        path = decode_frames(values, target, 3, word_gap_mask(target, 2))
        self.assertEqual(set(path), {-1, 0, 1, 2, 3})

    def test_invalid_word_gap_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "空档"):
            decode_frames(np.zeros((3, 3)), np.array([0, 1]), 2, np.ones(2))

    def test_pronunciation_groups_allow_pause_between_moras_inside_one_word(self):
        vocab = {char: i for i, char in enumerate("|abcdefghijklmnopqrstuvwxyz'")}
        target, owners, allowed = make_grouped_targets(["mada"], [["ma", "da"]], vocab)
        np.testing.assert_array_equal(allowed, [True, False, True, False, True])
        np.testing.assert_array_equal(owners, [0, 0, 0, 0])
        self.assertEqual(target.tolist(), [vocab[c] for c in "mada"])

    def test_pronunciation_groups_preserve_separators_and_repeated_lines(self):
        vocab = {char: i for i, char in enumerate("|abcdefghijklmnopqrstuvwxyz'")}
        target, owners, allowed = make_grouped_targets(["ma da", "ma da"], [["ma", "da"], ["ma", "da"]], vocab)
        self.assertEqual(len(allowed), len(target) + 1)
        self.assertTrue(all(allowed[i] and allowed[i + 1] for i in np.flatnonzero(target == vocab["|"])))
        self.assertEqual(set(owners), {-1, 0, 1})

    def test_mismatched_pronunciation_groups_fail_instead_of_guessing(self):
        vocab = {char: i for i, char in enumerate("|abcdefghijklmnopqrstuvwxyz'")}
        with self.assertRaisesRegex(ValueError, "不一致"):
            make_grouped_targets(["mada"], [["ma", "ta"]], vocab)

    def test_internal_gap_prior_is_not_a_silence_duration_cap(self):
        # Genuine silence between two strongly supported syllables survives.
        values = np.log(np.array([[.98, .01, .01]] + [[.01, .01, .98]] * 20
                                 + [[.01, .98, .01]], dtype=np.float32))
        path = decode_frames(values, np.array([0, 1]), 2,
                             gap_costs=np.array([0, .02, 0]))
        np.testing.assert_array_equal(path, [0] + [-1] * 20 + [1])

    def test_internal_gap_cost_does_not_penalize_inter_word_silence(self):
        target = np.array([0, 2, 1])
        costs = (~word_gap_mask(target, 2)).astype(np.float32) * .02
        np.testing.assert_array_equal(costs, np.zeros(4))

    def test_invalid_internal_gap_cost_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "代价"):
            decode_frames(np.zeros((3, 3)), np.array([0, 1]), 2, gap_costs=np.array([0, -1, 0]))

    def test_adaptive_wait_scale_comes_from_audio_and_tracks_time_stretch(self):
        target = np.array([0, 1, 2, 3])
        gaps = np.array([True, False, True, False, True])
        path = np.array([0, 0, 1, 1, -1, -1, 2, 2, 3, 3])
        costs = acoustic_gap_costs(path, target, gaps, 9)
        stretched = acoustic_gap_costs(np.repeat(path, 2), target, gaps, 9)
        np.testing.assert_allclose(stretched, costs / 2)
        self.assertEqual(costs[0], 0)
        self.assertEqual(costs[-1], 0)

    def test_japanese_closure_stays_with_following_mora(self):
        from types import SimpleNamespace as NS
        from lyric_alignment.backend import _roman_word, roman_units
        word = NS(text="がっこう", readings=[NS(paths=[[NS(script=s) for s in ["ga", "cl", "ko", "u"]]])])
        self.assertEqual(roman_units(word), ["ga", "kko", "u"])
        self.assertEqual(_roman_word(word), "gakkou")

    def test_english_normalization_keeps_existing_spaces_and_apostrophes(self):
        from types import SimpleNamespace as NS
        from lyric_alignment.backend import _roman_word, roman_units
        word = NS(text="Don't stop")
        self.assertEqual(_roman_word(word), "don't stop")
        self.assertEqual(roman_units(word), ["don't", "stop"])

    def test_reading_annotations_are_not_sung_twice(self):
        from lyric_alignment.backend import spoken_text
        self.assertEqual(spoken_text("声明（こえ）を、点火 Ctrl＋Z"), "こえを 点火 Ctrl Z")

    def test_japanese_quotes_are_not_treated_as_unclaimed_words(self):
        from lyric_alignment.backend import spoken_text
        self.assertEqual(spoken_text("「異端」 don't"), " 異端  don't")


class RefinementContractTests(unittest.TestCase):
    def test_refiner_bounds_exclude_other_lines_but_keep_instrumental_gaps(self):
        lines = [LineSpan(0, 2, 5, "coarse"), LineSpan(1, 10, 15, "coarse"), LineSpan(2, 30, 35, "coarse")]
        self.assertEqual(line_bounds(lines, 1, 40), (5, 30))
        self.assertEqual(line_bounds(lines, 0, 40), (0, 10))
        self.assertEqual(line_bounds(lines, 2, 40), (15, 40))

    def test_context_windows_are_bounded_and_own_each_line_once(self):
        lines = [LineSpan(i, i * 5.0 + 3, i * 5.0 + 5, "coarse") for i in range(50)]
        windows = plan_windows(lines, 260)
        self.assertEqual([i for window in windows for i in window.keep_ids], list(range(50)))
        self.assertTrue(all(0 <= w.start < w.end <= 260 and w.end - w.start <= 50 for w in windows))
        self.assertTrue(all(set(w.keep_ids) <= set(w.context_ids) for w in windows))

    def test_long_line_is_not_split_using_character_proportions(self):
        lines = [LineSpan(0, 1, 70, "coarse"), LineSpan(1, 80, 84, "coarse")]
        windows = plan_windows(lines, 100)
        self.assertEqual([i for window in windows for i in window.keep_ids], [1])

    def test_phone_offsets_retain_original_audio_time(self):
        result = aggregate_phones(7, [(1, 1.5), (1.5, 2)], [.3, .7], 30, 10)
        self.assertEqual((result.line_id, result.start, result.end), (7, 31, 32))

    def test_missing_edge_phone_does_not_silently_shrink_caption(self):
        result = aggregate_phones(0, [(10, 10), (1, 2)], [.2], 0, 10)
        self.assertIsNone(result.start)
        self.assertIn("missing_edge_phoneme", result.flags)

    def test_unresolved_refinement_uses_flagged_acoustic_coarse_span(self):
        result = merge_refinement([LineSpan(0, 2, 4, "coarse")], [], 10)
        self.assertEqual((result[0].start, result[0].end, result[0].stage), (2, 4, "coarse_only"))
        self.assertIn("refinement_unresolved", result[0].flags)

    def test_overlaps_are_flagged_not_silently_clamped(self):
        result = merge_refinement([LineSpan(0, 1, 2, "coarse"), LineSpan(1, 3, 4, "coarse")],
                                  [LineSpan(0, 1, 3.5, "refined"), LineSpan(1, 3, 4, "refined")], 5)
        self.assertEqual(result[0].stage, "coarse_only")
        self.assertIn("refinement_outside_acoustic_support", result[0].flags)
        self.assertEqual(result[0].end, 2)

    def test_local_tail_cannot_expand_into_instrumental_gap(self):
        result = merge_refinement([LineSpan(0, 2, 4, "coarse")], [LineSpan(0, 2, 12, "refined")], 20)
        self.assertEqual((result[0].start, result[0].end), (2, 4))

    def test_refinement_within_acoustic_support_can_tighten_boundaries(self):
        result = merge_refinement([LineSpan(0, 2, 4, "coarse")], [LineSpan(0, 2.1, 3.9, "refined")], 20)
        self.assertEqual(result[0].stage, "refined")


class WorkflowTests(unittest.TestCase):
    def test_tifa_weights_load_once_after_pronunciation_preparation(self):
        from contextlib import ExitStack
        from lyric_alignment import backend as acoustic
        events = []
        model = Mock(sample_rate=48000)
        def weights(*args):
            events.append("weights")
            return model, Mock(), Mock()
        def readings(*args, **kwargs):
            events.append("readings")
            return [[Mock()]], ["ma"]
        with ExitStack() as stack:
            for name in ("check_runtime", "prepare_tifa", "prepare_mms", "_pipeline"):
                stack.enter_context(patch.object(acoustic, name))
            acoustic.prepare_tifa.return_value = (Path("source"), Path("model"))
            stack.enter_context(patch.object(acoustic, "_load_tifa_config", return_value=({}, Mock())))
            load = stack.enter_context(patch.object(acoustic, "_load_tifa", side_effect=weights))
            stack.enter_context(patch.object(acoustic, "prepare_pronunciations", side_effect=readings))
            stack.enter_context(patch.object(acoustic, "read_audio", side_effect=[np.zeros(160000), np.zeros(480000)]))
            stack.enter_context(patch.object(acoustic, "_refine_window", return_value=(
                [LineSpan(0, 1.1, 1.9, "refined")], [], .9)))
            stack.enter_context(patch("importlib.metadata.version", return_value="test"))
            result = acoustic.align_audio(Path("audio.wav"), ["原詞"], progress=lambda *args: None,
                coarse_cache={"romanized_lines": ["ma"], "duration": 10,
                              "coarse": [LineSpan(0, 1, 2, "coarse").record()]})
        self.assertEqual(events, ["readings", "weights"])
        self.assertEqual(load.call_count, 1)
        self.assertEqual(result["lines"][0]["start"], 1.1)

    def test_srt_rounding_carries_into_minutes_and_hours(self):
        self.assertEqual(sync._fmt_srt_time(59.9999), "00:01:00,000")
        self.assertEqual(sync._fmt_srt_time(3599.9999), "01:00:00,000")

    def _result(self):
        return {"profile": PROFILE, "duration": 10, "lines": [LineSpan(0, 1, 2, "refined").record()]}

    def test_chinese_translation_cannot_change_inference_identity(self):
        fingerprint = lambda p: {"path": str(p), "size": 1}
        a = cache_metadata(Path("video"), ["原詞"], "vocals", fingerprint)
        b = cache_metadata(Path("video"), ["別の原詞"], "vocals", fingerprint)
        self.assertNotEqual(a, b)
        # Chinese text is absent from the cache key; changing translation reuses audio work.
        result = self._result()
        items = chinese_items(result, [sync.LyricLine("原詞", "中文（保持标点）")], sync.AlignmentItem)
        self.assertEqual(items[0].text, "中文（保持标点）")

    def test_invalid_cached_results_are_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "alignment.json"
            result = self._result()
            path.write_text(json.dumps({"metadata": {"id": 1}, "acoustic_alignment": result}), encoding="utf-8")
            self.assertIsNotNone(load_cache(path, {"id": 1}, 1))
            self.assertIsNone(load_cache(path, {"id": 2}, 1))
            self.assertIsNone(load_cache(path, {"id": 1}, 2))
            result["lines"][0]["start"] = None
            self.assertFalse(valid_result(result, 1))

    def test_coarse_cache_reuses_only_same_video_source_lyrics_and_acoustic_model(self):
        from lyric_alignment import COARSE_PROFILE
        from lyric_alignment.resources import MMS_REVISION
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "alignment.json"
            metadata = {"profile": "old-refiner", "video": {"size": 1}, "source_sha256": "source-a"}
            result = {"coarse_profile": COARSE_PROFILE, "models": {"coarse_revision": MMS_REVISION}, "coarse": []}
            path.write_text(json.dumps({"metadata": metadata, "acoustic_alignment": result}), encoding="utf-8")
            expected = {**metadata, "profile": PROFILE}
            self.assertIsNotNone(load_coarse_cache(path, expected))
            self.assertIsNone(load_coarse_cache(path, {**expected, "source_sha256": "source-b"}))
            self.assertIsNone(load_coarse_cache(path, {**expected, "video": {"size": 2}}))

    def test_old_numeric_evidence_replays_exactly_current_assembly_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "alignment.json"
            old_profile = "ja-mms-frame-tifa-local24-bounded-fp32-v4"
            result = {"profile": old_profile, "duration": 20,
                      "coarse": [LineSpan(0, 2, 4, "coarse").record()],
                      "refinement": [LineSpan(0, 2, 12, "refined").record()]}
            path.write_text(json.dumps({"metadata": {"profile": old_profile, "video": "same"},
                                        "acoustic_alignment": result}), encoding="utf-8")
            replay = load_cache(path, {"profile": PROFILE, "video": "same"}, 1)
            self.assertEqual(replay["lines"][0]["end"], 4)
            self.assertEqual(replay["assembly_replay_from"], old_profile)
            self.assertIsNone(load_cache(path, {"profile": PROFILE, "video": "different"}, 1))

    def test_bilingual_count_mismatch_fails_before_model_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lyrics.txt"
            path.write_text("中文一\n中文二\n\n\n日本語一\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "行数不一致"):
                sync.load_lyrics(path, require_pairs=True)

    def test_chinese_only_is_not_accepted_by_acoustic_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lyrics.txt"
            path.write_text("只有中文\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "双语歌词"):
                sync.load_lyrics(path, require_pairs=True)

    def test_new_engine_dispatch_does_not_transcribe_or_fuzzy_match(self):
        with tempfile.TemporaryDirectory() as temp, patch("lyric_alignment.workflow.process_known_lyrics") as process:
            with patch.object(sync, "transcribe_audio", side_effect=AssertionError("ASR called")):
                sync.process_single(Path(temp) / "v.mp4", Path(temp) / "l.txt", model_name=LYRIC_MODEL, no_burn=True)
            self.assertEqual(process.call_count, 1)

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "package.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("../outside", "unsafe")
            with self.assertRaises(ValueError):
                safe_extract(archive, Path(temp) / "unpacked")

    def test_full_workflow_writes_chinese_and_only_one_json_then_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, lyrics = root / "video.mp4", root / "lyrics.txt"
            video.write_bytes(b"test-video")
            lyrics.write_text("原詞|中文（第一版）", encoding="utf-8")
            output = root / "out.mp4"
            result = self._result()
            with patch("lyric_alignment.backend.check_runtime"), patch("lyric_alignment.backend.align_audio", return_value=result) as align:
                with patch("lyric_alignment.workflow.extract_timeline_audio") as extract:
                    sync.process_single(video, lyrics, output, model_name=LYRIC_MODEL, audio_source="original", no_burn=True)
                    lyrics.write_text("原詞|中文（修改翻译）", encoding="utf-8")
                    sync.process_single(video, lyrics, output, model_name=LYRIC_MODEL, audio_source="original", no_burn=True)
            self.assertEqual(align.call_count, 1)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual([p.name for p in root.glob("*.json")], ["out.alignment.json"])
            self.assertIn("中文（修改翻译）", output.with_suffix(".srt").read_text(encoding="utf-8"))
            report = json.loads(output.with_suffix(".alignment.json").read_text(encoding="utf-8"))
            self.assertFalse(report["summary"]["accuracy_verified"])

    def test_extraction_keeps_container_offset_and_inserts_leading_silence(self):
        from lyric_alignment.workflow import extract_timeline_audio
        helpers = Mock()
        helpers._get_ffmpeg_path.return_value = "ffmpeg"
        extract_timeline_audio(Path("in.mp4"), Path("out.wav"), helpers)
        command = helpers._run.call_args.args[0]
        self.assertIn("-copyts", command)
        self.assertIn("-start_at_zero", command)
        self.assertIn("aresample=async=1:first_pts=0", command)


class GuiIntegrationTests(unittest.TestCase):
    def test_acoustic_engine_uses_its_environment_without_whisper_or_qwen(self):
        modules = gui._required_modules({"model": LYRIC_MODEL, "audio_source": "original"})
        self.assertIn("g2pflow", modules)
        self.assertNotIn("faster_whisper", modules)
        self.assertNotIn("qwen3_asr", modules)

    def test_acoustic_command_cannot_inherit_ocr_or_vad_checkboxes(self):
        command = gui._build_processing_command("python", {
            "mode": "single", "video": "v", "lyrics": "l", "audio_source": "original",
            "model": LYRIC_MODEL, "no_vad": True, "ocr_assist": True, "refresh": True,
        })
        self.assertNotIn("--ocr-assist", command)
        self.assertNotIn("--no-vad", command)
        self.assertIn("--refresh-transcript", command)

    def test_lyric_progress_changes_progress_bar(self):
        event = gui.parse_runtime_line("歌词对齐进度: 逐句精对齐 3/6")
        self.assertEqual(event["kind"], "lyric_progress")
        self.assertGreater(event["progress"], 60)


if __name__ == "__main__":
    unittest.main()
