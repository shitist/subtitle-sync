"""Unconvertible words and lines retain the rest of the lyric workflow."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from g2pflow.converters.base import G2PConversionError
from lyric_alignment import PROFILE, backend, workflow
from lyric_alignment.core import LineSpan, make_targets
import subtitle_sync as sync
import subtitle_sync_gui as gui


class PronunciationRecoveryTests(unittest.TestCase):
    def test_one_unknown_word_retains_other_words_and_original_text(self):
        pipeline = Mock()
        pipeline.convert.side_effect = [G2PConversionError(["b4d "]),
                                        [SimpleNamespace(text="Hello"), SimpleNamespace(text="world")]]
        lines = ["Hello B4d world"]
        issues = []
        words, roman = backend.prepare_pronunciations(pipeline, lines, issues=issues)
        self.assertEqual(roman, ["hello world"])
        self.assertEqual([word.text for word in words[0]], ["Hello", "world"])
        self.assertEqual(lines, ["Hello B4d world"])
        self.assertEqual(issues[0]["unconverted_tokens"], ["B4d"])
        self.assertEqual(issues[0]["status"], "partial")
        self.assertFalse(issues[0]["missing_start"])
        self.assertFalse(issues[0]["missing_end"])
        rows = [LineSpan(0, 1, 5, "refined")]
        backend._mark_pronunciation_issues(rows, issues)
        self.assertEqual(rows[0].flags, [])
        self.assertEqual((rows[0].start, rows[0].end), (1, 5))
        self.assertEqual(pipeline.convert.call_count, 2)

    def test_repeated_conversion_failures_remove_only_reported_spans(self):
        pipeline = Mock()
        pipeline.convert.side_effect = [G2PConversionError(["b4d"]), G2PConversionError(["w0rd"]),
                                        [SimpleNamespace(text="Say")]]
        issues = []
        _, roman = backend.prepare_pronunciations(pipeline, ["B4d W0rd Say"], issues=issues)
        self.assertEqual(roman, ["say"])
        self.assertEqual(len(issues[0]["unconverted_tokens"]), 2)

    def test_unconvertible_line_does_not_abort_following_lines(self):
        pipeline = Mock()
        pipeline.convert.side_effect = [G2PConversionError(["b4d"]), [SimpleNamespace(text="Next")]]
        issues = []
        words, roman = backend.prepare_pronunciations(pipeline, ["B4d", "Next"], issues=issues)
        self.assertEqual(words[0], [])
        self.assertEqual(roman, ["", "next"])
        self.assertEqual(issues[0]["line_id"], 0)
        self.assertEqual(issues[0]["status"], "unresolved")

    def test_valid_conversion_is_unchanged(self):
        pipeline = Mock()
        word = SimpleNamespace(text="Don't stop")
        pipeline.convert.return_value = [word]
        issues = []
        words, roman = backend.prepare_pronunciations(pipeline, ["Don't stop"], issues=issues)
        self.assertEqual(words, [[word]])
        self.assertEqual(roman, ["don't stop"])
        self.assertEqual(issues, [])
        self.assertEqual(pipeline.convert.call_count, 1)

    def test_missing_edge_words_flag_only_the_affected_boundaries(self):
        for source, start, end in (("B4d Hello", True, False), ("Hello B4d", False, True)):
            with self.subTest(source=source):
                pipeline = Mock()
                pipeline.convert.side_effect = [G2PConversionError(["b4d"]), [SimpleNamespace(text="Hello")]]
                issues = []
                backend.prepare_pronunciations(pipeline, [source], issues=issues)
                self.assertEqual(issues[0]["missing_start"], start)
                self.assertEqual(issues[0]["missing_end"], end)
                rows = [LineSpan(0, 2, 3, "refined")]
                backend._mark_pronunciation_issues(rows, issues)
                expected = "g2p_missing_start" if start else "g2p_missing_end"
                self.assertEqual(rows[0].flags, [expected])

    def test_runtime_errors_are_not_classified_as_unknown_words(self):
        pipeline = Mock()
        pipeline.convert.side_effect = RuntimeError("model runtime failed")
        with self.assertRaisesRegex(ValueError, "model runtime failed"):
            backend.prepare_pronunciations(pipeline, ["Hello"])

    def test_missing_reading_keeps_original_line_ids(self):
        vocab = {char: index for index, char in enumerate("|abcdefghijklmnopqrstuvwxyz'")}
        _, owners = make_targets(["a", "", "b"], vocab, allow_empty=True)
        self.assertEqual(owners.tolist(), [0, -1, 2])
        with self.assertRaises(ValueError):
            make_targets(["a", "", "b"], vocab)

    def test_recovery_messages_are_visible_in_gui(self):
        for line in ("第 5 行：模型不支持 B4d 中的数字。", "字幕生成：2/3 行"):
            self.assertEqual(gui.parse_runtime_line(line)["message"], line)


class PartialWorkflowTests(unittest.TestCase):
    def test_unresolved_line_is_reported_while_other_chinese_cues_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, lyrics, output = root / "song.mp4", root / "lyrics.txt", root / "out.mp4"
            video.touch()
            lyrics.write_text("あ|甲\nい|乙\nう|丙", encoding="utf-8")
            rows = [LineSpan(0, 1, 2, "refined"),
                    LineSpan(1, None, None, "unresolved", flags=["g2p_unresolved"]),
                    LineSpan(2, 5, 6, "coarse_only")]
            result = {"profile": PROFILE, "duration": 10, "lines": [row.record() for row in rows]}
            with patch.object(backend, "check_runtime"), patch.object(workflow, "extract_timeline_audio"), \
                 patch.object(backend, "align_audio", return_value=result):
                workflow.process_known_lyrics(video, lyrics, output, audio_source="original", no_burn=True,
                                              refresh=True, helpers=sync)
            srt = output.with_suffix(".srt").read_text(encoding="utf-8")
            self.assertIn("甲", srt)
            self.assertIn("丙", srt)
            self.assertNotIn("乙", srt)
            report = json.loads(output.with_suffix(".alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["unresolved_lines"], [2])
            self.assertEqual([row["text"] for row in report["subtitles"]], ["甲", "乙", "丙"])
            self.assertTrue(workflow.valid_result(result, 3))

    def test_all_unknown_lines_skip_acoustic_model_loading(self):
        import numpy as np
        pipeline = Mock()
        pipeline.convert.side_effect = G2PConversionError(["b4d"])
        with patch.object(backend, "check_runtime"), \
             patch.object(backend, "prepare_tifa", return_value=(Path("source"), Path("model"))), \
             patch.object(backend, "_load_tifa_config", return_value=({}, Mock())), \
             patch.object(backend, "_pipeline", return_value=pipeline), \
             patch.object(backend, "read_audio", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(backend, "prepare_mms", side_effect=AssertionError("unexpected MMS loading")), \
             patch.object(backend, "_load_tifa", side_effect=AssertionError("unexpected TIFA loading")):
            result = backend.align_audio(Path("audio.wav"), ["B4d"])
        self.assertEqual(result["lines"][0]["stage"], "unresolved")
        self.assertTrue(workflow.valid_result(result, 1))


if __name__ == "__main__":
    unittest.main()
