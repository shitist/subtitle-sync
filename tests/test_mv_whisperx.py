"""MV WhisperX line ownership, cache and workflow checks."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mv_whisperx as mv
import subtitle_sync as sync


class MvWhisperXTests(unittest.TestCase):
    def test_anchor_search_does_not_fill_missing_lines_in_new_route(self):
        words = [sync.WordTiming(1, 2, "あ"), sync.WordTiming(5, 6, "う")]
        lines = [sync.LyricLine("あ", "甲"), sync.LyricLine("い", "乙"), sync.LyricLine("う", "丙")]
        anchors = sync._align_with_text(words, lines, 75, strict_anchors=True, fill_missing=False)
        self.assertEqual(anchors[1].status, "unresolved")
        self.assertIsNone(anchors[1].start)
        self.assertIsNone(anchors[1].end)

    def test_missing_run_uses_observed_neighbor_anchors(self):
        anchors = [sync.AlignmentItem(10, 12, "a", score=90, status="matched"),
                   sync.AlignmentItem(999, 1000, "b", status="interpolated"),
                   sync.AlignmentItem(20, 22, "c", score=90, status="matched")]
        windows = mv.plan_windows(anchors, 30)
        missing = next(w for w in windows if w["keep"] == [1])
        self.assertEqual(missing["context"], [0, 1, 2])
        self.assertEqual((missing["start"], missing["end"]), (9.5, 22.5))
        self.assertEqual(sorted(i for w in windows for i in w["keep"]), [0, 1, 2])

    def test_unmatched_song_uses_audio_bounds_not_character_lengths(self):
        anchors = [sync.AlignmentItem(999, 1000, "a", status="interpolated"),
                   sync.AlignmentItem(1001, 1002, "long text", status="interpolated")]
        windows = mv.plan_windows(anchors, 18)
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0]["start"], windows[0]["end"]), (0, 18))

    def test_character_times_map_to_the_original_line_ids(self):
        result = {"segments": [{"chars": [
            {"char": "あ", "start": 0.1, "end": 0.3, "score": .8},
            {"char": " ", "start": 0.3, "end": 0.5},
            {"char": "い", "start": 0.8, "end": 1.2, "score": .9}]}]}
        rows = mv.collect_lines(result, ["あ", "い"], [3, 4], 10)
        self.assertEqual((rows[3]["start"], rows[3]["end"]), (10.1, 10.3))
        self.assertEqual((rows[4]["start"], rows[4]["end"]), (10.8, 11.2))

    def test_missing_character_is_unresolved_not_interpolated(self):
        result = {"segments": [{"chars": [{"char": "あ", "start": .1, "end": .2}, {"char": "い"}]}]}
        row = mv.collect_lines(result, ["あい"], [0], 5)[0]
        self.assertIsNone(row["start"])
        self.assertEqual(row["coverage"], .5)

    def test_returned_text_mismatch_is_not_mapped_to_wrong_lines(self):
        with self.assertRaisesRegex(ValueError, "字符"):
            mv.collect_lines({"segments": [{"chars": [{"char": "い"}]}]}, ["あ"], [0], 0)

    def test_translation_change_reuses_alignment_but_original_change_does_not(self):
        old = [sync.LyricLine("あ", "旧翻译")]
        new = [sync.LyricLine("あ", "新翻译")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out.alignment.json"
            item = sync.AlignmentItem(1, 2, "旧翻译", source_text="あ", status="whisperx")
            expected = mv.metadata({"video": "test"}, old)
            path.write_text(json.dumps({"whisperx": {"metadata": expected, "items": [asdict(item)]}}), encoding="utf-8")
            result = mv.load_cache(path, mv.metadata({"video": "test"}, new), new, sync)
            self.assertEqual(result[0][0].text, "新翻译")
            changed = [sync.LyricLine("い", "新翻译")]
            self.assertIsNone(mv.load_cache(path, mv.metadata({"video": "test"}, changed), changed, sync))

    def test_cached_transcription_still_passes_lyrics_to_whisperx(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, lyrics, output = root / "v.mp4", root / "l.txt", root / "out.mp4"
            video.touch()
            lyrics.write_text("あ|甲\nい|乙", encoding="utf-8")
            segments = [sync.SimpleSegment(1, 3, "あい", [sync.WordTiming(1, 2, "あ"), sync.WordTiming(2, 3, "い")])]
            sync.save_transcript_cache(output.with_suffix(".whisper.json"), segments,
                sync._transcript_metadata(video, True, "original", "large-v3"))
            items = [sync.AlignmentItem(1.1, 1.8, "甲", source_text="あ", status="whisperx"),
                     sync.AlignmentItem(2.2, 2.8, "乙", source_text="い", status="whisperx")]
            backend = {"aligned_lines": 2, "asr_anchor_lines": 0, "unresolved_lines": []}
            with patch.object(sync, "extract_audio"), patch.object(sync, "_run"), \
                 patch.object(sync, "transcribe_audio", side_effect=AssertionError("unexpected ASR")), \
                 patch.object(mv, "align_lyrics", return_value=(items, backend)) as align:
                sync.process_single(video, lyrics, output, no_burn=True, audio_source="original")
            self.assertEqual([line.ja for line in align.call_args.args[2]], ["あ", "い"])
            self.assertIn("00:00:01,100 --> 00:00:01,800", output.with_suffix(".srt").read_text(encoding="utf-8"))
            report = json.loads(output.with_suffix(".alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["mode"], "whisperx")
            self.assertEqual(report["summary"]["estimated_lines"], 0)

    def test_unresolved_lyrics_write_report_and_preserve_existing_srt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, lyrics, output = root / "v.mp4", root / "l.txt", root / "out.mp4"
            video.touch()
            lyrics.write_text("あ|甲", encoding="utf-8")
            srt = output.with_suffix(".srt")
            srt.write_text("existing subtitle", encoding="utf-8")
            segments = [sync.SimpleSegment(1, 2, "あ", [sync.WordTiming(1, 2, "あ")])]
            sync.save_transcript_cache(output.with_suffix(".whisper.json"), segments,
                sync._transcript_metadata(video, True, "original", "large-v3"))
            backend = {"aligned_lines": 0, "asr_anchor_lines": 0, "unresolved_lines": [1]}
            unresolved = sync.AlignmentItem(None, None, "甲", source_text="あ", status="unresolved")
            with patch.object(sync, "extract_audio"), patch.object(sync, "_run"), \
                 patch.object(mv, "align_lyrics", return_value=([unresolved], backend)):
                with self.assertRaisesRegex(RuntimeError, "第 1 行"):
                    sync.process_single(video, lyrics, output, no_burn=True, audio_source="original")
            self.assertEqual(srt.read_text(encoding="utf-8"), "existing subtitle")
            self.assertEqual(json.loads(output.with_suffix(".alignment.json").read_text(encoding="utf-8"))[
                "summary"]["unresolved_lines"], [1])


if __name__ == "__main__":
    unittest.main()
