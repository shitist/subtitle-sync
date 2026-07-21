import json
import tempfile
import unittest
from pathlib import Path

from subtitle_sync import (
    AlignmentItem,
    LyricLine,
    SimpleSegment,
    WordTiming,
    _normalize_japanese,
    _transcript_metadata,
    align_lyrics_to_segments,
    load_lyrics,
    load_transcript_cache,
    process_single,
    save_alignment_report,
    save_transcript_cache,
)


def segment_with_words(words):
    return SimpleSegment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(word.text for word in words),
        words=words,
    )


class LyricsParsingTests(unittest.TestCase):
    def _load(self, content):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lyrics.txt"
            path.write_text(content, encoding="utf-8")
            return load_lyrics(path)

    def test_double_blank_separates_blocks_when_japanese_starts_with_english(self):
        prompt, lines = self._load(
            "中文字幕一\n\n中文字幕二\n中文字幕三\n\n\n"
            "English source one\nEnglish source two\n日本語三\n"
        )

        self.assertEqual(
            [line.zh for line in lines],
            ["中文字幕一", "中文字幕二", "中文字幕三"],
        )
        self.assertEqual(
            [line.ja for line in lines],
            ["English source one", "English source two", "日本語三"],
        )
        self.assertIn("日本語三", prompt)

    def test_single_blank_is_only_paragraph_spacing(self):
        prompt, lines = self._load("第一行\n\n第二行\n")

        self.assertEqual(prompt, "")
        self.assertEqual([line.zh for line in lines], ["第一行", "第二行"])

    def test_legacy_block_without_double_blank_still_uses_kana_boundary(self):
        prompt, lines = self._load("中文一\n中文二\nにほんご一\nにほんご二\n")

        self.assertEqual([line.zh for line in lines], ["中文一", "中文二"])
        self.assertEqual([line.ja for line in lines], ["にほんご一", "にほんご二"])
        self.assertEqual(prompt, "にほんご一。にほんご二")


class TextAlignmentTests(unittest.TestCase):
    def test_normalization_unifies_width_katakana_and_punctuation(self):
        self.assertEqual(_normalize_japanese(" セカイ、カタカナ！ "), "せかいかたかな")

    def test_recognized_text_drives_alignment_and_skips_hallucinations(self):
        words = [
            WordTiming(0.0, 0.8, "雑音"),
            WordTiming(2.0, 4.0, "世界の終わりで"),
            WordTiming(4.1, 4.4, "ラララ"),
            WordTiming(5.0, 7.0, "生まれた光"),
            WordTiming(8.0, 10.0, "君を探して"),
        ]
        lyrics = [
            LyricLine("世界の終わりで", "在世界终结之处"),
            LyricLine("生まれた光", "诞生的光芒"),
            LyricLine("君を探して", "寻找着你"),
        ]

        aligned = align_lyrics_to_segments([segment_with_words(words)], lyrics)

        self.assertEqual([item.status for item in aligned], ["matched"] * 3)
        self.assertEqual([item.start for item in aligned], [2.0, 5.0, 8.0])
        self.assertEqual([item.recognized_text for item in aligned], [
            "世界の終わりで", "生まれた光", "君を探して",
        ])

    def test_missing_line_is_interpolated_only_between_neighboring_anchors(self):
        words = [
            WordTiming(1.0, 2.0, "一番目"),
            WordTiming(5.0, 6.0, "三番目"),
        ]
        lyrics = [
            LyricLine("一番目", "第一行"),
            LyricLine("二番目", "第二行"),
            LyricLine("三番目", "第三行"),
        ]

        aligned = align_lyrics_to_segments([segment_with_words(words)], lyrics)

        self.assertEqual(aligned[0].status, "matched")
        self.assertEqual(aligned[1].status, "interpolated")
        self.assertEqual(aligned[2].status, "matched")
        self.assertEqual((aligned[1].start, aligned[1].end), (2.0, 5.0))
        self.assertEqual(aligned[2].start, 5.0)

    def test_repeated_chorus_keeps_monotonic_order(self):
        words = [
            WordTiming(1.0, 2.0, "君と歩く"),
            WordTiming(8.0, 9.0, "君と歩く"),
        ]
        lyrics = [
            LyricLine("君と歩く", "与你同行"),
            LyricLine("君と歩く", "与你同行"),
        ]

        aligned = align_lyrics_to_segments([segment_with_words(words)], lyrics)

        self.assertEqual([item.start for item in aligned], [1.0, 8.0])

    def test_missing_prefix_is_placed_before_first_anchor_without_overlap(self):
        words = [WordTiming(5.0, 6.0, "三番目")]
        lyrics = [
            LyricLine("一番目", "第一行"),
            LyricLine("二番目", "第二行"),
            LyricLine("三番目", "第三行"),
        ]

        aligned = align_lyrics_to_segments([segment_with_words(words)], lyrics)

        self.assertEqual(aligned[2].status, "matched")
        self.assertLess(aligned[0].start, aligned[2].start)
        self.assertLessEqual(aligned[1].end, aligned[2].start)

    def test_chinese_only_input_is_explicit_timing_fallback(self):
        segments = [
            SimpleSegment(1.0, 2.0, "音声一", []),
            SimpleSegment(3.0, 4.0, "音声二", []),
        ]
        lyrics = [LyricLine("", "第一行"), LyricLine("", "第二行")]

        aligned = align_lyrics_to_segments(segments, lyrics)

        self.assertEqual([item.status for item in aligned], ["timing_fallback"] * 2)
        self.assertEqual([item.text for item in aligned], ["第一行", "第二行"])

    def test_chinese_only_uses_ordered_latin_anchors_before_timing_fallback(self):
        words = [
            WordTiming(1.0, 1.4, "Start"),
            WordTiming(1.4, 1.8, "now"),
            WordTiming(5.0, 5.8, "中间日语"),
            WordTiming(10.0, 10.5, "Final"),
            WordTiming(10.5, 11.0, "light"),
        ]
        lyrics = [
            LyricLine("", "Start now"),
            LyricLine("", "中间的中文字幕"),
            LyricLine("", "Final light"),
        ]

        aligned = align_lyrics_to_segments([segment_with_words(words)], lyrics)

        self.assertEqual(aligned[0].status, "latin_anchor")
        self.assertEqual(aligned[0].start, 1.0)
        self.assertEqual(aligned[1].status, "interpolated")
        self.assertEqual(aligned[2].status, "latin_anchor")
        self.assertEqual(aligned[2].start, 10.0)

    def test_low_whisper_probability_is_flagged_for_review(self):
        words = [WordTiming(1.0, 2.0, "正しい歌詞", 0.2)]
        lyrics = [LyricLine("正しい歌詞", "正确歌词")]

        aligned = align_lyrics_to_segments([segment_with_words(words)], lyrics)

        self.assertEqual(aligned[0].status, "low_confidence")
        self.assertEqual(aligned[0].word_probability, 0.2)


class ArtifactTests(unittest.TestCase):
    def test_transcript_metadata_does_not_depend_on_lyrics_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "sample.mp4"
            lyrics_a = root / "a.txt"
            lyrics_b = root / "b.txt"
            video.write_bytes(b"same-video")
            lyrics_a.write_text("纯中文", encoding="utf-8")
            lyrics_b.write_text("日本語\n\n\n日文", encoding="utf-8")

            metadata_a = _transcript_metadata(video, True, "vocals")
            metadata_b = _transcript_metadata(video, True, "vocals")

        self.assertEqual(metadata_a, metadata_b)
        self.assertNotIn("lyrics", metadata_a)

    def test_transcript_metadata_changes_with_selected_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "sample.mp4"
            video.write_bytes(b"same-video")

            large = _transcript_metadata(video, True, "vocals", "large-v3")
            turbo = _transcript_metadata(
                video,
                True,
                "vocals",
                "large-v3-turbo",
            )

        self.assertEqual(large["model"], "large-v3")
        self.assertEqual(turbo["model"], "large-v3-turbo")
        self.assertNotEqual(large, turbo)

    def test_transcript_cache_round_trip_and_metadata_guard(self):
        segments = [SimpleSegment(
            1.0,
            2.0,
            "歌詞",
            [WordTiming(1.1, 1.9, "歌詞", 0.9)],
        )]
        metadata = {"schema_version": 2, "key": "value"}
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "sample.whisper.json"
            save_transcript_cache(cache, segments, metadata)

            loaded = load_transcript_cache(cache, metadata)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded[0].words[0].text, "歌詞")
            self.assertIsNone(load_transcript_cache(cache, {"key": "other"}))

    def test_alignment_report_marks_review_lines(self):
        aligned = [
            AlignmentItem(1.0, 2.0, "第一行", "一番目", "一番目", 100.0, "matched"),
            AlignmentItem(2.0, 3.0, "第二行", "二番目", "", None, "interpolated"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "sample.alignment.json"
            summary = save_alignment_report(aligned, report, 55.0)
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(summary["matched_coverage"], 0.5)
        self.assertEqual(payload["summary"]["needs_review"], [2])

    def test_cached_no_burn_flow_generates_reviewable_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "sample.mp4"
            lyrics = root / "sample.txt"
            output = root / "sample_subtitled.mp4"
            cache = output.with_suffix(".whisper.json")
            video.write_bytes(b"test-video-fingerprint")
            lyrics.write_text("一番目|第一行\n二番目|第二行\n", encoding="utf-8")
            segments = [segment_with_words([
                WordTiming(1.0, 2.0, "一番目", 0.95),
                WordTiming(3.0, 4.0, "二番目", 0.95),
            ])]
            metadata = _transcript_metadata(video, True, "vocals")
            save_transcript_cache(cache, segments, metadata)

            process_single(video, lyrics, output, no_burn=True)

            self.assertFalse(output.exists())
            self.assertTrue(output.with_suffix(".srt").exists())
            self.assertTrue(output.with_suffix(".alignment.json").exists())
            report = json.loads(
                output.with_suffix(".alignment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["summary"]["matched_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
