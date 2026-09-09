import json
import tempfile
import unittest
from pathlib import Path

from subtitle_sync import (
    AlignmentItem,
    LyricLine,
    OcrFrame,
    OcrLineAnchor,
    SimpleSegment,
    WordTiming,
    _ocr_metadata,
    _normalize_japanese,
    _classify_ocr_anchor_conflicts,
    _transcript_metadata,
    apply_ocr_timing_assist,
    align_lyrics_to_segments,
    build_ocr_line_anchors,
    load_lyrics,
    load_ocr_cache,
    load_transcript_cache,
    process_single,
    save_alignment_report,
    save_ocr_cache,
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


class OcrAssistanceTests(unittest.TestCase):
    def test_global_ocr_order_detects_backward_anchors(self):
        anchors = [
            OcrLineAnchor(0, 10.0, 0.0, "第一句", "第一句", 99.0, 0.5),
            OcrLineAnchor(1, 20.0, 0.0, "第二句", "第二句", 99.0, 0.5),
            OcrLineAnchor(2, 15.0, 0.0, "第三句", "第三句", 85.0, 0.5),
        ]

        hard, repeated = _classify_ocr_anchor_conflicts(anchors, 0.5)

        self.assertEqual(hard, {1, 2})
        self.assertEqual(repeated, set())

    def test_global_ocr_order_allows_distinct_lines_on_the_same_frame(self):
        anchors = [
            OcrLineAnchor(0, 10.0, 0.0, "第一句", "第一句", 99.0, 0.5),
            OcrLineAnchor(1, 10.0, 0.0, "第二句", "第二句", 99.0, 0.5),
        ]

        hard, repeated = _classify_ocr_anchor_conflicts(anchors, 0.5)

        self.assertEqual(hard, set())
        self.assertEqual(repeated, set())

    def test_global_ocr_order_does_not_reuse_one_frame_for_repeated_lyrics(self):
        anchors = [
            OcrLineAnchor(0, 10.0, 0.0, "同じ歌詞", "同じ歌詞", 99.0, 0.5),
            OcrLineAnchor(1, 10.0, 0.0, "同じ歌詞", "同じ歌詞", 85.0, 0.5),
        ]

        hard, repeated = _classify_ocr_anchor_conflicts(anchors, 0.5)

        self.assertEqual(hard, set())
        self.assertEqual(repeated, {0, 1})

    def test_backward_ocr_sequence_falls_back_without_moving_timing(self):
        baseline = [
            AlignmentItem(0.0, 4.0, "第一行"),
            AlignmentItem(4.0, 8.0, "第二行"),
            AlignmentItem(8.0, 12.0, "第三行"),
        ]
        anchors = [
            OcrLineAnchor(0, 10.0, 10.0, "第一句", "第一句", 99.0, 0.5),
            OcrLineAnchor(1, 20.0, 16.0, "第二句", "第二句", 99.0, 0.5),
            OcrLineAnchor(2, 15.0, 7.0, "第三句", "第三句", 99.0, 0.5),
        ]

        corrected = apply_ocr_timing_assist(baseline, anchors)

        self.assertEqual(
            [(item.start, item.end) for item in corrected],
            [(item.start, item.end) for item in baseline],
        )
        self.assertEqual(corrected[1].review_reason, "ocr_sequence_conflict")
        self.assertEqual(corrected[2].review_reason, "ocr_sequence_conflict")

    def test_vertical_ocr_characters_form_a_lyric_anchor(self):
        frames = [OcrFrame(11.0, ("待", "っ", "た"), (0.99, 0.98, 0.99))]
        lyrics = [LyricLine("待った！ナシだ", "等等！这可不算")]
        baseline = [AlignmentItem(10.0, 12.0, lyrics[0].zh, lyrics[0].ja)]

        anchors = build_ocr_line_anchors(frames, lyrics, baseline)

        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].recognized_text, "待った")
        self.assertGreaterEqual(anchors[0].score, 82.0)

    def test_ocr_correction_preserves_gaps_and_minimum_duration(self):
        baseline = [
            AlignmentItem(0.0, 2.0, "第一行"),
            AlignmentItem(2.2, 2.7, "短句"),
            AlignmentItem(2.9, 5.0, "第三行"),
        ]
        anchors = [
            OcrLineAnchor(i, item.start + 1.0, 1.0, item.text, item.text, 99.0, 0.5)
            for i, item in enumerate(baseline)
        ]

        corrected = apply_ocr_timing_assist(baseline, anchors)

        self.assertAlmostEqual(corrected[0].start, 1.0)
        self.assertAlmostEqual(corrected[-1].end, 6.0)
        self.assertGreaterEqual(corrected[1].end - corrected[1].start, 0.5)
        self.assertLessEqual(corrected[0].end, corrected[1].start)
        self.assertLessEqual(corrected[1].end, corrected[2].start)
        self.assertEqual(corrected[1].ocr_score, 99.0)

    def test_too_few_ocr_anchors_leave_baseline_unchanged(self):
        baseline = [AlignmentItem(i, i + 0.8, str(i)) for i in range(10)]
        anchors = [OcrLineAnchor(0, 0.5, 0.5, "一", "一", 99.0, 0.5)]

        self.assertIs(apply_ocr_timing_assist(baseline, anchors), baseline)

    def test_long_whisper_span_is_tightened_from_late_visual_anchor(self):
        baseline = [
            AlignmentItem(0.0, 20.0, "吞入间奏的第一行"),
            AlignmentItem(20.0, 25.0, "第二行"),
            AlignmentItem(25.0, 30.0, "第三行"),
        ]
        anchors = [
            OcrLineAnchor(
                0,
                18.5,
                0.0,
                "第一行歌词",
                "第一行歌词",
                99.0,
                0.5,
                target_coverage=1.0,
                visual_start=17.25,
                visual_end=19.75,
                sample_count=5,
            ),
            OcrLineAnchor(1, 22.5, 0.0, "第二行", "第二行", 99.0, 0.5),
            OcrLineAnchor(2, 27.5, 0.0, "第三行", "第三行", 99.0, 0.5),
        ]

        corrected = apply_ocr_timing_assist(baseline, anchors)

        self.assertAlmostEqual(corrected[0].start, 14.75)
        self.assertAlmostEqual(corrected[0].end, 19.75)
        self.assertEqual(corrected[0].review_reason, "ocr_duration_outlier")

    def test_sustained_visual_span_uses_its_own_start_boundary(self):
        baseline = [
            AlignmentItem(0.0, 20.0, "吞入片头的第一行"),
            AlignmentItem(20.0, 25.0, "第二行"),
            AlignmentItem(25.0, 30.0, "第三行"),
        ]
        anchors = [
            OcrLineAnchor(
                0,
                17.5,
                0.0,
                "第一行歌词",
                "第一行歌词",
                99.0,
                0.65,
                target_coverage=0.65,
                visual_start=14.75,
                visual_end=19.75,
                sample_count=10,
            ),
            OcrLineAnchor(1, 22.5, 0.0, "第二行", "第二行", 99.0, 0.5),
            OcrLineAnchor(2, 27.5, 0.0, "第三行", "第三行", 99.0, 0.5),
        ]

        corrected = apply_ocr_timing_assist(baseline, anchors)

        self.assertAlmostEqual(corrected[0].start, 14.75)

    def test_long_whisper_span_keeps_the_nearest_start_edge(self):
        baseline = [
            AlignmentItem(10.0, 20.0, "合并了下一行的字幕"),
            AlignmentItem(20.0, 23.0, "第二行"),
            AlignmentItem(23.0, 26.0, "第三行"),
        ]
        anchors = [
            OcrLineAnchor(
                0,
                12.5,
                0.0,
                "第一行歌词",
                "第一行歌词",
                99.0,
                0.5,
                target_coverage=1.0,
                visual_start=10.25,
                visual_end=14.75,
                sample_count=9,
            ),
            OcrLineAnchor(1, 21.5, 0.0, "第二行", "第二行", 99.0, 0.5),
            OcrLineAnchor(2, 24.5, 0.0, "第三行", "第三行", 99.0, 0.5),
        ]

        corrected = apply_ocr_timing_assist(baseline, anchors)

        self.assertAlmostEqual(corrected[0].start, 10.25)
        self.assertAlmostEqual(corrected[0].end, 14.75)

    def test_late_partial_fragment_extends_end_without_dragging_start(self):
        baseline = [
            AlignmentItem(0.0, 5.0, "第一行"),
            AlignmentItem(5.0, 10.0, "倒数第二行"),
            AlignmentItem(10.0, 10.9, "最后一行"),
        ]
        anchors = [
            OcrLineAnchor(0, 2.5, 0.0, "第一行", "第一行", 99.0, 0.5),
            OcrLineAnchor(1, 7.5, 0.0, "倒数第二行", "倒数第二行", 99.0, 0.5),
            OcrLineAnchor(
                2,
                15.0,
                1.0,
                "最后一行歌词",
                "歌词",
                90.0,
                0.8,
                target_coverage=0.4,
                visual_start=14.25,
                visual_end=15.75,
                sample_count=3,
            ),
        ]

        corrected = apply_ocr_timing_assist(baseline, anchors)

        self.assertAlmostEqual(corrected[2].start, 10.0)
        self.assertAlmostEqual(corrected[2].end, 16.0)
        self.assertEqual(corrected[2].review_reason, "ocr_partial_span")

    def test_long_baseline_does_not_reject_true_late_ocr_frame(self):
        lyrics = [
            LyricLine("昨日も今日も晴天で", "第一行"),
            LyricLine("二番目の歌詞", "第二行"),
            LyricLine("三番目の歌詞", "第三行"),
        ]
        baseline = [
            AlignmentItem(0.0, 20.0, "第一行"),
            AlignmentItem(20.0, 25.0, "第二行"),
            AlignmentItem(25.0, 30.0, "第三行"),
        ]
        frames = [OcrFrame(18.5, ("昨日も今日も晴天で",), (0.99,))]

        anchors = build_ocr_line_anchors(frames, lyrics, baseline)

        self.assertEqual([anchor.line_index for anchor in anchors], [0])

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

    def test_ocr_cache_round_trip_and_metadata_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "sample.mp4"
            cache = root / "sample.ocr.json"
            video.write_bytes(b"same-video")
            metadata = _ocr_metadata(video, 0.5)
            frames = [OcrFrame(1.5, ("日本語",), (0.95,))]
            save_ocr_cache(cache, frames, metadata)

            loaded = load_ocr_cache(cache, metadata)

            self.assertEqual(loaded, frames)
            self.assertIsNone(load_ocr_cache(cache, _ocr_metadata(video, 1.0)))

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

    def test_alignment_report_marks_ocr_timing_risk(self):
        aligned = [AlignmentItem(
            1.0,
            2.0,
            "第一行",
            "一番目",
            "一番目",
            100.0,
            "matched",
            review_reason="ocr_duration_outlier",
        )]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "sample.alignment.json"
            summary = save_alignment_report(aligned, report, 55.0)

        self.assertEqual(summary["needs_review"], [1])
        self.assertEqual(
            summary["review_reasons"],
            {"1": "ocr_duration_outlier"},
        )

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
