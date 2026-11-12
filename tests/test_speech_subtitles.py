"""Focused checks for speech captions, mode isolation, and WhisperX integration."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import speech_catalog as catalog
import speech_subtitles as speech
import subtitle_sync as sync
import subtitle_sync_gui as gui
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication


def word(text, start, end, stage="aligned"):
    return {"word": text, "start": start, "end": end, "stage": stage}


class CaptionTests(unittest.TestCase):
    def test_language_list_matches_whisper(self):
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        self.assertEqual(set(catalog.LANGUAGE_CODES), set(_LANGUAGE_CODES))
        self.assertEqual(len(catalog.ALIGN_MODELS), 41)

    def test_complete_aligned_words_replace_asr_times(self):
        original = {"text": "Hello world!", "start": 0, "end": 3,
                    "words": [word("Hello", 0, 1), word("world!", 1, 3)]}
        selected, stage = speech.choose_words(original, {"words": [word("Hello", .2, .8), word("world!", 1.2, 2)]})
        self.assertEqual(stage, "aligned")
        self.assertEqual(selected[0]["start"], .2)

    def test_incomplete_alignment_retains_asr_words_without_estimating(self):
        original = {"text": "Hello world!", "start": 0, "end": 3,
                    "words": [word("Hello", 0, 1), word("world!", 1, 3)]}
        selected, stage = speech.choose_words(original, {"words": [word("Hello", .2, .8), {"word": "world!"}]})
        self.assertEqual(stage, "asr")
        self.assertEqual([(w["start"], w["end"]) for w in selected], [(0, 1), (1, 3)])

    def test_missing_text_in_alignment_is_not_silently_dropped(self):
        original = {"text": "Hello world", "start": 1, "end": 3}
        selected, stage = speech.choose_words(original, {"words": [word("Hello", 1, 2)]})
        self.assertEqual(stage, "asr_segment")
        self.assertEqual(selected[0]["word"], "Hello world")

    def test_pause_creates_a_gap_without_stretching_cues(self):
        result = speech.make_cues([word("Hello", 1, 1.5), word("world", 1.5, 2), word("Again", 8, 9)], "en")
        self.assertEqual([(r["start"], r["end"]) for r in result], [(1, 2), (8, 9)])
        self.assertEqual(result[0]["text"], "Hello world")

    def test_japanese_words_are_not_separated_by_inserted_spaces(self):
        result = speech.make_cues([word("今日", 1, 2), word("は", 2, 2.5), word("晴れ。", 2.5, 3)], "ja")
        self.assertEqual(result[0]["text"], "今日は晴れ。")

    def test_punctuation_splits_at_word_boundaries(self):
        result = speech.make_cues([word("Hi!", 1, 2), word("Yes.", 2.1, 3)], "en")
        self.assertEqual([(r["start"], r["end"]) for r in result], [(1, 2), (2.1, 3)])

    def test_cached_speech_generates_srt_without_lyrics_or_audio_processing(self):
        with tempfile.TemporaryDirectory() as temp:
            video, output = Path(temp) / "clip.mp4", Path(temp) / "out.mp4"
            video.touch()
            metadata = {"profile": speech.PROFILE, "video": sync._file_fingerprint(video), "model": "small",
                        "language": "en", "vad": True, "whisperx": "test", "faster_whisper": "test"}
            output.with_suffix(".speech.json").write_text(json.dumps({
                "metadata": metadata, "transcript": {"language": "en"},
                "words": [word("Hello!", 1.25, 2.5)],
                "alignment": {"model": "example", "aligned_segments": 1, "asr_segments": 0, "errors": []},
            }), encoding="utf-8")
            with patch.object(speech, "version", return_value="test"), \
                 patch.object(speech, "transcribe", side_effect=AssertionError("unexpected ASR")), \
                 patch.object(sync, "_run", side_effect=AssertionError("unexpected audio extraction")):
                report = speech.process_video(video, output, helpers=sync, model_name="small", language="en", no_burn=True)
            self.assertIn("00:00:01,250 --> 00:00:02,500", output.with_suffix(".srt").read_text(encoding="utf-8-sig"))
            self.assertEqual(report["language"], "en")

    def test_cli_speech_dispatch_requires_no_lyrics(self):
        args = ["subtitle_sync.py", "--task", "speech", "--video", "clip.mp4", "--language", "fr", "--no-burn"]
        with patch.object(sys, "argv", args), patch.object(speech, "process_video") as run:
            sync.main()
        self.assertEqual(run.call_args.kwargs["language"], "fr")
        self.assertTrue(run.call_args.kwargs["no_burn"])

    def test_cli_speech_batch_dispatch_requires_no_lyrics_dir(self):
        args = ["subtitle_sync.py", "--task", "speech", "--batch", "--video-dir", "clips", "--language", "auto"]
        with patch.object(sys, "argv", args), patch.object(speech, "process_batch") as run:
            sync.main()
        run.assert_called_once()

    def test_unsupported_alignment_language_retains_original_times(self):
        transcript = {"language": "yue", "segments": [{"text": "你好", "start": 2, "end": 3,
                                                        "words": [word("你好", 2.1, 2.9)]}]}
        words, report = speech.refine(Path("not-loaded.wav"), transcript)
        self.assertEqual(report["asr_segments"], 1)
        self.assertEqual(words[0]["start"], 2.1)
        self.assertIsNone(report["model"])

    def test_failed_model_download_can_reuse_completed_transcription(self):
        with tempfile.TemporaryDirectory() as temp:
            video, output = Path(temp) / "clip.mp4", Path(temp) / "out.mp4"
            video.touch()
            transcript = {"language": "en", "segments": [{"text": "Hello!", "start": 1, "end": 2}]}
            with patch.object(speech, "version", return_value="test"), patch.object(sync, "_run"), \
                 patch.object(speech, "transcribe", return_value=transcript) as asr, \
                 patch.object(speech, "refine", side_effect=RuntimeError("download failed")):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    speech.process_video(video, output, helpers=sync, no_burn=True)
                asr.assert_called_once()
            with patch.object(speech, "version", return_value="test"), patch.object(sync, "_run"), \
                 patch.object(speech, "transcribe", side_effect=AssertionError("ASR repeated")), \
                 patch.object(speech, "refine", return_value=([word("Hello!", 1.2, 1.9)], {
                     "model": "example", "aligned_segments": 1, "asr_segments": 0, "errors": []})):
                speech.process_video(video, output, helpers=sync, no_burn=True)
            self.assertTrue(output.with_suffix(".srt").is_file())


class SpeechGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.temp.name) / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(gui, "QSettings", return_value=settings):
            self.window = gui.App()

    def tearDown(self):
        self.window.close()
        self.temp.cleanup()

    def test_recommended_lyric_alignment_is_first_with_short_description(self):
        self.assertEqual(self.window.model_combo.itemData(0), gui.LYRIC_MODEL)
        self.assertEqual(self.window.model_combo.itemText(0), "歌词声学对齐（推荐）")
        self.assertEqual(self.window.model_combo.currentData(), gui.LYRIC_MODEL)
        self.assertEqual(self.window.model_description.text(), "MMS+TIFA路线：轻量准确，推荐使用。")
        self.window.task_combo.setCurrentIndex(1)
        self.assertEqual(self.window.model_combo.currentData(), "large-v3")

    def test_mode_switch_hides_lyrics_and_preserves_each_modes_settings(self):
        self.window.model_combo.setCurrentIndex(self.window.model_combo.findData("lyrics-mms-tifa"))
        self.window.task_combo.setCurrentIndex(1)
        self.assertTrue(self.window.lyrics_edit.isHidden())
        self.assertTrue(self.window.lyrics_dir_edit.isHidden())
        self.assertEqual(self.window.model_combo.count(), 4)
        self.assertEqual(self.window.no_vad_cb.text(), "启用 VAD")
        self.assertTrue(self.window.no_vad_cb.isChecked())
        self.assertEqual(self.window.audio_source_combo.currentData(), "original")
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("en"))
        self.window.task_combo.setCurrentIndex(0)
        self.assertEqual(self.window.model_combo.currentData(), "lyrics-mms-tifa")
        self.assertFalse(self.window.lyrics_edit.isHidden())
        self.assertFalse(self.window.no_vad_cb.isEnabled())
        self.window.task_combo.setCurrentIndex(1)
        self.assertEqual(self.window.language_combo.currentData(), "en")

    def test_all_whisper_languages_are_available(self):
        self.assertEqual(self.window.language_combo.count(), len(catalog.LANGUAGE_CODES) + 1)
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("yue"))
        self.assertIn("没有配置声学对齐模型", self.window.language_combo.toolTip())
        self.assertFalse(hasattr(self.window, "alignment_description"))

    def test_batch_copy_matches_single_file(self):
        single = (self.window.video_edit, self.window.lyrics_edit, self.window.output_edit)
        batch = (self.window.video_dir_edit, self.window.lyrics_dir_edit, self.window.output_dir_edit)
        for row, (file_edit, directory_edit) in enumerate(zip(single, batch)):
            file_label = file_edit.parentWidget().layout().itemAtPosition(row, 0).widget()
            directory_label = directory_edit.parentWidget().layout().itemAtPosition(row, 0).widget()
            self.assertEqual(directory_label.text(), file_label.text())
            self.assertTrue(directory_edit.directories)
        self.assertEqual(self.window.video_dir_edit.placeholderText(), "选择或拖入视频目录")
        self.assertEqual(self.window.lyrics_dir_edit.placeholderText(), "选择或拖入 UTF-8 歌词文本目录")
        self.assertEqual(self.window.output_dir_edit.placeholderText(), self.window.output_edit.placeholderText())
        self.assertFalse(hasattr(self.window, "batch_note"))
        description = self.window.model_description.text()
        self.window.mode_tabs.setCurrentIndex(1)
        self.assertEqual(self.window.model_description.text(), description)

    def test_file_panel_and_options_are_compact(self):
        self.assertFalse(hasattr(self.window, "drop_area"))
        self.assertTrue(self.window.video_edit.acceptDrops())
        self.assertIn("拖入", self.window.video_edit.placeholderText())
        self.window.task_combo.setCurrentIndex(1)
        self.window.show()
        self.app.processEvents()
        controls = (self.window.no_vad_cb, self.window.no_burn_cb, self.window.refresh_cb)
        self.assertTrue(all(control.parentWidget() is self.window.option_checks for control in controls))
        self.assertEqual(len({control.y() for control in controls}), 1)
        self.assertNotIn("首次", self.window.model_description.text())
        self.assertNotIn("首次", self.window.model_combo.toolTip())

    def test_speech_command_omits_mv_options(self):
        command = gui._build_processing_command("python", {"task": "speech", "mode": "single",
            "video": "clip.mp4", "model": "small", "language": "de", "audio_source": "vocals",
            "no_burn": True, "ocr_assist": True, "refresh": True})
        self.assertNotIn("--lyrics", command)
        self.assertNotIn("--ocr-assist", command)
        self.assertEqual(command[command.index("--language") + 1], "de")
        self.assertEqual(command[command.index("--audio-source") + 1], "original")

    def test_speech_progress_has_separate_stages(self):
        result = gui.parse_runtime_line("视频字幕进度: 对齐时间轴 2/4")
        self.assertEqual(result["kind"], "speech_progress")
        self.assertEqual(result["progress"], 77)

    def test_environment_does_not_require_mv_modules(self):
        required = gui._required_modules({"task": "speech", "audio_source": "original"})
        self.assertIn("whisperx", required)
        self.assertNotIn("demucs", required)
        self.assertNotIn("g2pflow", required)

    def test_caption_controls_remain_visible_at_minimum_size(self):
        self.window.task_combo.setCurrentIndex(1)
        self.window.resize(720, 560)
        self.window.show()
        self.app.processEvents()
        for widget in (self.window.start_btn, self.window.progress_bar, self.window.task_combo):
            origin = widget.mapTo(self.window, widget.rect().topLeft())
            self.assertGreaterEqual(origin.y(), 0)
            self.assertLessEqual(origin.y() + widget.height(), self.window.height())


if __name__ == "__main__":
    unittest.main()
