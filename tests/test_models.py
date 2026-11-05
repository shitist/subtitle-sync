"""模型目录、运行环境和通用文本锚点回归。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import asr_models as models
import subtitle_sync as sync
import subtitle_sync_gui as gui


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_contains_only_whisper_and_lyric_alignment(self):
        self.assertEqual(models.SUPPORTED_MODELS, (*models.WHISPER_MODELS, models.LYRIC_MODEL))
        self.assertEqual({item["id"] for item in gui.MODEL_OPTIONS}, set(models.SUPPORTED_MODELS))

    def test_removed_model_is_rejected_before_transcription_or_download(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(sync, "transcribe_audio") as transcribe:
            with self.assertRaisesRegex(ValueError, "不支持的语音模型"):
                sync.process_single(Path(temp) / "video.mp4", Path(temp) / "lyrics.txt", model_name="qwen3-asr-1.7b")
            transcribe.assert_not_called()
        with self.assertRaisesRegex(ValueError, "不支持的 Whisper 模型"):
            sync.transcribe_audio(Path("audio.wav"), "qwen3-asr-1.7b")

    def test_probe_no_longer_requires_removed_asr_package(self):
        record = gui._probe_python(sys.executable, "test", 100)
        self.assertIsNone(record["error"])
        self.assertEqual(set(record["modules"]), set(gui.PYTHON_MODULES))
        self.assertNotIn("qwen3_asr", record["modules"])
        self.assertNotIn("nagisa", record["modules"])

    def test_whisper_mode_has_no_transformers_requirement(self):
        required = gui._required_modules({"model": "large-v3", "audio_source": "original"})
        self.assertEqual(required, {"faster_whisper", "rapidfuzz", "imageio_ffmpeg"})

    def test_cached_whisper_flow_does_not_run_audio_processing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, lyrics, output = root / "clip.mp4", root / "lyrics.txt", root / "out.mp4"
            video.write_bytes(b"video")
            lyrics.write_text("一番目|第一行\n二番目|第二行\n", encoding="utf-8")
            segments = [sync.SimpleSegment(1, 4, "一番目二番目", [
                sync.WordTiming(1, 2, "一番目"), sync.WordTiming(3, 4, "二番目"),
            ])]
            metadata = sync._transcript_metadata(video, True, "vocals", "large-v3")
            sync.save_transcript_cache(output.with_suffix(".whisper.json"), segments, metadata)
            with patch.object(sync, "transcribe_audio", side_effect=AssertionError("unexpected inference")), \
                 patch.object(sync, "extract_audio", side_effect=AssertionError("unexpected extraction")):
                sync.process_single(video, lyrics, output, model_name="large-v3", no_burn=True)
            self.assertIn("第一行", output.with_suffix(".srt").read_text(encoding="utf-8"))
            self.assertFalse(output.with_suffix(".asr.json").exists())


class StrictTextAnchorTests(unittest.TestCase):
    def test_short_english_match_does_not_accept_a_similar_phrase(self):
        words = [sync.WordTiming(1, 2, "Alpha"), sync.WordTiming(5, 6, "open floor"), sync.WordTiming(9, 10, "Omega")]
        lyrics = [sync.LyricLine("Alpha", "甲"), sync.LyricLine("Open door", "乙"), sync.LyricLine("Omega", "丙")]
        result = sync.align_lyrics_to_segments([sync.SimpleSegment(1, 10, "", words)], lyrics, strict_anchors=True)
        self.assertEqual(result[1].status, "interpolated")

    def test_zero_duration_word_is_not_an_anchor(self):
        words = [sync.WordTiming(1, 2, "Alpha"), sync.WordTiming(5, 5, "Yes"), sync.WordTiming(9, 10, "Omega")]
        lyrics = [sync.LyricLine("Alpha", "甲"), sync.LyricLine("Yes", "乙"), sync.LyricLine("Omega", "丙")]
        result = sync.align_lyrics_to_segments([sync.SimpleSegment(1, 10, "", words)], lyrics, strict_anchors=True)
        self.assertEqual(result[1].status, "interpolated")

    def test_zero_duration_tail_does_not_extend_phrase(self):
        words = [sync.WordTiming(1, 2, "Can"), sync.WordTiming(2, 3, "you"),
                 sync.WordTiming(3, 3, "hear"), sync.WordTiming(12, 12, "me")]
        result = sync.align_lyrics_to_segments([sync.SimpleSegment(1, 12, "", words)],
            [sync.LyricLine("Can you hear me", "听得见吗")], strict_anchors=True)
        self.assertEqual((result[0].start, result[0].end), (1, 3))

    def test_chinese_only_keeps_exact_english_anchors(self):
        words = [sync.WordTiming(1, 1.4, "Start"), sync.WordTiming(1.4, 2, "now"),
                 sync.WordTiming(5, 6, "音声"), sync.WordTiming(9, 10, "Omega")]
        lyrics = [sync.LyricLine("", "Start now"), sync.LyricLine("", "中文"), sync.LyricLine("", "Omega")]
        result = sync.align_lyrics_to_segments([sync.SimpleSegment(1, 10, "", words)], lyrics, strict_anchors=True)
        self.assertEqual([row.status for row in result], ["latin_anchor", "interpolated", "latin_anchor"])


if __name__ == "__main__":
    unittest.main()
