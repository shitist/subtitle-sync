import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subtitle_style as style
import subtitle_sync as sync
import subtitle_sync_gui as gui
from asr_models import LYRIC_MODEL
from PySide6.QtCore import QSettings


ASS = """[Script Info]
PlayResX: 384
PlayResY: 288
[V4+ Styles]
Format: Name, Fontname, Fontsize, Bold, Outline, MarginV
Style: Default,Arial,16,0,1,10
Style: Other,Arial,20,0,1,10
[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,中文（测试） 日本語
"""


class SubtitleStyleTests(unittest.TestCase):
    def test_presets_match_toolbox_sizes(self):
        self.assertEqual(style.font_settings(), ("Noto Sans CJK SC", 16 * 1.1))
        self.assertEqual(style.font_settings("noto-jp"), ("Noto Sans CJK JP", 17.6))
        self.assertEqual(style.font_settings("yahei"), ("Microsoft YaHei", 16))

    def test_style_changes_only_default_font_and_size(self):
        actual = style.apply_ass_style(ASS, "noto-jp", 22.5)
        expected = ASS.replace("Style: Default,Arial,16,", "Style: Default,Noto Sans CJK JP,22.5,")
        self.assertEqual(actual, expected)

    def test_font_change_preserves_windows_line_endings(self):
        actual = style.apply_ass_style(ASS.replace("\n", "\r\n"))
        self.assertNotIn("\n", actual.replace("\r\n", ""))

    def test_invalid_sizes_are_rejected(self):
        for size in (0, -1, 97, float("nan"), float("inf")):
            with self.subTest(size=size), self.assertRaises(ValueError):
                style.font_settings("noto-sc", size)

    def test_bundled_fonts_match_toolbox_assets(self):
        expected = {
            "NotoSansCJK-Regular.ttc": "b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a",
            "NotoSansCJK-Bold.ttc": "faa5f3656a78b2e2d450d27fe8382c778bc2b6bb5ea29c986664a6a435056ceb",
        }
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((style.FONT_DIR / filename).read_bytes()).hexdigest(), digest)

    def test_both_engines_receive_font_and_size(self):
        for model in (LYRIC_MODEL, "large-v3"):
            config = {"mode": "single", "video": "v.mp4", "lyrics": "l.txt",
                      "model": model, "audio_source": "original", "subtitle_font": "noto-jp",
                      "subtitle_size": 22.5}
            command = gui._build_processing_command("python", config)
            self.assertEqual(command[command.index("--subtitle-font") + 1], "noto-jp")
            self.assertEqual(command[command.index("--subtitle-size") + 1], "22.5")
            config["no_burn"] = True
            command = gui._build_processing_command("python", config)
            self.assertNotIn("--subtitle-font", command)
            self.assertNotIn("--subtitle-size", command)

    def test_acoustic_dispatch_keeps_font_selection(self):
        with tempfile.TemporaryDirectory() as temp, patch("lyric_alignment.workflow.process_known_lyrics") as process:
            sync.process_single(Path(temp) / "v.mp4", Path(temp) / "l.txt", model_name=LYRIC_MODEL,
                                subtitle_font="noto-jp", subtitle_size=21)
            self.assertEqual(process.call_args.kwargs["subtitle_font"], "noto-jp")
            self.assertEqual(process.call_args.kwargs["subtitle_size"], 21)

    def test_cached_acoustic_result_burns_with_selected_style(self):
        result = {"duration": 3, "lines": [
            {"line_id": 0, "start": 0.5, "end": 2, "stage": "refined", "flags": []},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, lyrics = root / "v.mp4", root / "l.txt"
            video.write_bytes(b"video")
            lyrics.write_text("原詞|中文", encoding="utf-8")
            with patch("lyric_alignment.workflow.load_cache", return_value=result), patch.object(sync, "burn_subtitles") as burn:
                sync.process_single(video, lyrics, root / "out.mp4", model_name=LYRIC_MODEL,
                                    subtitle_font="noto-jp", subtitle_size=21)
            self.assertEqual(burn.call_args.kwargs, {"subtitle_font": "noto-jp", "subtitle_size": 21})

    def test_batch_passes_style_to_every_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.mp4").touch()
            (root / "a.txt").write_text("原詞|中文", encoding="utf-8")
            with patch.object(sync, "process_single") as process:
                sync.process_batch(root, root, subtitle_font="noto-jp", subtitle_size=21)
            self.assertEqual(process.call_args.kwargs["subtitle_font"], "noto-jp")
            self.assertEqual(process.call_args.kwargs["subtitle_size"], 21)

    def test_gui_defaults_persistence_and_srt_only_controls(self):
        application = gui.QApplication.instance() or gui.QApplication([])
        with tempfile.TemporaryDirectory() as temp:
            settings_path = str(Path(temp) / "settings.ini")
            settings = QSettings(settings_path, QSettings.Format.IniFormat)
            with patch.object(gui, "QSettings", return_value=settings):
                window = gui.App()
                self.assertEqual(window.subtitle_font_combo.currentData(), "noto-sc")
                self.assertEqual(window.subtitle_size_spin.value(), 17.6)
                window.subtitle_font_combo.setCurrentIndex(window.subtitle_font_combo.findData("noto-jp"))
                window.subtitle_size_spin.setValue(23.5)
                window.no_burn_cb.setChecked(True)
                self.assertFalse(window.subtitle_font_combo.isEnabled())
                self.assertFalse(window.subtitle_size_spin.isEnabled())
                window.no_burn_cb.setChecked(False)
                self.assertTrue(window.subtitle_font_combo.isEnabled())
                window.close()
                restored = gui.App()
                self.assertEqual(restored.subtitle_font_combo.currentData(), "noto-jp")
                self.assertEqual(restored.subtitle_size_spin.value(), 23.5)
                restored.close()
        application.processEvents()


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
class SubtitleRenderTests(unittest.TestCase):
    def test_real_burn_uses_selected_fonts_and_toolbox_defaults(self):
        ffmpeg = sync._get_ffmpeg_path()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, srt = root / "source.mp4", root / "歌词, quote's.srt"
            subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:s=640x360:r=5:d=0.4",
                            "-c:v", "libx264", "-y", str(source)], capture_output=True, check=True)
            srt.write_text("1\n00:00:00,000 --> 00:00:00,400\n字幕测试 日本語テスト\n", encoding="utf-8")
            for font, face in (("noto-sc", "NotoSansCJKsc-Regular"), ("noto-jp", "NotoSansCJKjp-Regular"),
                               ("yahei", "MicrosoftYaHei")):
                with self.subTest(font=font):
                    output = root / f"{font}.mp4"
                    logs, styles, ass_paths, rendered_styles = [], [], [], []

                    def run(command, _label):
                        if "-vf" in command:
                            rendered_styles.append(ass_paths[0].read_text(encoding="utf-8"))
                        result = subprocess.run(command, capture_output=True, check=True)
                        logs.append(result.stderr.decode("utf-8", errors="replace"))
                        if command[-1].endswith(".ass"):
                            ass_paths.append(Path(command[-1]))
                            styles.append(ass_paths[0].read_text(encoding="utf-8"))

                    with patch.object(sync, "_run", side_effect=run):
                        sync.burn_subtitles(source, srt, output, subtitle_font=font)
                    self.assertTrue(output.is_file())
                    self.assertIn("Style: Default,Arial,16,", styles[0])
                    family, size = style.font_settings(font)
                    self.assertIn(f"Style: Default,{family},{size:g},", rendered_styles[0])
                    log = "\n".join(logs)
                    self.assertIn(face, log)
                    self.assertNotIn("Thin", log)
                    frame = subprocess.run([ffmpeg, "-v", "error", "-i", str(output), "-frames:v", "1",
                                            "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=True)
                    self.assertGreater(max(frame.stdout), 100)

    def test_filter_handles_unicode_spaces_commas_and_apostrophes(self):
        ffmpeg = sync._get_ffmpeg_path()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "日本語, quote's"
            folder.mkdir()
            ass = folder / "字幕.ass"
            # A complete ASS file produced by FFmpeg supplies all layout fields.
            srt = folder / "字幕.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n字幕测试\n", encoding="utf-8")
            subprocess.run([ffmpeg, "-v", "error", "-i", str(srt), str(ass)], capture_output=True, check=True)
            ass.write_text(style.apply_ass_style(ass.read_text(encoding="utf-8-sig")), encoding="utf-8")
            result = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:s=640x360",
                                     "-vf", style.subtitle_filter(ass), "-frames:v", "1", "-pix_fmt", "gray",
                                     "-f", "rawvideo", "-"], capture_output=True, check=True)
            self.assertGreater(max(result.stdout), 100)


if __name__ == "__main__":
    unittest.main()
