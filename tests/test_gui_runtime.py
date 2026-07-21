import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subtitle_sync_gui as gui


class _FakeApp:
    def __init__(self):
        self.running = True
        self.messages = []
        self.dialogs = []

    def _log(self, text):
        self.messages.append(text)

    def _set_status(self, text, _kind):
        self.status = text

    def _show_message(self, kind, title, text):
        self.dialogs.append((kind, title, text))

    def _processing_finished(self):
        self.running = False


class PythonSelectionTests(unittest.TestCase):
    def test_invalid_first_path_candidate_is_not_selected(self):
        probes = [
            {
                "path": "first-python.exe",
                "priority": 80,
                "version": [3, 11, 0],
                "modules": {name: False for name in gui.PYTHON_MODULES},
            },
            {
                "path": "complete-python.exe",
                "priority": 10,
                "version": [3, 12, 0],
                "modules": {name: True for name in gui.PYTHON_MODULES},
            },
        ]

        selected = gui._choose_python(
            probes,
            {"faster_whisper", "rapidfuzz", "demucs"},
        )

        self.assertEqual(selected["path"], "complete-python.exe")

    def test_interpreter_missing_current_mode_dependency_is_rejected(self):
        probe = {
            "path": "no-demucs.exe",
            "priority": 100,
            "version": [3, 12, 0],
            "modules": {
                **{name: True for name in gui.PYTHON_MODULES},
                "demucs": False,
            },
        }

        selected = gui._choose_python(
            [probe],
            {"faster_whisper", "rapidfuzz", "demucs"},
        )

        self.assertIsNone(selected)


class PersistentDiagnosticsTests(unittest.TestCase):
    def test_run_output_is_written_to_latest_log_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_paths = (
                gui.LOG_DIR,
                gui.LATEST_LOG,
                gui.LATEST_DIAGNOSTIC,
            )
            original_showinfo = gui.messagebox.showinfo
            gui.LOG_DIR = root
            gui.LATEST_LOG = root / "latest.log"
            gui.LATEST_DIAGNOSTIC = root / "latest.json"
            gui.messagebox.showinfo = lambda *_args, **_kwargs: None
            try:
                fake = _FakeApp()
                command = [sys.executable, "-c", "print('diagnostic-ok')"]
                context = {
                    "run_id": "test-run",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "status": "starting",
                    "mode": "single",
                    "audio_source": "original",
                    "required_modules": ["faster_whisper", "rapidfuzz"],
                    "selected_python": {
                        "path": sys.executable,
                        "source": "test",
                    },
                    "python_candidates": [],
                    "command": command,
                    "log_path": str(gui.LATEST_LOG),
                    "run_log_path": str(root / "run-test.log"),
                }

                gui.App._run_cmd(fake, command, context)

                self.assertIn("diagnostic-ok", gui.LATEST_LOG.read_text(encoding="utf-8"))
                diagnostic = json.loads(gui.LATEST_DIAGNOSTIC.read_text(encoding="utf-8"))
                self.assertEqual(diagnostic["status"], "completed")
                self.assertEqual(diagnostic["exit_code"], 0)
            finally:
                gui.LOG_DIR, gui.LATEST_LOG, gui.LATEST_DIAGNOSTIC = original_paths
                gui.messagebox.showinfo = original_showinfo


class RuntimePresentationTests(unittest.TestCase):
    def test_stage_line_drives_progress(self):
        event = gui.parse_runtime_line("[4/7] Whisper large-v3 日语识别...")

        self.assertEqual(event["kind"], "stage")
        self.assertEqual(event["stage"], 4)
        self.assertEqual(event["progress"], 38)

    def test_recognized_text_is_not_copied_into_visible_log(self):
        event = gui.parse_runtime_line(
            "[012] 24.10s → 26.20s [words: 9] 一整段识别文字"
        )

        self.assertEqual(event, {"kind": "recognition", "count": 12})

    def test_missing_module_error_remains_visible(self):
        event = gui.parse_runtime_line(
            r"C:\Python\python.exe: No module named demucs"
        )

        self.assertEqual(event["kind"], "log")
        self.assertIn("No module named demucs", event["message"])

    def test_dragged_video_and_lyrics_are_classified(self):
        result = gui.classify_dropped_paths(
            [r"C:\media\song.mp4", r"C:\media\歌词.txt"]
        )

        self.assertEqual(result["video"], r"C:\media\song.mp4")
        self.assertEqual(result["lyrics"], r"C:\media\歌词.txt")


if __name__ == "__main__":
    unittest.main()
