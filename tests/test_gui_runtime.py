import json
import sys
import tempfile
import unittest
from pathlib import Path

import subtitle_sync_gui as gui


class _DummyButton:
    def config(self, **_kwargs):
        pass


class _FakeApp:
    def __init__(self):
        self.running = True
        self.start_btn = _DummyButton()
        self.messages = []

    def _log(self, text):
        self.messages.append(text)

    def _set_status(self, text, _color):
        self.status = text

    def after(self, _delay, callback):
        callback()


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


if __name__ == "__main__":
    unittest.main()
