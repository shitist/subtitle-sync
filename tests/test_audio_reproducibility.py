"""Reproducible preprocessing: no song data or reference timings."""
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import subtitle_sync as sync
from lyric_alignment.workflow import cache_metadata


class SeparationReproducibilityTests(unittest.TestCase):
    def test_seed_is_applied_inside_demucs_child_before_module_runs(self):
        offsets = []
        torch = types.SimpleNamespace(manual_seed=Mock())
        numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=Mock()))
        runner = types.SimpleNamespace(run_module=Mock(
            side_effect=lambda *a, **kw: offsets.append(random.randint(0, 22050))))
        state = random.getstate()
        try:
            with patch.dict(sys.modules, {"torch": torch, "numpy": numpy, "runpy": runner}):
                for _ in range(2):
                    random.seed()  # Simulate two freshly created interpreters.
                    exec(sync.DEMUCS_LAUNCHER, {})
            self.assertEqual(offsets[0], offsets[1])
            self.assertEqual(torch.manual_seed.call_args.args, (0,))
            self.assertEqual(numpy.random.seed.call_args.args, (0,))
            runner.run_module.assert_called_with("demucs", run_name="__main__")
        finally:
            random.setstate(state)

    def test_separation_launch_keeps_shift_augmentation_and_exact_output_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = root / "htdemucs" / "audio" / "vocals.wav"
            expected.parent.mkdir(parents=True)
            expected.touch()
            with patch.object(sync, "_get_ffmpeg_path", return_value="ffmpeg"), patch.object(sync, "_run") as run:
                self.assertEqual(sync.separate_vocals(root / "audio.wav", root, deterministic=True), expected)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], [sys.executable, "-c", sync.DEMUCS_LAUNCHER])
            self.assertEqual(command[command.index("--shifts") + 1], "1")
            self.assertEqual(command[command.index("-n") + 1], "htdemucs")

    def test_application_cache_identity_is_unchanged_by_experimental_seed_mode(self):
        metadata = cache_metadata(Path("video"), ["原詞"], "vocals", lambda p: {"size": 1})
        self.assertNotIn("audio_preparation", metadata)

    def test_whisper_default_cache_identity_is_unchanged(self):
        with patch.object(sync, "_file_fingerprint", return_value={"size": 1}):
            self.assertNotIn("audio_preparation", sync._transcript_metadata(Path("video"), False, "vocals"))

    def test_default_application_command_keeps_legacy_demucs_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = root / "htdemucs" / "audio" / "vocals.wav"
            expected.parent.mkdir(parents=True)
            expected.touch()
            with patch.object(sync, "_get_ffmpeg_path", return_value="ffmpeg"), patch.object(sync, "_run") as run:
                sync.separate_vocals(root / "audio.wav", root)
            self.assertEqual(run.call_args.args[0][:3], [sys.executable, "-m", "demucs"])


if __name__ == "__main__":
    unittest.main()
