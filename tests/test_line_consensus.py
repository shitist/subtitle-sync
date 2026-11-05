import unittest
import numpy as np

from lyric_alignment.core import LineSpan
from tools.probe_line_consensus import consensus, voiced_expansion


def phone(name, start, end):
    return {"line_id": 0, "word": "原詞", "phone": name, "start": start, "end": end}


class ContextConsensusTests(unittest.TestCase):
    def test_every_phone_must_have_shared_acoustic_interval(self):
        result, common = consensus([
            [phone("m", 1, 1.2), phone("a", 1.2, 2)],
            [phone("m", 1.02, 1.22), phone("a", 1.22, 2.02)],
        ], 0)
        self.assertEqual((result.start, result.end), (1.02, 2))
        self.assertEqual(common[0]["end"], 1.2)

    def test_equal_line_boxes_do_not_hide_different_phone_locations(self):
        self.assertIsNone(consensus([
            [phone("m", 1, 1.1), phone("a", 1.1, 3)],
            [phone("m", 2, 2.1), phone("a", 2.1, 3)],
        ], 0))

    def test_missing_phone_is_not_accepted(self):
        self.assertIsNone(consensus([[phone("m", 1, 1), phone("a", 1, 2)],
                                     [phone("m", 1, 1.1), phone("a", 1.1, 2)]], 0))

    def test_different_pronunciations_are_not_merged(self):
        self.assertIsNone(consensus([[phone("m", 1, 2)], [phone("n", 1, 2)]], 0))

    def test_two_contexts_are_required(self):
        self.assertIsNone(consensus([[phone("m", 1, 2)]], 0))

    def test_expansion_needs_voice_evidence_for_each_endpoint_separately(self):
        base = LineSpan(0, 2, 4, "coarse")
        candidate = LineSpan(0, 1, 6, "refined")
        evidence = voiced_expansion(base, candidate, np.arange(8), np.array([1, 1, 1, 1, -3, -3, -3, -3]))
        self.assertEqual((candidate.start, candidate.end), (1, 4))
        self.assertEqual((base.start, base.end), (2, 4))
        self.assertGreater(evidence["start"]["vocal_log_odds"], 0)
        self.assertLess(evidence["end"]["vocal_log_odds"], 0)


if __name__ == "__main__":
    unittest.main()
