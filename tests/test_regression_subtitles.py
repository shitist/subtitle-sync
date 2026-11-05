import unittest
from tools.regression_subtitles import evaluate


class AcceptanceTests(unittest.TestCase):
    def test_one_second_is_checked_for_each_boundary_not_the_average(self):
        reference = [{"start":10.0,"end":12.0,"text":"字幕"}]
        self.assertTrue(evaluate([{"start":11.0,"end":13.0}],reference)["initial_pass"])
        result = evaluate([{"start":10.0,"end":13.001}],reference)
        self.assertFalse(result["initial_pass"])
        self.assertEqual(result["failed_lines"],[1])

    def test_missing_and_empty_results_never_pass(self):
        self.assertFalse(evaluate([],[])["initial_pass"])
        self.assertFalse(evaluate([],[{"start":1,"end":2,"text":"字幕"}])["initial_pass"])

    def test_one_failed_cue_fails_the_whole_song(self):
        reference = [{"start":i*3,"end":i*3+2,"text":str(i)} for i in range(10)]
        actual = [dict(row) for row in reference]
        actual[-1]["start"] += 1.001
        result = evaluate(actual,reference)
        self.assertFalse(result["initial_pass"])
        self.assertEqual(result["within_one_second"],9)
