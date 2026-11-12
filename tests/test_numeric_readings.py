"""Numeric expansion, acoustic language selection and cache identity."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from g2pflow import G2PGroup, G2PReading

from lyric_alignment.backend import select_number_readings, spoken_text
from lyric_alignment.core import LineSpan
from lyric_alignment.numbers import NumberWord, number_readings, number_spans
from lyric_alignment.workflow import cache_metadata


class NumericReadingTests(unittest.TestCase):
    def test_countdown_is_not_read_as_one_integer(self):
        readings = dict(number_readings("3, 2, 1"))
        self.assertEqual(readings["en:cardinal"], "three, two, one")
        self.assertEqual(readings["ja:cardinal"], "三, 二, 一")
        self.assertEqual(readings["loan:cardinal"], "スリー, ツー, ワン")
        self.assertNotIn("hundred", readings["en:cardinal"])

    def test_integer_keeps_cardinal_and_digitwise_candidates(self):
        readings = dict(number_readings("321"))
        self.assertEqual(readings["en:digits"], "three two one")
        self.assertIn("hundred", readings["en:cardinal"])

    def test_numeric_punctuation_and_counter_are_preserved(self):
        self.assertEqual(spoken_text("3, 2, 1, Say!"), "3, 2, 1  Say ")
        self.assertEqual(spoken_text("3.14%"), "3.14%")
        self.assertEqual(number_spans("3人"), [(0, 2)])
        self.assertEqual(dict(number_readings("3人"))["ja:cardinal"], "三人")

    def test_stylized_alphanumeric_words_are_not_split_into_numbers(self):
        self.assertEqual(number_spans("Sk8er B4d"), [])

    def test_numeric_rule_changes_only_numeric_cache_identity(self):
        fingerprint = lambda path: {"path": str(path)}
        plain = cache_metadata(Path("song"), ["日文原詞"], "original", fingerprint)
        numeric = cache_metadata(Path("song"), ["3, 2, 1"], "original", fingerprint)
        self.assertNotIn("numeric_readings", plain)
        self.assertIn("numeric_readings", numeric)

    def test_voice_scores_select_english_or_japanese_without_text_guessing(self):
        for choice in (0, 1, 2):
            with self.subTest(choice=choice):
                word = NumberWord(text="3", language=None,
                    readings=[G2PReading(paths=[[G2PGroup(str(i), [str(i)])]]) for i in range(3)],
                    romans=["three", "san", "suri"], labels=["en", "ja", "loan"])
                wanted = word.romans[choice]
                with patch("lyric_alignment.backend._reading_likelihood",
                           side_effect=lambda values, romans, vocab: 5 if romans == [wanted] else 1):
                    records = select_number_readings(np.zeros((50, 2)), np.arange(50) * .02, {},
                        [[word]], [LineSpan(0, .2, .8, "coarse")], lambda *args: None)
                self.assertEqual(word.chosen, choice)
                self.assertEqual(records[0]["reading"], word.labels[choice])
                self.assertEqual(len(word.readings), 1)


if __name__ == "__main__":
    unittest.main()
