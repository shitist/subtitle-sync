"""Numeric pronunciation alternatives; source lyrics remain unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from g2pflow import G2PGroup, G2PReading, G2PWord, Language
from g2pflow.converters.base import G2PConversionError

RULES_VERSION = "ja-en-numeric-acoustic-v1"
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])")
_COUNTER = re.compile(r"(?:人|日|月|年|時|分|秒|回|個|本|匹|枚|歳|才|番|階|杯|台|度|曲|歩|点|つ)")
_LOAN = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion and point minus plus percent".split(),
    "ゼロ ワン ツー スリー フォー ファイブ シックス セブン エイト ナイン テン イレブン トゥエルブ サーティーン フォーティーン フィフティーン シックスティーン セブンティーン エイティーン ナインティーン トゥエンティ サーティ フォーティ フィフティ シックスティ セブンティ エイティ ナインティ ハンドレッド サウザンド ミリオン ビリオン トリリオン アンド ポイント マイナス プラス パーセント".split(),
))


@dataclass
class NumberWord(G2PWord):
    romans: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    chosen: int = 0

    def select(self, index: int) -> None:
        self.chosen = index
        self.readings = [self.readings[index]]


def number_spans(text: str) -> list[tuple[int, int]]:
    matches = list(_NUMBER.finditer(text))
    spans = []
    for match in matches:
        start, end = match.span()
        counter = _COUNTER.match(text, end)
        if counter:
            end = counter.end()
        if spans and re.fullmatch(r"[\s,、]+", text[spans[-1][1]:start] or "x"):
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def number_readings(text: str) -> list[tuple[str, str]]:
    from num2words import num2words
    matches = list(_NUMBER.finditer(text))
    results = []
    for mode in ("cardinal", "digits"):
        if mode == "digits" and not any(len(re.sub(r"\D", "", m.group())) > 1 for m in matches):
            continue
        for language in ("en", "ja", "loan"):
            pieces, cursor = [], 0
            for match in matches:
                raw = match.group().replace(",", "")
                percent = raw.endswith("%")
                raw = raw.removesuffix("%")
                digitwise = mode == "digits" or len(raw) > 1 and raw.startswith("0") and "." not in raw
                lang = "ja" if language == "ja" else "en"
                if digitwise:
                    reading = " ".join(num2words(int(char), lang=lang) for char in raw if char.isdigit())
                else:
                    reading = num2words(raw, lang=lang)
                if percent:
                    reading += " パーセント" if lang == "ja" else " percent"
                if language == "loan":
                    parts = re.findall(r"[a-z]+", reading.lower())
                    if any(part not in _LOAN for part in parts):
                        reading = ""
                    else:
                        reading = " ".join(_LOAN[part] for part in parts)
                pieces.extend((text[cursor:match.start()], reading))
                cursor = match.end()
            pieces.append(text[cursor:])
            value = "".join(pieces)
            if value.strip() and value not in {item[1] for item in results}:
                results.append((language + ":" + mode, value))
    return results


def make_number_word(text: str, pipeline, romanize, clean) -> NumberWord:
    readings, romans, labels = [], [], []
    try:
        alternatives = number_readings(text)
    except (ValueError, OverflowError, NotImplementedError) as error:
        raise G2PConversionError([text], reason="数字超出读音转换范围。") from error
    for label, value in alternatives:
        try:
            converted = pipeline.convert(clean(value), languages=["ja", "en"])
        except (G2PConversionError, KeyError):
            continue
        groups = []
        for word in converted:
            path = word.readings[0].paths[0]
            groups.extend(G2PGroup(group.script, [phone if "/" in phone else f"{word.language}/{phone}"
                                                for phone in group.phonemes]) for group in path)
        if groups:
            readings.append(G2PReading(paths=[groups]))
            romans.append(" ".join(romanize(word) for word in converted))
            labels.append(label)
    if not readings:
        raise G2PConversionError([text], reason=f"找不到数字 {text} 的读音。")
    return NumberWord(text=text, language=Language.ANY, readings=readings, romans=romans, labels=labels)


def convert_numbers(text: str, pipeline, romanize, clean) -> list[G2PWord]:
    words, cursor = [], 0
    for start, end in number_spans(text):
        if text[cursor:start].strip():
            words.extend(pipeline.convert(text[cursor:start], languages=["ja", "en"]))
        words.append(make_number_word(text[start:end], pipeline, romanize, clean))
        cursor = end
    if text[cursor:].strip():
        words.extend(pipeline.convert(text[cursor:], languages=["ja", "en"]))
    return words
