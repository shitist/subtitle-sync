"""Hard-subtitle font presets shared by the GUI and processing backends."""
from __future__ import annotations

import math
import re
from pathlib import Path


FONT_PRESETS = {
    "noto-sc": {"label": "Noto Sans 中文（默认）", "family": "Noto Sans CJK SC", "size": 17.6},
    "noto-jp": {"label": "Noto Sans 日文", "family": "Noto Sans CJK JP", "size": 17.6},
    "yahei": {"label": "微软雅黑", "family": "Microsoft YaHei", "size": 16.0},
}
DEFAULT_FONT = "noto-sc"
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts" / "noto-cjk"


def font_settings(font: str = DEFAULT_FONT, size: float | None = None) -> tuple[str, float]:
    preset = FONT_PRESETS[font]
    size = preset["size"] if size is None else float(size)
    if not math.isfinite(size) or not 6 <= size <= 96:
        raise ValueError("字幕字号须在 6 到 96 之间")
    return preset["family"], size


def apply_ass_style(content: str, font: str = DEFAULT_FONT, size: float | None = None) -> str:
    """Change Default font and size, keeping FFmpeg's SRT-to-ASS layout."""
    family, size = font_settings(font, size)
    in_styles, columns, changed = False, [], False
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        section = re.match(r"\s*\[([^]]+)\]", line)
        if section:
            in_styles = section[1].casefold() == "v4+ styles"
            columns = []
        elif in_styles and line.startswith("Format:"):
            columns = [part.strip().casefold() for part in line.split(":", 1)[1].split(",")]
        elif in_styles and columns and line.startswith("Style:"):
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            values = line.split(":", 1)[1].strip().split(",")
            if len(values) != len(columns) or values[columns.index("name")].strip() != "Default":
                continue
            values[columns.index("fontname")] = family
            values[columns.index("fontsize")] = f"{size:g}"
            lines[index] = "Style: " + ",".join(values) + ending
            changed = True
    if not changed:
        raise ValueError("字幕中未找到 Default 样式")
    return "".join(lines)


def filter_path(path: Path) -> str:
    # Escape once for the option parser and once for the filter graph.
    value = str(path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return "'" + value.replace("'", "'\\''") + "'"


def subtitle_filter(ass: Path, font: str = DEFAULT_FONT) -> str:
    result = f"subtitles=filename={filter_path(ass)}"
    if font.startswith("noto-"):
        for filename in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
            if not (FONT_DIR / filename).is_file():
                raise FileNotFoundError(f"字幕字体缺失: {FONT_DIR / filename}")
        result += f":fontsdir={filter_path(FONT_DIR)}"
    return result
