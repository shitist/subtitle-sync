#!/usr/bin/env python3
"""Subtitle Sync 的 PySide6 桌面界面。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from asr_models import LYRIC_MODEL
from subtitle_style import DEFAULT_FONT, FONT_PRESETS
from speech_catalog import ALIGN_MODELS, LANGUAGE_CODES, LANGUAGE_NAMES

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyleFactory,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).parent
    external_script = EXE_DIR / "subtitle_sync.py"
    if external_script.exists():
        APP_DIR = EXE_DIR
    elif hasattr(sys, "_MEIPASS"):
        APP_DIR = Path(sys._MEIPASS)
    else:
        APP_DIR = EXE_DIR
else:
    EXE_DIR = Path(__file__).parent
    APP_DIR = Path(__file__).parent

SYNC_SCRIPT = APP_DIR / "subtitle_sync.py"
LOG_DIR = EXE_DIR / "logs" if getattr(sys, "frozen", False) else APP_DIR / "logs"
LATEST_LOG = LOG_DIR / "latest.log"
LATEST_DIAGNOSTIC = LOG_DIR / "latest.json"

PYTHON_MODULES = (
    "faster_whisper",
    "demucs",
    "rapidfuzz",
    "rapidocr",
    "onnxruntime",
    "imageio_ffmpeg",
    "torch",
    "transformers",
    "librosa",
    "safetensors",
    "g2pflow", "fugashi", "unidic", "numba", "scipy", "soundfile",
    "omegaconf", "pydantic", "einops", "loguru", "num2words",
    "whisperx", "torchaudio", "pandas", "nltk",
)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v"}
LYRIC_EXTENSIONS = {".txt", ".lrc"}
STAGE_PROGRESS = {
    5: {1: 5, 2: 15, 3: 65, 4: 90, 5: 95},
    7: {1: 5, 2: 14, 3: 24, 4: 38, 5: 78, 6: 88, 7: 95},
    8: {1: 5, 2: 14, 3: 24, 4: 38, 5: 48, 6: 55, 7: 88, 8: 95},
}
MODEL_OPTIONS = (
    {
        "id": LYRIC_MODEL,
        "label": "歌词声学对齐（推荐）",
        "description": "MMS+TIFA路线：轻量准确，推荐使用。",
    },
    {
        "id": "large-v3",
        "label": "Large v3 — 日语歌曲、复杂音频",
        "description": "WhisperX 路线：识别后用日文原词声学对齐，再映射中文字幕。精度优先。",
    },
    {
        "id": "large-v3-turbo",
        "label": "Large v3 Turbo — 访谈、解说、一般视频",
        "description": "WhisperX 路线：识别后用日文原词声学对齐。处理速度更快。",
    },
    {
        "id": "medium",
        "label": "Medium — 清晰语音、低配置",
        "description": "WhisperX 路线：识别后用日文原词声学对齐。速度与精度折中。",
    },
    {
        "id": "small",
        "label": "Small — 快速草稿、CPU",
        "description": "WhisperX 路线：识别后用日文原词声学对齐。识别模型较小。",
    },
)

THEMES = {
    "light": {
        "window": "#f4f5f7",
        "surface": "#ffffff",
        "surface_alt": "#f8f9fa",
        "border": "#d8dce2",
        "text": "#20242a",
        "muted": "#68707b",
        "accent": "#18794e",
        "accent_hover": "#11663f",
        "accent_soft": "#e7f4ed",
        "success": "#18794e",
        "warning": "#a15c00",
        "error": "#c53929",
        "selection": "#cce9da",
    },
    "dark": {
        "window": "#111315",
        "surface": "#1a1d20",
        "surface_alt": "#202428",
        "border": "#34393f",
        "text": "#eef0f2",
        "muted": "#a0a7af",
        "accent": "#46b97a",
        "accent_hover": "#58c98b",
        "accent_soft": "#173728",
        "success": "#55c98a",
        "warning": "#e7a63a",
        "error": "#ef6a61",
        "selection": "#24563b",
    },
}


class _MessageBoxAdapter:
    """保留简洁的调用接口，也方便无界面测试替换。"""

    @staticmethod
    def showinfo(title: str, text: str) -> None:
        QMessageBox.information(None, title, text)

    @staticmethod
    def showwarning(title: str, text: str) -> None:
        QMessageBox.warning(None, title, text)

    @staticmethod
    def showerror(title: str, text: str) -> None:
        QMessageBox.critical(None, title, text)


messagebox = _MessageBoxAdapter()


def _required_modules(config: dict) -> set[str]:
    if config.get("task") == "speech":
        return {"whisperx", "faster_whisper", "torch", "torchaudio", "transformers",
                "pandas", "nltk", "soundfile", "imageio_ffmpeg"}
    if config.get("model") == LYRIC_MODEL:
        required = {
            "torch", "transformers", "g2pflow", "fugashi", "unidic", "numba",
            "librosa", "soundfile", "scipy", "omegaconf", "pydantic", "einops",
            "loguru", "onnxruntime", "safetensors", "imageio_ffmpeg", "num2words",
        }
        if config["audio_source"] == "vocals":
            required.add("demucs")
        return required
    required = {"whisperx", "faster_whisper", "torch", "torchaudio", "transformers",
                "pandas", "nltk", "soundfile", "rapidfuzz", "imageio_ffmpeg"}
    if config["audio_source"] == "vocals":
        required.add("demucs")
    if config.get("ocr_assist"):
        required.update({"rapidocr", "onnxruntime"})
    return required


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _candidate_pythons(include_align: bool = False, include_speech: bool = False) -> list[tuple[str, str, int]]:
    """列出候选解释器；返回 (路径, 来源, 优先级)。"""
    candidates: list[tuple[str, str, int]] = []
    override = os.environ.get("SUBTITLE_SYNC_PYTHON", "").strip()
    if override:
        candidates.append((override, "SUBTITLE_SYNC_PYTHON", 100))

    project_roots = list(dict.fromkeys([APP_DIR, EXE_DIR]))
    if EXE_DIR.name.lower() in {"release", "dist"}:
        project_roots.append(EXE_DIR.parent)
    for root in project_roots:
        if include_speech:
            executable = root / ".venv-whisperx" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            candidates.append((str(executable), "视频字幕专用环境", 97))
        if include_align:
            align_executable = root / ".venv-align" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            candidates.append((str(align_executable), "歌词对齐专用环境", 96))
        executable = (
            root / ".venv" / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else root / ".venv" / "bin" / "python"
        )
        candidates.append((str(executable), "项目 .venv", 90))

    if not getattr(sys, "frozen", False):
        candidates.append((sys.executable, "当前 Python", 80))

    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append((found, f"PATH:{name}", 0))

    launcher = shutil.which("py") if sys.platform == "win32" else None
    if launcher:
        try:
            result = subprocess.run(
                [launcher, "-0p"],
                capture_output=True,
                text=True,
                timeout=8,
                creationflags=_creation_flags(),
            )
            for line in result.stdout.splitlines():
                match = re.search(r"([A-Za-z]:\\.*python\.exe)\s*$", line, re.IGNORECASE)
                if match:
                    candidates.append((match.group(1), "Python Launcher", 10))
        except Exception:
            pass

    deduplicated: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for raw_path, source, priority in candidates:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue
        key = str(path.resolve()).casefold()
        if "windowsapps" in key or key in seen:
            continue
        seen.add(key)
        deduplicated.append((str(path.resolve()), source, priority))
    return deduplicated


def _probe_python(path: str, source: str, priority: int) -> dict:
    """在独立子进程中检查版本与本工具所需模块。"""
    code = f"""
import importlib.util, json, sys
names = {list(PYTHON_MODULES)!r}
modules = {{name: bool(importlib.util.find_spec(name)) for name in names}}
print(json.dumps({{'version': list(sys.version_info[:3]), 'modules': modules}}))
"""
    record = {
        "path": path,
        "source": source,
        "priority": priority,
        "version": None,
        "modules": {name: False for name in PYTHON_MODULES},
        "error": None,
    }
    try:
        result = subprocess.run(
            [path, "-c", code],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=_creation_flags(),
        )
        if result.returncode != 0:
            record["error"] = (result.stderr or result.stdout).strip()[-500:]
            return record
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        record["version"] = payload["version"]
        record["modules"] = payload["modules"]
    except Exception as error:
        record["error"] = str(error)
    return record


def _choose_python(probes: list[dict], required_modules: set[str]) -> dict | None:
    """选择满足必需模块且附加能力最完整的解释器。"""
    usable = []
    for probe in probes:
        version = probe.get("version") or [0, 0, 0]
        modules = probe.get("modules", {})
        if tuple(version[:2]) < (3, 10):
            continue
        if not all(modules.get(module, False) for module in required_modules):
            continue
        optional_score = sum(bool(modules.get(module)) for module in PYTHON_MODULES)
        usable.append((probe.get("priority", 0), optional_score, tuple(version), probe))
    return max(usable, default=None, key=lambda item: item[:3])[3] if usable else None


def _requirements_path(align: bool = False, speech: bool = False) -> Path:
    filename = "requirements-speech.txt" if speech else "requirements-align.txt" if align else "requirements.txt"
    candidates = [APP_DIR / filename, EXE_DIR.parent / filename]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / filename)
    return next((path for path in candidates if path.exists()), candidates[0])


def _find_python(required_modules: set[str]) -> tuple[dict, list[dict]]:
    """探测所有 Python，选择具备当前处理模式依赖的解释器。"""
    probes = [
        _probe_python(path, source, priority)
        for path, source, priority in _candidate_pythons(
            include_align="g2pflow" in required_modules,
            include_speech="whisperx" in required_modules,
        )
    ]
    selected = _choose_python(probes, required_modules)
    if selected:
        return selected, probes

    checked = []
    for probe in probes:
        missing = [
            module
            for module in sorted(required_modules)
            if not probe.get("modules", {}).get(module, False)
        ]
        version = ".".join(map(str, probe.get("version") or [])) or "不可用"
        checked.append(
            f"- {probe['path']}（Python {version}，缺少: {', '.join(missing) or '无'}）"
        )
    requirements = _requirements_path(align="g2pflow" in required_modules, speech="whisperx" in required_modules)
    best_path = probes[0]["path"] if probes else "python"
    raise RuntimeError(
        "没有找到可用的 Python 环境。\n\n"
        + ("已检查：\n" + "\n".join(checked) + "\n\n" if checked else "")
        + "安装所需组件：\n"
        + f'"{best_path}" -X utf8 -m pip install -r "{requirements}"'
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def classify_dropped_paths(paths: list[str]) -> dict[str, str]:
    """从拖入的路径中找出第一个视频、歌词和目录。"""
    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir() and "directory" not in result:
            result["directory"] = str(path)
        elif path.suffix.casefold() in VIDEO_EXTENSIONS and "video" not in result:
            result["video"] = str(path)
        elif path.suffix.casefold() in LYRIC_EXTENSIONS and "lyrics" not in result:
            result["lyrics"] = str(path)
    return result


def parse_runtime_line(raw_line: str) -> dict | None:
    """把后端的详细输出压缩成界面需要的阶段和简短记录。"""
    line = raw_line.strip()
    if not line or not line.strip("─═-"):
        return None

    stage_match = re.match(r"^\[(\d+)/(\d+)\]\s*(.+?)(?:\.\.\.)?$", line)
    if stage_match:
        stage = int(stage_match.group(1))
        total = int(stage_match.group(2))
        return {
            "kind": "stage",
            "stage": stage,
            "progress": STAGE_PROGRESS.get(total, {}).get(
                stage,
                round(stage / max(total, 1) * 95),
            ),
            "message": stage_match.group(3).strip(),
        }

    ocr_progress = re.match(r"^OCR 扫描进度:\s*(\d+)/(\d+)$", line)
    if ocr_progress:
        current = int(ocr_progress.group(1))
        total = max(1, int(ocr_progress.group(2)))
        return {
            "kind": "ocr_progress",
            "count": current,
            "total": total,
            "progress": 55 + round(current / total * 30),
        }

    lyric_progress = re.match(r"^歌词对齐进度:\s*(\S+)\s+(\d+)/(\d+)$", line)
    if lyric_progress:
        phase, current, total = lyric_progress.groups()
        low, high = (38, 59) if phase == "整曲定位" else (59, 62) if phase == "数字读音" else (62, 77)
        return {
            "kind": "lyric_progress", "phase": phase, "count": int(current), "total": int(total),
            "progress": low + round(int(current) / max(1, int(total)) * (high - low)),
        }

    speech_progress = re.match(r"^视频字幕进度:\s*(\S+)\s+(\d+)/(\d+)$", line)
    if speech_progress:
        phase, current, total = speech_progress.groups()
        low, high = (15, 64) if phase == "识别语音" else (65, 89)
        return {"kind": "speech_progress", "phase": phase, "count": int(current), "total": int(total),
                "progress": low + round(int(current) / max(1, int(total)) * (high - low))}

    mv_progress = re.match(r"^MV 对齐进度:\s*(\d+)/(\d+)$", line)
    if mv_progress:
        current, total = map(int, mv_progress.groups())
        return {"kind": "lyric_progress", "phase": "WhisperX 歌词对齐", "count": current, "total": total,
                "progress": 50 + round(current / max(1, total) * 27)}

    segment_match = re.match(r"^\[(\d{3})\]\s+\d+(?:\.\d+)?s\s*[→-]", line)
    if segment_match:
        return {"kind": "recognition", "count": int(segment_match.group(1))}

    if line.startswith("仍在处理") or "已运行" in line and line.startswith("⏳"):
        return None
    if line.startswith(("运行时间:", "Python:", "Python 来源:", "命令:")):
        return None

    replacements = {
        "▶": "",
        "✅": "",
        "❌": "错误：",
        "⚠️": "注意：",
        "♻️": "",
        "📄": "",
        "📊": "",
        "📁": "",
        "💡": "说明：",
        "⏳": "",
    }
    cleaned = line
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.match(r"^第\s*[\d, ]+\s*行：", cleaned):
        return {"kind": "log", "message": cleaned}

    if cleaned.startswith("处理:"):
        return {"kind": "log", "message": "文件：" + cleaned.removeprefix("处理:").strip()}
    if cleaned.startswith("[") and "words:" in cleaned:
        return None

    useful_prefixes = (
        "检测到",
        "对齐模型:",
        "复用视频字幕缓存",
        "复用 WhisperX 歌词对齐缓存",
        "WhisperX 歌词对齐:",
        "复用 Whisper",
        "复用转录",
        "已有转录",
        "转录缓存",
        "正在加载歌词对齐模型",
        "下载对齐资源:",
        "歌词对齐完成:",
        "字幕生成：",
        "复用歌词对齐缓存",
        "复用整曲声学定位缓存",
        "精对齐窗口",
        "复用 OCR",
        "已有 Whisper",
        "Whisper 识别完成",
        "Whisper 转录缓存",
        "正在加载 Whisper 模型:",
        "使用设备:",
        "文本锚点:",
        "英文锚点",
        "OCR 画面锚点:",
        "OCR 修正明显错位边界:",
        "OCR 可靠锚点不足",
        "OCR 扫描缓存",
        "OCR 辅助覆盖率:",
        "对轴模式:",
        "SRT 文件",
        "对轴报告",
        "文本匹配覆盖率:",
        "英文锚点覆盖率:",
        "完成",
        "对轴完成",
        "字幕烧入完成",
        "错误：",
        "注意：",
    )
    useful_error_fragments = (
        "No module named",
        "ModuleNotFoundError",
        "FileNotFoundError",
        "CUDA out of memory",
        "memory allocation",
        "RuntimeError:",
        "ValueError:",
        "ImportError:",
    )
    if cleaned.startswith(useful_prefixes) or any(
        fragment in cleaned for fragment in useful_error_fragments
    ):
        return {"kind": "log", "message": cleaned}
    return None


def _build_processing_command(python_exe: str, config: dict) -> list[str]:
    """根据界面中有限的选项构建核心脚本命令。"""
    command = [python_exe, "-u", str(SYNC_SCRIPT)]
    speech = config.get("task") == "speech"
    if speech:
        command += ["--task", "speech", "--language", config.get("language", "auto")]
    if config["mode"] == "single":
        command += ["--video", config["video"]]
        if not speech:
            command += ["--lyrics", config["lyrics"]]
        if config.get("output"):
            command += ["--output", config["output"]]
    else:
        command += ["--batch", "--video-dir", config["video_dir"]]
        if not speech:
            command += ["--lyrics-dir", config["lyrics_dir"]]
        if config.get("output_dir"):
            command += ["--output-dir", config["output_dir"]]

    command += [
        "--audio-source",
        "original" if speech else config["audio_source"],
        "--model",
        config["model"],
    ]
    if config.get("no_vad") and config["model"] != LYRIC_MODEL:
        command.append("--no-vad")
    if config.get("no_burn"):
        command.append("--no-burn")
    else:
        font = config.get("subtitle_font", DEFAULT_FONT)
        command += ["--subtitle-font", font, "--subtitle-size",
                    str(config.get("subtitle_size", FONT_PRESETS[font]["size"]))]
    if config.get("refresh"):
        command.append("--refresh-transcript")
    if config.get("ocr_assist") and config["model"] != LYRIC_MODEL and not speech:
        command.append("--ocr-assist")
        if config.get("refresh"):
            command.append("--refresh-ocr")
    return command


class PathLineEdit(QLineEdit):
    pathDropped = Signal(str)

    def __init__(self, *, directories: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.directories = directories
        self.setAcceptDrops(True)
        self.setClearButtonEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        local_paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        path = next(
            (
                value
                for value in local_paths
                if Path(value).is_dir() == self.directories
            ),
            "",
        )
        if path:
            self.setText(path)
            self.pathDropped.emit(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class AppSignals(QObject):
    output = Signal(str)
    status = Signal(str, str)
    message = Signal(str, str, str)
    runningChanged = Signal(bool)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle Sync")
        self.setMinimumSize(720, 560)
        self.resize(900, 720)

        self.running = False
        self._started_monotonic = 0.0
        self._theme_mode = "system"
        self._colors = THEMES["light"]
        self._settings = QSettings("SubtitleSync", "SubtitleSync")
        self._active_task = None
        self._task_states = {}
        self._signals = AppSignals()
        self._signals.output.connect(self._receive_process_output)
        self._signals.status.connect(self._apply_status)
        self._signals.message.connect(self._display_message)
        self._signals.runningChanged.connect(self._apply_running_state)

        self._build_ui()
        saved_task = self._settings.value("processing/task", "lyrics")
        self.task_combo.setCurrentIndex(max(0, self.task_combo.findData(saved_task)))
        self._task_changed()

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        saved_theme = self._settings.value("appearance/theme", "system")
        index = self.theme_combo.findData(saved_theme)
        self.theme_combo.setCurrentIndex(index if index >= 0 else 0)
        self._theme_mode = self.theme_combo.currentData()
        self._apply_theme()

        hints = QApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._on_system_theme_changed)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 15, 24, 14)
        header_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("字幕对轴")
        title.setObjectName("appTitle")
        subtitle = QLabel("视频自动字幕 · MV 歌词打轴")
        subtitle.setObjectName("mutedLabel")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        theme_label = QLabel("外观")
        theme_label.setObjectName("mutedLabel")
        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("themeCombo")
        self.theme_combo.addItem("跟随系统", "system")
        self.theme_combo.addItem("浅色", "light")
        self.theme_combo.addItem("深色", "dark")
        self.theme_combo.setMinimumWidth(112)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        header_layout.addWidget(theme_label)
        header_layout.addWidget(self.theme_combo)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        form_scroll = QScrollArea()
        form_scroll.setObjectName("formScroll")
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_scroll.setMinimumHeight(285)
        form_content = QWidget()
        form_layout = QVBoxLayout(form_content)
        form_layout.setContentsMargins(24, 18, 24, 16)
        form_layout.setSpacing(14)
        form_scroll.setWidget(form_content)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("处理模式"))
        self.task_combo = QComboBox()
        self.task_combo.addItem("MV 歌词打轴", "lyrics")
        self.task_combo.addItem("视频自动字幕", "speech")
        task_row.addWidget(self.task_combo, 1)
        form_layout.addLayout(task_row)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setDocumentMode(True)
        self.mode_tabs.addTab(self._build_single_page(), "单文件")
        self.mode_tabs.addTab(self._build_batch_page(), "批量处理")
        form_layout.addWidget(self.mode_tabs)
        form_layout.addWidget(self._build_options_card())
        form_layout.addStretch()
        splitter.addWidget(form_scroll)
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([470, 150])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        footer = QFrame()
        footer.setObjectName("footer")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(24, 11, 24, 13)
        footer_layout.setSpacing(8)

        status_row = QHBoxLayout()
        self.status_label = QLabel("等待开始")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("kind", "idle")
        self.elapsed_label = QLabel("")
        self.elapsed_label.setObjectName("mutedLabel")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(self.elapsed_label)
        footer_layout.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        footer_layout.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.open_logs_btn = QPushButton("打开日志目录")
        self.open_logs_btn.clicked.connect(self._open_log_directory)
        action_row.addWidget(self.open_logs_btn)
        action_row.addStretch()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setMinimumWidth(124)
        self.start_btn.clicked.connect(self._start)
        action_row.addWidget(self.start_btn)
        footer_layout.addLayout(action_row)
        root.addWidget(footer)
        self.task_combo.currentIndexChanged.connect(self._task_changed)

    def _build_single_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSpacing(11)

        file_row = QHBoxLayout()
        file_row.setSpacing(11)
        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.video_edit = PathLineEdit()
        self.video_edit.setPlaceholderText("选择或拖入视频文件")
        self.video_edit.pathDropped.connect(lambda value: self._on_video_selected(Path(value)))
        self.lyrics_edit = PathLineEdit()
        self.lyrics_edit.setPlaceholderText("选择或拖入 UTF-8 歌词文本")
        self.output_edit = PathLineEdit()
        self.output_edit.setPlaceholderText("不填写时自动命名")

        self._add_path_row(grid, 0, "视频", self.video_edit, self._browse_video)
        self.lyrics_row = self._add_path_row(grid, 1, "歌词", self.lyrics_edit, self._browse_lyrics)
        self._add_path_row(grid, 2, "输出", self.output_edit, self._browse_output)
        file_row.addWidget(card, 3)
        layout.addLayout(file_row)
        return page

    def _build_batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSpacing(11)

        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.video_dir_edit = PathLineEdit(directories=True)
        self.video_dir_edit.setPlaceholderText("选择或拖入视频目录")
        self.lyrics_dir_edit = PathLineEdit(directories=True)
        self.lyrics_dir_edit.setPlaceholderText("选择或拖入 UTF-8 歌词文本目录")
        self.output_dir_edit = PathLineEdit(directories=True)
        self.output_dir_edit.setPlaceholderText(self.output_edit.placeholderText())

        self._add_path_row(grid, 0, "视频", self.video_dir_edit, lambda: self._browse_dir(self.video_dir_edit))
        self.lyrics_dir_row = self._add_path_row(grid, 1, "歌词", self.lyrics_dir_edit, lambda: self._browse_dir(self.lyrics_dir_edit))
        self._add_path_row(grid, 2, "输出", self.output_dir_edit, lambda: self._browse_dir(self.output_dir_edit))
        layout.addWidget(card)
        return page

    def _add_path_row(self, grid: QGridLayout, row: int, label: str, edit: QLineEdit, callback) -> tuple:
        label_widget = QLabel(label)
        label_widget.setObjectName("fieldLabel")
        label_widget.setMinimumWidth(64)
        button = QPushButton("选择")
        button.setObjectName("pathButton")
        button.setFixedWidth(62)
        button.clicked.connect(callback)
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(edit, row, 1)
        grid.addWidget(button, row, 2)
        return label_widget, edit, button

    def _build_options_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        grid = self.options_grid = QGridLayout(card)
        grid.setContentsMargins(16, 13, 16, 13)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(9)

        heading = QLabel("处理设置")
        heading.setObjectName("sectionTitle")
        grid.addWidget(heading, 0, 0, 1, 4)

        model_label = self.model_field_label = QLabel("打轴方式")
        model_label.setObjectName("fieldLabel")
        self.model_combo = QComboBox()
        for index, option in enumerate(MODEL_OPTIONS):
            self.model_combo.addItem(option["label"], option["id"])
            self.model_combo.setItemData(
                index,
                option["description"],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.model_description = QLabel()
        self.model_description.setObjectName("mutedLabel")
        self.model_description.setWordWrap(True)
        self.model_combo.currentIndexChanged.connect(self._update_model_description)
        self._update_model_description()
        grid.addWidget(model_label, 1, 0)
        grid.addWidget(self.model_combo, 1, 1, 1, 3)
        grid.addWidget(self.model_description, 2, 1, 1, 3)

        source_label = self.source_field_label = QLabel("识别音源")
        source_label.setObjectName("fieldLabel")
        self.audio_source_combo = QComboBox()
        self.audio_source_combo.addItem("分离人声（较慢）", "vocals")
        self.audio_source_combo.addItem("原始音频（较快）", "original")
        self.audio_source_combo.setToolTip(
            "分离人声：提取人声后打轴。\n"
            "原始音频：使用视频中的混合音频打轴。"
        )
        grid.addWidget(source_label, 3, 0)
        grid.addWidget(self.audio_source_combo, 3, 1)

        self.no_burn_cb = QCheckBox("仅生成 SRT")
        self.no_burn_cb.setToolTip("生成 SRT 和对轴报告。")
        self.refresh_cb = QCheckBox("重新识别")
        self.refresh_cb.setToolTip("忽略已有转录缓存，重新提取音频并识别。")
        self.ocr_assist_cb = QCheckBox("画面歌词辅助")
        self.ocr_assist_cb.setToolTip(
            "识别画面歌词并校正字幕时间。"
        )
        self.no_vad_cb = QCheckBox("禁用 VAD")
        self.no_vad_cb.setToolTip("保留所有音频区间。")
        self.option_checks = QWidget()
        checks_layout = QHBoxLayout(self.option_checks)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.setSpacing(18)
        for checkbox in (self.no_vad_cb, self.ocr_assist_cb, self.no_burn_cb, self.refresh_cb):
            checks_layout.addWidget(checkbox)
        checks_layout.addStretch()
        grid.addWidget(self.option_checks, 4, 1, 1, 3)

        self.subtitle_font_combo = QComboBox()
        installed = set(QFontDatabase.families())
        for key, preset in FONT_PRESETS.items():
            if key != "yahei" or preset["family"] in installed:
                self.subtitle_font_combo.addItem(preset["label"], key)
        saved_font = self._settings.value("subtitles/font", DEFAULT_FONT)
        saved_index = self.subtitle_font_combo.findData(saved_font)
        self.subtitle_font_combo.setCurrentIndex(max(0, saved_index))
        self.subtitle_size_spin = QDoubleSpinBox()
        self.subtitle_size_spin.setRange(6, 96)
        self.subtitle_size_spin.setDecimals(1)
        self.subtitle_size_spin.setSingleStep(0.5)
        default_size = FONT_PRESETS[self.subtitle_font_combo.currentData()]["size"]
        self.subtitle_size_spin.setValue(float(self._settings.value("subtitles/size", default_size)))
        self.subtitle_size_spin.setToolTip("ASS 相对字号，与 FFmpeg 工具箱一致。")
        self.font_field_label = QLabel("字幕字体")
        self.size_field_label = QLabel("字号")
        grid.addWidget(self.font_field_label, 5, 0)
        grid.addWidget(self.subtitle_font_combo, 5, 1)
        grid.addWidget(self.size_field_label, 5, 2)
        grid.addWidget(self.subtitle_size_spin, 5, 3)
        self.language_label = QLabel("识别语言")
        self.language_combo = QComboBox()
        self.language_combo.addItem("自动检测", "auto")
        preferred = ["zh", "ja", "en", "ko", "yue"]
        for code in preferred + [c for c in LANGUAGE_CODES if c not in preferred]:
            self.language_combo.addItem(f"{LANGUAGE_NAMES.get(code, code)} · {code}", code)
        self.language_combo.currentIndexChanged.connect(self._update_language_hint)
        grid.addWidget(self.language_label, 6, 0)
        grid.addWidget(self.language_combo, 6, 1, 1, 3)
        self.subtitle_font_combo.currentIndexChanged.connect(self._subtitle_font_changed)
        self.subtitle_size_spin.valueChanged.connect(self._remember_subtitle_style)
        self.no_burn_cb.toggled.connect(self._update_subtitle_controls)
        self._update_subtitle_controls()
        grid.setColumnStretch(1, 1)
        self._update_model_description()
        return card

    def _subtitle_font_changed(self) -> None:
        self.subtitle_size_spin.setValue(FONT_PRESETS[self.subtitle_font_combo.currentData()]["size"])
        self._remember_subtitle_style()

    def _remember_subtitle_style(self) -> None:
        self._settings.setValue("subtitles/font", self.subtitle_font_combo.currentData())
        self._settings.setValue("subtitles/size", self.subtitle_size_spin.value())

    def _update_subtitle_controls(self) -> None:
        enabled = not self.running and not self.no_burn_cb.isChecked()
        self.subtitle_font_combo.setEnabled(enabled)
        self.subtitle_size_spin.setEnabled(enabled)

    def _update_model_description(self) -> None:
        if not hasattr(self, "model_description"):
            return
        model_id = self.model_combo.currentData()
        option = next(item for item in MODEL_OPTIONS if item["id"] == model_id)
        description = option["description"]
        if hasattr(self, "task_combo") and self.task_combo.currentData() == "speech":
            description = {"large-v3": "识别准确度优先。", "large-v3-turbo": "处理更快，适合一般视频。",
                           "medium": "速度与准确度折中。", "small": "资源占用较低，适合清晰语音。"}[model_id]
        self.model_description.setText(description)
        self._update_engine_controls()

    def _update_engine_controls(self) -> None:
        if not hasattr(self, "no_vad_cb"):
            return
        acoustic = self.model_combo.currentData() == LYRIC_MODEL
        speech = self.task_combo.currentData() == "speech"
        self.no_vad_cb.setEnabled(not self.running and not acoustic)
        self.ocr_assist_cb.setEnabled(not self.running and not acoustic and not speech)
        self.ocr_assist_cb.setVisible(not speech)
        self.no_vad_cb.setText("启用 VAD" if speech else "禁用 VAD")
        self.refresh_cb.setText("重新对齐" if acoustic else "重新识别")
        self.no_vad_cb.setToolTip("歌词声学对齐始终保留整曲音频，不使用 VAD 裁切。" if acoustic else "保留所有音频区间。")
        self.ocr_assist_cb.setToolTip("歌词声学对齐使用音频与日文原词。" if acoustic else "用画面中已有的歌词校正局部漂移。")

    def _update_language_hint(self) -> None:
        language = self.language_combo.currentData()
        self.language_combo.setToolTip(
            "该语言没有配置声学对齐模型。" if language != "auto" and language not in ALIGN_MODELS
            else "选择视频中的主要语言，或使用自动检测。"
        )

    def _task_changed(self) -> None:
        task = self.task_combo.currentData()
        if self._active_task == task:
            return
        if self._active_task is not None:
            state = {"model": self.model_combo.currentData(), "audio_source": self.audio_source_combo.currentData(),
                     "vad_checked": self.no_vad_cb.isChecked(), "language": self.language_combo.currentData()}
            self._task_states[self._active_task] = state
            self._settings.setValue(f"processing/{self._active_task}", state)
        speech = task == "speech"
        state = self._task_states.get(task) or self._settings.value(f"processing/{task}", {}) or {}
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for option in MODEL_OPTIONS:
            if speech and option["id"] == LYRIC_MODEL:
                continue
            name = option["label"].split(" — ")[0]
            label = name if speech else option["label"] if option["id"] == LYRIC_MODEL else f"WhisperX · {name}"
            self.model_combo.addItem(label, option["id"])
        desired = state.get("model", "large-v3" if speech else LYRIC_MODEL)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(desired)))
        self.model_combo.blockSignals(False)
        source = state.get("audio_source", "original" if speech else "vocals")
        self.audio_source_combo.setCurrentIndex(max(0, self.audio_source_combo.findData(source)))
        self.no_vad_cb.setChecked(state.get("vad_checked", speech))
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(state.get("language", "auto"))))
        for widget in (*self.lyrics_row, *self.lyrics_dir_row, self.source_field_label, self.audio_source_combo):
            widget.setVisible(not speech)
        for widget in (self.language_label, self.language_combo):
            widget.setVisible(speech)
        # Reuse the same controls while keeping each mode's field order compact.
        placements = (
            (self.language_label, 3 if speech else 6, 0, 1),
            (self.language_combo, 3 if speech else 6, 1, 3),
            (self.option_checks, 4, 1, 3),
            (self.font_field_label, 5, 0, 1),
            (self.subtitle_font_combo, 5, 1, 1),
            (self.size_field_label, 5, 2, 1),
            (self.subtitle_size_spin, 5, 3, 1),
        )
        for widget, row, column, span in placements:
            self.options_grid.removeWidget(widget)
            self.options_grid.addWidget(widget, row, column, 1, span)
        self.model_field_label.setText("识别模型" if speech else "打轴方式")
        self._active_task = task
        self._settings.setValue("processing/task", task)
        self._update_model_description()
        self._update_language_hint()

    def closeEvent(self, event) -> None:
        task = self.task_combo.currentData()
        self._settings.setValue(f"processing/{task}", {
            "model": self.model_combo.currentData(), "audio_source": self.audio_source_combo.currentData(),
            "vad_checked": self.no_vad_cb.isChecked(), "language": self.language_combo.currentData(),
        })
        super().closeEvent(event)

    def _build_log_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("logPanel")
        panel.setMinimumHeight(115)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 10, 24, 12)
        layout.setSpacing(7)

        header = QHBoxLayout()
        title = QLabel("运行记录")
        title.setObjectName("sectionTitle")
        clear_btn = QPushButton("清空")
        clear_btn.setObjectName("quietButton")
        clear_btn.clicked.connect(self._clear_log)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self.log_text = QPlainTextEdit()
        self.log_text.setObjectName("logText")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(500)
        self.log_text.setPlaceholderText("处理阶段和输出文件会显示在这里")
        layout.addWidget(self.log_text, 1)
        return panel

    def _on_theme_selected(self) -> None:
        if not hasattr(self, "theme_combo"):
            return
        self._theme_mode = self.theme_combo.currentData() or "system"
        self._settings.setValue("appearance/theme", self._theme_mode)
        self._apply_theme()

    def _on_system_theme_changed(self, _scheme) -> None:
        if self._theme_mode == "system":
            self._apply_theme()

    def _effective_theme(self) -> str:
        if self._theme_mode in {"light", "dark"}:
            return self._theme_mode
        scheme = QApplication.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"

    def _apply_theme(self) -> None:
        application = QApplication.instance()
        if application is None:
            return
        colors = THEMES[self._effective_theme()]
        self._colors = colors

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface_alt"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["selection"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["text"]))
        application.setPalette(palette)
        application.setStyleSheet(self._style_sheet(colors))

    @staticmethod
    def _style_sheet(c: dict[str, str]) -> str:
        return f"""
            * {{
                font-family: "Microsoft YaHei UI", "Segoe UI";
                font-size: 13px;
                color: {c['text']};
            }}
            QMainWindow, QWidget#central, QScrollArea#formScroll,
            QScrollArea#formScroll > QWidget > QWidget {{
                background: {c['window']};
            }}
            QFrame#header, QFrame#footer {{
                background: {c['surface']};
                border: 0;
            }}
            QFrame#header {{ border-bottom: 1px solid {c['border']}; }}
            QFrame#footer {{ border-top: 1px solid {c['border']}; }}
            QLabel#appTitle {{ font-size: 20px; font-weight: 650; }}
            QLabel#sectionTitle, QLabel#dropTitle {{ font-weight: 650; }}
            QLabel#mutedLabel, QLabel#fieldLabel {{ color: {c['muted']}; }}
            QLabel#statusLabel {{ font-weight: 600; }}
            QLabel#statusLabel[kind="success"] {{ color: {c['success']}; }}
            QLabel#statusLabel[kind="warning"] {{ color: {c['warning']}; }}
            QLabel#statusLabel[kind="error"] {{ color: {c['error']}; }}
            QFrame#card {{
                background: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 9px;
            }}
            QFrame#dropArea {{
                background: {c['surface_alt']};
                border: 1px dashed {c['border']};
                border-radius: 9px;
            }}
            QFrame#dropArea[dragActive="true"] {{
                background: {c['accent_soft']};
                border: 1px solid {c['accent']};
            }}
            QFrame#logPanel {{
                background: {c['surface']};
                border-top: 1px solid {c['border']};
            }}
            QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit {{
                background: {c['surface_alt']};
                border: 1px solid {c['border']};
                border-radius: 6px;
                padding: 7px 9px;
                selection-background-color: {c['selection']};
            }}
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
                border-color: {c['accent']};
            }}
            QComboBox {{ padding-right: 24px; }}
            QComboBox::drop-down {{ border: 0; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {c['surface']};
                border: 1px solid {c['border']};
                selection-background-color: {c['selection']};
                outline: 0;
            }}
            QPlainTextEdit#logText {{
                font-family: "Cascadia Mono", "Consolas";
                font-size: 12px;
                padding: 8px;
            }}
            QPushButton {{
                background: {c['surface_alt']};
                border: 1px solid {c['border']};
                border-radius: 6px;
                padding: 7px 13px;
            }}
            QPushButton:hover {{ border-color: {c['accent']}; }}
            QPushButton:pressed {{ background: {c['accent_soft']}; }}
            QPushButton:disabled {{ color: {c['muted']}; border-color: {c['border']}; }}
            QPushButton#primaryButton {{
                color: white;
                background: {c['accent']};
                border-color: {c['accent']};
                font-weight: 650;
                padding: 8px 18px;
            }}
            QPushButton#primaryButton:hover {{ background: {c['accent_hover']}; }}
            QPushButton#quietButton {{
                color: {c['muted']};
                background: transparent;
                border: 0;
                padding: 4px 7px;
            }}
            QTabWidget::pane {{ border: 0; }}
            QTabBar::tab {{
                color: {c['muted']};
                background: transparent;
                border: 0;
                border-bottom: 2px solid transparent;
                padding: 8px 18px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                color: {c['text']};
                border-bottom-color: {c['accent']};
                font-weight: 650;
            }}
            QCheckBox {{ spacing: 7px; }}
            QCheckBox::indicator {{ width: 16px; height: 16px; }}
            QCheckBox::indicator:unchecked {{
                background: {c['surface_alt']};
                border: 1px solid {c['border']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background: {c['accent']};
                border: 1px solid {c['accent']};
                border-radius: 3px;
            }}
            QProgressBar {{
                background: {c['border']};
                border: 0;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
            QSplitter::handle {{ background: {c['border']}; height: 1px; }}
            QScrollBar:vertical {{
                background: transparent; width: 10px; margin: 1px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['border']}; min-height: 28px; border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "视频文件 (*.mp4 *.mkv *.avi *.mov *.flv *.webm *.m4v)",
        )
        if path:
            self.video_edit.setText(path)
            self._on_video_selected(Path(path))

    def _browse_lyrics(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择歌词", "", "歌词文本 (*.txt *.lrc)")
        if path:
            self.lyrics_edit.setText(path)

    def _browse_output(self) -> None:
        initial = self.output_edit.text().strip()
        path, _ = QFileDialog.getSaveFileName(self, "输出文件", initial, "MP4 视频 (*.mp4)")
        if path:
            self.output_edit.setText(path)

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", target.text().strip())
        if path:
            target.setText(path)

    def _on_files_dropped(self, paths: list[str]) -> None:
        classified = classify_dropped_paths(paths)
        if video := classified.get("video"):
            self.video_edit.setText(video)
            self._on_video_selected(Path(video))
        if lyrics := classified.get("lyrics"):
            self.lyrics_edit.setText(lyrics)
        if not {"video", "lyrics"}.intersection(classified):
            self._append_log("未找到支持的视频或歌词文件")

    def _on_video_selected(self, path: Path) -> None:
        self.output_edit.setText(str(path.parent / f"{path.stem}_subtitled{path.suffix}"))
        candidate = path.with_suffix(".txt")
        if candidate.exists() and not self.lyrics_edit.text().strip():
            self.lyrics_edit.setText(str(candidate))

    def _start(self) -> None:
        if self.running:
            return
        if not SYNC_SCRIPT.exists():
            messagebox.showerror("缺少核心脚本", f"未找到：\n{SYNC_SCRIPT}")
            return

        mode = "single" if self.mode_tabs.currentIndex() == 0 else "batch"
        speech = self.task_combo.currentData() == "speech"
        config = {
            "mode": mode,
            "task": self.task_combo.currentData(),
            "language": self.language_combo.currentData(),
            "audio_source": "original" if speech else self.audio_source_combo.currentData(),
            "model": self.model_combo.currentData(),
            "no_vad": (not self.no_vad_cb.isChecked() if speech else self.no_vad_cb.isChecked()) and self.model_combo.currentData() != LYRIC_MODEL,
            "no_burn": self.no_burn_cb.isChecked(),
            "refresh": self.refresh_cb.isChecked(),
            "ocr_assist": self.ocr_assist_cb.isChecked() and self.model_combo.currentData() != LYRIC_MODEL and not speech,
            "subtitle_font": self.subtitle_font_combo.currentData(),
            "subtitle_size": self.subtitle_size_spin.value(),
        }

        if mode == "single":
            video = self.video_edit.text().strip()
            lyrics = self.lyrics_edit.text().strip()
            if not video:
                messagebox.showwarning("缺少视频", "请选择视频文件。")
                return
            if not Path(video).is_file():
                messagebox.showwarning("视频不存在", video)
                return
            if not speech and not lyrics:
                messagebox.showwarning("缺少歌词", "请选择歌词文件。")
                return
            if not speech and not Path(lyrics).is_file():
                messagebox.showwarning("歌词不存在", lyrics)
                return
            config.update(
                video=video,
                lyrics=lyrics,
                output=self.output_edit.text().strip(),
            )
        else:
            video_dir = self.video_dir_edit.text().strip()
            lyrics_dir = self.lyrics_dir_edit.text().strip()
            if not video_dir or not Path(video_dir).is_dir():
                messagebox.showwarning("视频目录无效", "请选择包含视频的目录。")
                return
            if not speech and (not lyrics_dir or not Path(lyrics_dir).is_dir()):
                messagebox.showwarning("歌词目录无效", "请选择包含歌词的目录。")
                return
            config.update(
                video_dir=video_dir,
                lyrics_dir=lyrics_dir,
                output_dir=self.output_dir_edit.text().strip(),
            )

        self.running = True
        self._apply_running_state(True)
        self.progress_bar.setValue(2)
        self._apply_status("检查运行环境", "working")
        self._append_log("开始处理")
        self._started_monotonic = time.monotonic()
        self.elapsed_label.setText("已运行 00:00")
        self._elapsed_timer.start()

        threading.Thread(
            target=self._prepare_and_run,
            args=(config,),
            daemon=True,
        ).start()

    def _prepare_and_run(self, config: dict) -> None:
        required_modules = _required_modules(config)

        try:
            selected_python, python_probes = _find_python(required_modules)
        except RuntimeError as error:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().astimezone().isoformat()
            LATEST_LOG.write_text(
                f"[{timestamp}] Python 环境检查失败\n{error}\n",
                encoding="utf-8",
            )
            _write_json(
                LATEST_DIAGNOSTIC,
                {
                    "started_at": timestamp,
                    "status": "preflight_failed",
                    "model": config["model"],
                    "required_modules": sorted(required_modules),
                    "error": str(error),
                    "log_path": str(LATEST_LOG),
                },
            )
            self._log(f"错误：运行环境检查失败\n{error}\n")
            self._set_status("运行环境不可用", "error")
            self._show_message("error", "运行环境不可用", str(error))
            self._processing_finished()
            return

        python_exe = selected_python["path"]
        cmd = _build_processing_command(python_exe, config)

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        context = {
            "run_id": run_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "status": "starting",
            "mode": config["mode"],
            "task": config.get("task", "lyrics"),
            "language": config.get("language", "ja"),
            "audio_source": config["audio_source"],
            "model": config["model"],
            "ocr_assist": config.get("ocr_assist", False),
            "subtitle_font": config.get("subtitle_font", DEFAULT_FONT),
            "subtitle_size": config.get("subtitle_size"),
            "required_modules": sorted(required_modules),
            "selected_python": selected_python,
            "python_candidates": python_probes,
            "command": cmd,
            "log_path": str(LATEST_LOG),
            "run_log_path": str(LOG_DIR / f"run-{run_id}.log"),
        }
        self._run_cmd(cmd, context)

    def _run_cmd(self, cmd: list[str], run_context: dict) -> None:
        no_burn = "--no-burn" in cmd
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        run_log_path = Path(run_context["run_log_path"])
        exit_code = None
        status = "failed"
        failure_detail = ""

        def update_diagnostics(**changes) -> None:
            run_context.update(changes)
            _write_json(LATEST_DIAGNOSTIC, run_context)
            _write_json(run_log_path.with_suffix(".json"), run_context)

        try:
            with (
                open(LATEST_LOG, "w", encoding="utf-8") as latest_log,
                open(run_log_path, "w", encoding="utf-8") as run_log,
            ):
                emit_lock = threading.Lock()

                def emit(text: str) -> None:
                    with emit_lock:
                        latest_log.write(text)
                        latest_log.flush()
                        run_log.write(text)
                        run_log.flush()
                        self._log(text)

                header = (
                    f"运行时间: {run_context['started_at']}\n"
                    f"Python: {run_context['selected_python']['path']}\n"
                    f"Python 来源: {run_context['selected_python']['source']}\n"
                    f"识别模型: {run_context['model']}\n"
                    f"识别音源: {run_context['audio_source']}\n"
                    f"画面歌词辅助: {'是' if run_context.get('ocr_assist') else '否'}\n"
                    f"命令: {subprocess.list2cmdline(cmd)}\n"
                    f"{'─' * 60}\n"
                )
                emit(header)
                update_diagnostics(status="running")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=_creation_flags(),
                )
                if process.stdout is not None:
                    with process.stdout:
                        for line in process.stdout:
                            if re.match(r"^\w*(?:Error|Exception):", line.strip()):
                                failure_detail = line.strip()
                            emit(line)
                exit_code = process.wait()

                if exit_code == 0:
                    status = "completed"
                    emit("\n处理完成\n")
                    self._set_status("处理完成", "success")
                    message = "SRT 和对轴报告已生成。" if no_burn else "字幕视频已生成。"
                    self._show_message("info", "处理完成", message)
                else:
                    status = "failed"
                    emit(f"\n错误：处理程序退出，返回码 {exit_code}\n")
                    self._set_status("处理失败", "error")
                    self._show_message(
                        "error",
                        "处理失败",
                        failure_detail or f"处理程序退出，返回码 {exit_code}。",
                    )
        except Exception as error:
            status = "launch_failed"
            error_text = f"\n错误：无法启动处理程序：{error}\n"
            self._log(error_text)
            try:
                with open(LATEST_LOG, "a", encoding="utf-8") as handle:
                    handle.write(error_text)
                with open(run_log_path, "a", encoding="utf-8") as handle:
                    handle.write(error_text)
            except OSError:
                pass
            run_context["error"] = str(error)
            self._set_status("无法启动", "error")
            self._show_message("error", "无法启动", str(error))
        finally:
            try:
                update_diagnostics(
                    status=status,
                    exit_code=exit_code,
                    finished_at=datetime.now().astimezone().isoformat(),
                )
            finally:
                self._processing_finished()

    def _log(self, text: str) -> None:
        self._signals.output.emit(text)

    def _set_status(self, text: str, kind: str) -> None:
        self._signals.status.emit(text, kind)

    def _show_message(self, kind: str, title: str, text: str) -> None:
        self._signals.message.emit(kind, title, text)

    def _processing_finished(self) -> None:
        self.running = False
        self._signals.runningChanged.emit(False)

    def _receive_process_output(self, text: str) -> None:
        for line in text.splitlines():
            event = parse_runtime_line(line)
            if event is None:
                continue
            if event["kind"] == "stage":
                self.progress_bar.setValue(event["progress"])
                self._apply_status(event["message"], "working")
                self._append_log(event["message"])
            elif event["kind"] == "recognition":
                self._apply_status(f"语音识别 · {event['count']} 个片段", "working")
            elif event["kind"] == "ocr_progress":
                self.progress_bar.setValue(event["progress"])
                self._apply_status(
                    f"扫描画面歌词 · {event['count']}/{event['total']}",
                    "working",
                )
            elif event["kind"] in {"lyric_progress", "speech_progress"}:
                self.progress_bar.setValue(event["progress"])
                self._apply_status(f"{event['phase']} · {event['count']}/{event['total']}", "working")
            elif event["kind"] == "log":
                self._append_log(event["message"])

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M")
        self.log_text.appendPlainText(f"{timestamp}  {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_log(self) -> None:
        self.log_text.clear()

    def _apply_status(self, text: str, kind: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("kind", kind)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        if kind == "success":
            self.progress_bar.setValue(100)

    def _apply_running_state(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.start_btn.setText("处理中" if running else "开始处理")
        self.mode_tabs.setEnabled(not running)
        self.task_combo.setEnabled(not running)
        self.language_combo.setEnabled(not running)
        self.audio_source_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.no_burn_cb.setEnabled(not running)
        self.refresh_cb.setEnabled(not running)
        self.ocr_assist_cb.setEnabled(not running)
        self.no_vad_cb.setEnabled(not running)
        self._update_engine_controls()
        self._update_subtitle_controls()
        if not running:
            self._elapsed_timer.stop()

    def _display_message(self, kind: str, title: str, text: str) -> None:
        if kind == "error":
            messagebox.showerror(title, text)
        elif kind == "warning":
            messagebox.showwarning(title, text)
        else:
            messagebox.showinfo(title, text)

    def _update_elapsed(self) -> None:
        elapsed = max(0, int(time.monotonic() - self._started_monotonic))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
        self.elapsed_label.setText(f"已运行 {formatted}")

    def _open_log_directory(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(LOG_DIR)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(LOG_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(LOG_DIR)])


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("Subtitle Sync")
    application.setOrganizationName("SubtitleSync")
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        application.setStyle(fusion)
    application.setFont(QFont("Microsoft YaHei UI", 9))
    window = App()
    if "--smoke-test" in sys.argv:
        application.processEvents()
        return 0
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
