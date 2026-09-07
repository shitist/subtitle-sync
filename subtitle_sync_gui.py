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

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QSizePolicy,
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
)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".m4v"}
LYRIC_EXTENSIONS = {".txt", ".lrc"}
STAGE_PROGRESS = {
    7: {1: 5, 2: 14, 3: 24, 4: 38, 5: 78, 6: 88, 7: 95},
    8: {1: 5, 2: 14, 3: 24, 4: 38, 5: 48, 6: 55, 7: 88, 8: 95},
}
MODEL_OPTIONS = (
    {
        "id": "large-v3",
        "label": "Large v3 — 日语歌曲、复杂音频（默认）",
        "description": "日语歌曲、现场或背景声复杂的视频；精度优先，处理最慢。",
    },
    {
        "id": "large-v3-turbo",
        "label": "Large v3 Turbo — 访谈、解说、一般视频",
        "description": "访谈、解说和一般视频；接近 Large v3，处理速度更快。",
    },
    {
        "id": "medium",
        "label": "Medium — 清晰语音、低配置",
        "description": "语音较清晰或硬件资源有限的视频；速度与精度折中。",
    },
    {
        "id": "small",
        "label": "Small — 快速草稿、CPU",
        "description": "清晰单人语音、快速草稿或 CPU；不建议用于歌曲和嘈杂音频。",
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


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _candidate_pythons() -> list[tuple[str, str, int]]:
    """列出候选解释器；返回 (路径, 来源, 优先级)。"""
    candidates: list[tuple[str, str, int]] = []
    override = os.environ.get("SUBTITLE_SYNC_PYTHON", "").strip()
    if override:
        candidates.append((override, "SUBTITLE_SYNC_PYTHON", 100))

    project_roots = [APP_DIR]
    if APP_DIR.name.lower() == "release":
        project_roots.append(APP_DIR.parent)
    for root in project_roots:
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
    code = (
        "import importlib.util,json,sys;"
        f"names={list(PYTHON_MODULES)!r};"
        "print(json.dumps({'version':list(sys.version_info[:3]),"
        "'modules':{name:bool(importlib.util.find_spec(name)) for name in names}}))"
    )
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


def _requirements_path() -> Path:
    candidates = [APP_DIR / "requirements.txt", EXE_DIR.parent / "requirements.txt"]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "requirements.txt")
    return next((path for path in candidates if path.exists()), candidates[0])


def _find_python(required_modules: set[str]) -> tuple[dict, list[dict]]:
    """探测所有 Python，选择具备当前处理模式依赖的解释器。"""
    probes = [
        _probe_python(path, source, priority)
        for path, source, priority in _candidate_pythons()
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
    requirements = _requirements_path()
    best_path = probes[0]["path"] if probes else "python"
    raise RuntimeError(
        "没有找到可用的 Python 环境。\n\n"
        + ("已检查：\n" + "\n".join(checked) + "\n\n" if checked else "")
        + "安装所需组件：\n"
        + f'"{best_path}" -m pip install -r "{requirements}"'
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

    if cleaned.startswith("处理:"):
        return {"kind": "log", "message": "文件：" + cleaned.removeprefix("处理:").strip()}
    if cleaned.startswith("[") and "words:" in cleaned:
        return None

    useful_prefixes = (
        "检测到",
        "复用 Whisper",
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
    )
    if cleaned.startswith(useful_prefixes) or any(
        fragment in cleaned for fragment in useful_error_fragments
    ):
        return {"kind": "log", "message": cleaned}
    return None


def _build_processing_command(python_exe: str, config: dict) -> list[str]:
    """根据界面中有限的选项构建核心脚本命令。"""
    command = [python_exe, "-u", str(SYNC_SCRIPT)]
    if config["mode"] == "single":
        command += ["--video", config["video"], "--lyrics", config["lyrics"]]
        if config.get("output"):
            command += ["--output", config["output"]]
    else:
        command += [
            "--batch",
            "--video-dir",
            config["video_dir"],
            "--lyrics-dir",
            config["lyrics_dir"],
        ]
        if config.get("output_dir"):
            command += ["--output-dir", config["output_dir"]]

    command += [
        "--audio-source",
        config["audio_source"],
        "--model",
        config["model"],
    ]
    if config.get("no_vad"):
        command.append("--no-vad")
    if config.get("no_burn"):
        command.append("--no-burn")
    if config.get("refresh"):
        command.append("--refresh-transcript")
    if config.get("ocr_assist"):
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


class DropArea(QFrame):
    filesDropped = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("dropArea")
        self.setProperty("dragActive", False)
        self.setAcceptDrops(True)
        self.setMinimumHeight(82)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setSpacing(3)
        title = QLabel("将视频和歌词拖到这里")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note = QLabel("也可以在下方分别选择文件")
        note.setObjectName("mutedLabel")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(note)

    def _set_drag_active(self, active: bool) -> None:
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self._set_drag_active(True)
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event) -> None:
        self._set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_drag_active(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()


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
        self._signals = AppSignals()
        self._signals.output.connect(self._receive_process_output)
        self._signals.status.connect(self._apply_status)
        self._signals.message.connect(self._display_message)
        self._signals.runningChanged.connect(self._apply_running_state)

        self._build_ui()

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
        subtitle = QLabel("日语歌曲中文字幕时间轴")
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

    def _build_single_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSpacing(11)

        file_row = QHBoxLayout()
        file_row.setSpacing(11)
        drop_area = DropArea()
        drop_area.filesDropped.connect(self._on_files_dropped)
        drop_area.setMinimumWidth(205)
        drop_area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        file_row.addWidget(drop_area, 1)

        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.video_edit = PathLineEdit()
        self.video_edit.setPlaceholderText("选择视频文件")
        self.video_edit.pathDropped.connect(lambda value: self._on_video_selected(Path(value)))
        self.lyrics_edit = PathLineEdit()
        self.lyrics_edit.setPlaceholderText("选择 UTF-8 歌词文本")
        self.output_edit = PathLineEdit()
        self.output_edit.setPlaceholderText("不填写时自动命名")

        self._add_path_row(grid, 0, "视频", self.video_edit, self._browse_video)
        self._add_path_row(grid, 1, "歌词", self.lyrics_edit, self._browse_lyrics)
        self._add_path_row(grid, 2, "输出", self.output_edit, self._browse_output)
        file_row.addWidget(card, 3)
        layout.addLayout(file_row)
        return page

    def _build_batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSpacing(10)

        note = QLabel("视频与歌词按文件名对应。目录也可以直接拖入输入框。")
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        self.video_dir_edit = PathLineEdit(directories=True)
        self.video_dir_edit.setPlaceholderText("包含视频的目录")
        self.lyrics_dir_edit = PathLineEdit(directories=True)
        self.lyrics_dir_edit.setPlaceholderText("包含歌词的目录")
        self.output_dir_edit = PathLineEdit(directories=True)
        self.output_dir_edit.setPlaceholderText("不填写时使用视频目录下的 output")

        self._add_path_row(grid, 0, "视频目录", self.video_dir_edit, lambda: self._browse_dir(self.video_dir_edit))
        self._add_path_row(grid, 1, "歌词目录", self.lyrics_dir_edit, lambda: self._browse_dir(self.lyrics_dir_edit))
        self._add_path_row(grid, 2, "输出目录", self.output_dir_edit, lambda: self._browse_dir(self.output_dir_edit))
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _add_path_row(self, grid: QGridLayout, row: int, label: str, edit: QLineEdit, callback) -> None:
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

    def _build_options_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 13, 16, 13)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(9)

        heading = QLabel("处理设置")
        heading.setObjectName("sectionTitle")
        grid.addWidget(heading, 0, 0, 1, 4)

        model_label = QLabel("语音模型")
        model_label.setObjectName("fieldLabel")
        self.model_combo = QComboBox()
        for index, option in enumerate(MODEL_OPTIONS):
            self.model_combo.addItem(option["label"], option["id"])
            self.model_combo.setItemData(
                index,
                option["description"],
                Qt.ItemDataRole.ToolTipRole,
            )
        self.model_combo.setToolTip("首次使用所选模型时会自动下载，之后从本机缓存加载。")
        self.model_description = QLabel()
        self.model_description.setObjectName("mutedLabel")
        self.model_description.setWordWrap(True)
        self.model_combo.currentIndexChanged.connect(self._update_model_description)
        self._update_model_description()
        grid.addWidget(model_label, 1, 0)
        grid.addWidget(self.model_combo, 1, 1, 1, 3)
        grid.addWidget(self.model_description, 2, 1, 1, 3)

        source_label = QLabel("识别音源")
        source_label.setObjectName("fieldLabel")
        self.audio_source_combo = QComboBox()
        self.audio_source_combo.addItem("分离人声（较慢）", "vocals")
        self.audio_source_combo.addItem("原始音频（较快）", "original")
        self.audio_source_combo.setToolTip(
            "分离人声会先去掉大部分伴奏，通常更适合歌曲识别，但处理时间更长。\n"
            "原始音频会直接识别视频中的混音。"
        )
        grid.addWidget(source_label, 3, 0)
        grid.addWidget(self.audio_source_combo, 3, 1)

        self.no_burn_cb = QCheckBox("仅生成 SRT")
        self.no_burn_cb.setToolTip("生成 SRT 和对轴报告，不重新编码视频。识别步骤仍会运行。")
        self.refresh_cb = QCheckBox("重新识别")
        self.refresh_cb.setToolTip("忽略已有 Whisper 缓存，重新提取音频并识别。")
        self.ocr_assist_cb = QCheckBox("画面歌词辅助")
        self.ocr_assist_cb.setToolTip(
            "视频画面本身带有歌词或硬字幕时，可用 OCR 校正局部漂移。\n"
            "会增加数分钟处理时间；没有画面文字时请勿开启。"
        )
        self.no_vad_cb = QCheckBox("禁用 VAD")
        self.no_vad_cb.setToolTip("保留所有音频区间。通常无需开启。")
        grid.addWidget(self.no_burn_cb, 3, 2)
        grid.addWidget(self.refresh_cb, 3, 3)
        grid.addWidget(self.ocr_assist_cb, 4, 1)
        grid.addWidget(self.no_vad_cb, 4, 2)
        grid.setColumnStretch(1, 1)
        return card

    def _update_model_description(self) -> None:
        if not hasattr(self, "model_description"):
            return
        model_id = self.model_combo.currentData()
        option = next(item for item in MODEL_OPTIONS if item["id"] == model_id)
        self.model_description.setText(
            option["description"] + " 首次使用会自动下载并缓存。"
        )

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
            QLineEdit, QComboBox, QPlainTextEdit {{
                background: {c['surface_alt']};
                border: 1px solid {c['border']};
                border-radius: 6px;
                padding: 7px 9px;
                selection-background-color: {c['selection']};
            }}
            QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{
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
        config = {
            "mode": mode,
            "audio_source": self.audio_source_combo.currentData(),
            "model": self.model_combo.currentData(),
            "no_vad": self.no_vad_cb.isChecked(),
            "no_burn": self.no_burn_cb.isChecked(),
            "refresh": self.refresh_cb.isChecked(),
            "ocr_assist": self.ocr_assist_cb.isChecked(),
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
            if not lyrics:
                messagebox.showwarning("缺少歌词", "请选择歌词文件。")
                return
            if not Path(lyrics).is_file():
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
            if not lyrics_dir or not Path(lyrics_dir).is_dir():
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
        required_modules = {"faster_whisper", "rapidfuzz"}
        if config["audio_source"] == "vocals":
            required_modules.add("demucs")
        if config.get("ocr_assist"):
            required_modules.update({"rapidocr", "onnxruntime"})

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
            "audio_source": config["audio_source"],
            "model": config["model"],
            "ocr_assist": config.get("ocr_assist", False),
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
                        "处理没有完成。界面中保留了错误摘要，完整信息可在日志目录中查看。",
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
        self.audio_source_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.no_burn_cb.setEnabled(not running)
        self.refresh_cb.setEnabled(not running)
        self.ocr_assist_cb.setEnabled(not running)
        self.no_vad_cb.setEnabled(not running)
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
