#!/usr/bin/env python3
"""
subtitle_sync_gui.py
====================
subtitle_sync 的图形界面版本  v2.0
现代深色主题  ·  自动 CJK 字体适配  ·  中日双语支持
"""

import sys
import os
import json
import re
import shutil
import threading
import subprocess
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

# ── Windows UTF-8 修复 ────────────────────────────────────────────────────────
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# ── 路径解析（兼容 PyInstaller 打包）─────────────────────────────────────────
if getattr(sys, "frozen", False):
    # PyInstaller 单文件模式：优先用 exe 同目录的 subtitle_sync.py（方便更新）
    EXE_DIR = Path(sys.executable).parent
    external_script = EXE_DIR / "subtitle_sync.py"
    if external_script.exists():
        APP_DIR = EXE_DIR
    elif hasattr(sys, "_MEIPASS"):
        APP_DIR = Path(sys._MEIPASS)   # 回退到打包内置版本
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
    "imageio_ffmpeg",
    "torch",
)


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
        if sys.platform == "win32":
            path = root / ".venv" / "Scripts" / "python.exe"
        else:
            path = root / ".venv" / "bin" / "python"
        candidates.append((str(path), "项目 .venv", 90))

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
    """探测所有 Python，选择真正具备当前模式依赖的解释器。"""
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
            module for module in sorted(required_modules)
            if not probe.get("modules", {}).get(module, False)
        ]
        version = ".".join(map(str, probe.get("version") or [])) or "不可用"
        checked.append(
            f"- {probe['path']}（Python {version}，缺少: {', '.join(missing) or '无'}）"
        )
    requirements = _requirements_path()
    best_path = probes[0]["path"] if probes else "python"
    raise RuntimeError(
        "没有找到满足当前处理模式依赖的 Python。\n\n"
        + ("已检查：\n" + "\n".join(checked) + "\n\n" if checked else "")
        + "可在专用环境中安装依赖：\n"
        + f'"{best_path}" -m pip install -r "{requirements}"'
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  字体检测                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _detect_cjk_font() -> str:
    """检测系统最佳 CJK 字体，优先选中文渲染好的。"""
    # 创建临时根窗口（tkfont.families 需要 root 存在）
    root = tk.Tk()
    root.withdraw()
    try:
        available = set(tkfont.families(root=root))
    finally:
        root.destroy()

    candidates = [
        "Microsoft YaHei UI",    # Win 中文 UI 优化版 ⬅ 最佳中文渲染
        "Microsoft YaHei",
        "Noto Sans SC",          # Google 思源黑体（简体）
        "Yu Gothic UI",          # Win 日文 UI
        "Meiryo UI",
        "Noto Sans JP",
        "SimHei",
        "MS UI Gothic",
    ]
    for f in candidates:
        if f in available:
            return f
    return "Segoe UI"


def _detect_mono_font() -> str:
    """检测系统最佳等宽字体。"""
    root = tk.Tk()
    root.withdraw()
    try:
        available = set(tkfont.families(root=root))
    finally:
        root.destroy()
    candidates = [
        "Cascadia Code",
        "Cascadia Mono",
        "JetBrains Mono",
        "Consolas",
        "Courier New",
    ]
    for f in candidates:
        if f in available:
            return f
    return "Consolas"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  配色方案  ——  现代 Slate 深色主题                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

C = {
    "bg":       "#0b1120",   # 最深背景
    "surface":  "#151d2e",   # 卡片底色
    "border":   "#1e2d45",   # 边框
    "accent":   "#6366f1",   # 靛蓝主色
    "accent2":  "#4f46e5",   # 靛蓝深色
    "text":     "#e2e8f0",   # 主文字
    "dim":      "#7c8798",   # 次要文字
    "green":    "#34d399",   # 成功
    "yellow":   "#fbbf24",   # 警告
    "red":      "#f87171",   # 错误
    "entry_bg": "#0f172a",   # 输入框底色
    "log_bg":   "#070c16",   # 日志底色
    "white":    "#ffffff",
}

# 全局字体（启动时注入）
FONT_CJK  = _detect_cjk_font()
FONT_MONO = _detect_mono_font()

F = {
    "title":  (FONT_CJK, 13, "bold"),   # 15→13
    "h2":     (FONT_CJK, 10, "bold"),   # 11→10
    "body":   (FONT_CJK, 9),            # 10→9
    "small":  (FONT_CJK, 8),            # 9→8
    "mono":   (FONT_MONO, 9),
    "btn":    (FONT_CJK, 10, "bold"),   # 11→10
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UI 组件                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ModernButton(tk.Button):
    """现代扁平按钮：圆角感 + hover 效果。"""
    def __init__(self, parent, text, style="primary", command=None, **kw):
        if style == "primary":
            bg, fg, hover_bg = C["accent"], C["white"], C["accent2"]
        elif style == "ghost":
            bg, fg, hover_bg = C["surface"], C["dim"], C["border"]
        else:
            bg, fg, hover_bg = style, C["white"], style

        super().__init__(
            parent, text=text,
            font=F["btn"],
            bg=bg, fg=fg,
            activebackground=hover_bg, activeforeground=fg,
            relief="flat",
            bd=0, padx=18, pady=6,   # 22/8 → 18/6
            cursor="hand2",
            command=command,
            **kw,
        )


class FileRow(tk.Frame):
    """文件选择行：标签 + 输入框 + 浏览按钮。"""
    def __init__(self, parent, label, var, filetypes=None, is_dir=False,
                 is_save=False, is_video=False, placeholder="", app=None):
        super().__init__(parent, bg=C["surface"])
        self.app = app
        self.var = var
        self.is_dir = is_dir
        self.is_save = is_save
        self.is_video = is_video
        self.filetypes = filetypes or []
        self.columnconfigure(1, weight=1)

        # 标签
        tk.Label(self, text=label, font=F["body"],
                 bg=C["surface"], fg=C["dim"], width=10, anchor="e"
                 ).grid(row=0, column=0, sticky="e", padx=(0, 10), pady=3)

        # 输入框
        self.entry = tk.Entry(
            self, textvariable=var, font=F["small"],
            bg=C["entry_bg"], fg=C["text"],
            insertbackground=C["accent"],
            relief="flat", bd=0,
        )
        self.entry.grid(row=0, column=1, sticky="ew", pady=3, ipady=3)   # 5→3
        self.entry.configure(highlightthickness=1, highlightbackground=C["border"],
                             highlightcolor=C["accent"])

        if placeholder:
            self._add_placeholder(placeholder)

        # 浏览按钮
        tk.Button(
            self, text="浏览", font=F["small"],
            bg=C["border"], fg=C["text"],
            activebackground=C["accent"], activeforeground=C["white"],
            relief="flat", bd=0, padx=14, pady=5,
            cursor="hand2",
            command=self._browse,
        ).grid(row=0, column=2, padx=(8, 0), pady=5)

    def _add_placeholder(self, text):
        self._placeholder = text
        def on_focus_in(e):
            if not self.var.get():
                self.entry.config(fg=C["text"])
        def on_focus_out(e):
            pass
        self.entry.bind("<FocusIn>", on_focus_in)
        self.entry.bind("<FocusOut>", on_focus_out)

    def _browse(self):
        path = ""
        if self.is_dir:
            path = filedialog.askdirectory()
        elif self.is_save:
            path = filedialog.asksaveasfilename(
                defaultextension=".mp4", filetypes=self.filetypes)
        else:
            path = filedialog.askopenfilename(filetypes=self.filetypes)

        if path:
            self.var.set(path)
            # ★ 仅视频行触发自动推断
            if self.app and self.is_video:
                self.app._on_video_selected(Path(path))


class SegmentedControl(tk.Frame):
    """分段控制器：iOS 风格的 pill 按钮组。"""
    def __init__(self, parent, var, options, command=None):
        super().__init__(parent, bg=C["bg"])
        self.var = var
        self.buttons = []

        inner = tk.Frame(self, bg=C["surface"])
        inner.pack()

        for i, (text, val) in enumerate(options):
            btn = tk.Button(
                inner, text=text, font=F["body"],
                relief="flat", bd=0, padx=18, pady=6,
                cursor="hand2",
                command=lambda v=val: self._select(v, command),
            )
            btn.pack(side="left")
            self.buttons.append((btn, val))

        self._update_style()

    def _select(self, val, command):
        self.var.set(val)
        self._update_style()
        if command:
            command()

    def _update_style(self):
        cur = self.var.get()
        for btn, val in self.buttons:
            if val == cur:
                btn.config(bg=C["accent"], fg=C["white"],
                           activebackground=C["accent2"], activeforeground=C["white"])
            else:
                btn.config(bg=C["surface"], fg=C["dim"],
                           activebackground=C["border"], activeforeground=C["text"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  主应用                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Subtitle Sync  —  日语歌曲中文字幕自动对轴")
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        print(f"  GUI 字体: CJK={FONT_CJK}  |  Mono={FONT_MONO}")

        # ── 状态变量 ──
        self.video_var      = tk.StringVar()
        self.lyrics_var     = tk.StringVar()
        self.output_var     = tk.StringVar()
        self.mode_var       = tk.StringVar(value="single")
        self.no_vad_var     = tk.BooleanVar(value=False)
        self.no_burn_var    = tk.BooleanVar(value=False)
        self.refresh_var    = tk.BooleanVar(value=False)
        self.audio_source_var = tk.StringVar(value="vocals")
        self.video_dir_var  = tk.StringVar()
        self.lyrics_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.running        = False

        self._build_ui()
        self._center_window(680, 500)

    # ── 自适应屏幕（兼容 DPI 缩放）──────────────────────────────────────────
    def _center_window(self, w, h):
        self.update_idletasks()
        sw = self.winfo_screenwidth()    # 物理像素
        sh = self.winfo_screenheight()   # 物理像素

        # ★ DPI 缩放修正：geometry() 使用逻辑像素，需换算
        #    125% 缩放 → scale=1.25 → 700 逻辑像素 = 875 物理像素
        try:
            dpi_scale = float(self.tk.call("tk", "scaling"))
        except Exception:
            dpi_scale = 1.0

        # 窗口逻辑高度上限 = (屏幕物理高度 - 任务栏等140px) / 缩放比
        max_h = int((sh - 140) / dpi_scale)
        max_w = int((sw - 40) / dpi_scale)

        w = min(w, max_w)
        h = min(h, max_h)
        x = int((sw / dpi_scale - w) // 2)
        y = int((sh / dpi_scale - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(min(w - 40, 580), min(h - 60, 380))

    def _build_ui(self):
        # ── 标题栏 ──
        header = tk.Frame(self, bg=C["surface"], pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="🎬  字幕自动对轴工具",
            font=F["title"], bg=C["surface"], fg=C["text"]
        ).pack()
        tk.Label(
            header,
            text="Demucs 人声分离  →  Whisper 识别  →  FFmpeg 硬字幕烧入",
            font=F["small"], bg=C["surface"], fg=C["dim"]
        ).pack(pady=(1, 0))

        # ── 模式切换 ──
        seg_frame = tk.Frame(self, bg=C["bg"], pady=8)
        seg_frame.pack(fill="x")
        self.mode_ctrl = SegmentedControl(
            seg_frame, self.mode_var,
            [("单文件", "single"), ("批量处理", "batch")],
            command=self._on_mode_change,
        )
        self.mode_ctrl.pack()

        # ── 文件选择卡片 ──
        card_outer = tk.Frame(self, bg=C["bg"])
        card_outer.pack(fill="x", padx=24, pady=(0, 4))
        card = tk.Frame(card_outer, bg=C["surface"])
        card.pack(fill="x")

        # 单文件模式
        self.single_frame = tk.Frame(card, bg=C["surface"])
        self.single_frame.pack(fill="x", padx=16, pady=(10, 2))
        FileRow(self.single_frame, "视频文件", self.video_var,
                [("视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.webm *.m4v")],
                is_video=True, app=self).pack(fill="x", pady=2)
        FileRow(self.single_frame, "歌词文件", self.lyrics_var,
                [("文本文件", "*.txt")],
                app=self).pack(fill="x", pady=2)
        FileRow(self.single_frame, "输出文件", self.output_var,
                [("MP4 视频", "*.mp4")], is_save=True,
                placeholder="留空 → 自动命名",
                app=self).pack(fill="x", pady=2)

        # 批量模式（初始隐藏）
        self.batch_frame = tk.Frame(card, bg=C["surface"])
        FileRow(self.batch_frame, "视频目录", self.video_dir_var,
                is_dir=True, app=self).pack(fill="x", pady=2)
        FileRow(self.batch_frame, "歌词目录", self.lyrics_dir_var,
                is_dir=True, app=self).pack(fill="x", pady=2)
        FileRow(self.batch_frame, "输出目录", self.output_dir_var,
                is_dir=True, placeholder="留空 → video-dir/output",
                app=self).pack(fill="x", pady=2)

        # 提示标签
        tk.Label(
            card, font=F["small"], bg=C["surface"], fg=C["yellow"],
            text="💡  歌词格式：纯中文 / 逐行日|中 / 中文段+日文段  —  自动检测",
        ).pack(anchor="w", padx=16, pady=(0, 4))

        source_row = tk.Frame(card, bg=C["surface"])
        source_row.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(
            source_row, text="识别音源", font=F["small"],
            bg=C["surface"], fg=C["dim"],
        ).pack(side="left", padx=(0, 8))
        SegmentedControl(
            source_row,
            self.audio_source_var,
            [("Demucs 人声", "vocals"), ("原始音频", "original")],
        ).pack(side="left")

        # ── 日志区域 ──
        log_outer = tk.Frame(self, bg=C["bg"])
        log_outer.pack(fill="both", expand=True, padx=24, pady=(0, 4))
        log_card = tk.Frame(log_outer, bg=C["surface"])
        log_card.pack(fill="both", expand=True)

        tk.Label(log_card, text="运行日志", font=F["h2"],
                 bg=C["surface"], fg=C["text"]
                 ).pack(anchor="w", padx=14, pady=(8, 2))

        log_inner = tk.Frame(log_card, bg=C["log_bg"])
        log_inner.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.log_text = tk.Text(
            log_inner, font=F["mono"], bg=C["log_bg"], fg=C["green"],
            insertbackground=C["green"], relief="flat", wrap="word",
            state="disabled", bd=0, padx=8, pady=6,
        )
        self.log_text.pack(fill="both", expand=True, side="left")

        scroll = ttk.Scrollbar(log_inner, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        # ── 底部操作栏 ──
        bar = tk.Frame(self, bg=C["surface"], pady=8)
        bar.pack(fill="x", padx=24, pady=(0, 10))
        bar_inner = tk.Frame(bar, bg=C["surface"])
        bar_inner.pack(fill="x", padx=10)

        self.start_btn = ModernButton(bar_inner, "▶  开始处理", "primary",
                                       command=self._start)
        self.start_btn.pack(side="left")

        ModernButton(bar_inner, "清空日志", "ghost",
                     command=self._clear_log).pack(side="left", padx=10)

        self.no_vad_cb = tk.Checkbutton(
            bar_inner, text=" 禁用 VAD ", variable=self.no_vad_var,
            font=F["body"], bg=C["surface"], fg=C["dim"],
            selectcolor=C["bg"], activebackground=C["surface"],
            activeforeground=C["accent"],
            relief="flat", bd=0, padx=6, cursor="hand2",
        )
        self.no_vad_cb.pack(side="left")

        self.no_burn_cb = tk.Checkbutton(
            bar_inner, text=" 仅生成SRT ", variable=self.no_burn_var,
            font=F["small"], bg=C["surface"], fg=C["dim"],
            selectcolor=C["bg"], activebackground=C["surface"],
            activeforeground=C["accent"], relief="flat", bd=0,
            cursor="hand2",
        )
        self.no_burn_cb.pack(side="left")

        self.refresh_cb = tk.Checkbutton(
            bar_inner, text=" 重新识别 ", variable=self.refresh_var,
            font=F["small"], bg=C["surface"], fg=C["dim"],
            selectcolor=C["bg"], activebackground=C["surface"],
            activeforeground=C["accent"], relief="flat", bd=0,
            cursor="hand2",
        )
        self.refresh_cb.pack(side="left")

        self.status_label = tk.Label(
            bar_inner, text="就绪", font=F["body"],
            bg=C["surface"], fg=C["dim"]
        )
        self.status_label.pack(side="right")

    # ──────────────────────────────────────────────────────────────────────────
    #  交互逻辑
    # ──────────────────────────────────────────────────────────────────────────

    def _on_video_selected(self, path: Path):
        """选视频后自动推断输出 & 歌词路径。"""
        self.output_var.set(str(path.parent / f"{path.stem}_subtitled{path.suffix}"))
        candidate = path.with_suffix(".txt")
        if candidate.exists() and not self.lyrics_var.get():
            self.lyrics_var.set(str(candidate))

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "single":
            self.batch_frame.pack_forget()
            self.single_frame.pack(fill="x", padx=16, pady=(10, 2))
        else:
            self.single_frame.pack_forget()
            self.batch_frame.pack(fill="x", padx=16, pady=(10, 2))

    # ──────────────────────────────────────────────────────────────────────────
    #  运行控制
    # ──────────────────────────────────────────────────────────────────────────

    def _start(self):
        if self.running:
            return

        mode = self.mode_var.get()
        audio_source = self.audio_source_var.get()
        required_modules = {"faster_whisper", "rapidfuzz"}
        if audio_source == "vocals":
            required_modules.add("demucs")
        try:
            selected_python, python_probes = _find_python(required_modules)
        except RuntimeError as error:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().astimezone().isoformat()
            LATEST_LOG.write_text(
                f"[{timestamp}] Python 环境预检失败\n{error}\n",
                encoding="utf-8",
            )
            _write_json(LATEST_DIAGNOSTIC, {
                "started_at": timestamp,
                "status": "preflight_failed",
                "required_modules": sorted(required_modules),
                "error": str(error),
                "log_path": str(LATEST_LOG),
            })
            messagebox.showerror("Python 环境不完整", str(error))
            return

        python_exe = selected_python["path"]

        if mode == "single":
            video  = self.video_var.get().strip()
            lyrics = self.lyrics_var.get().strip()
            output = self.output_var.get().strip()

            if not video:
                messagebox.showwarning("缺少文件", "请选择视频文件")
                return
            if not lyrics:
                messagebox.showwarning("缺少文件", "请选择歌词文件")
                return

            cmd = [python_exe, "-u", str(SYNC_SCRIPT),
                   "--video", video, "--lyrics", lyrics]
            if output:
                cmd += ["--output", output]
        else:
            vdir = self.video_dir_var.get().strip()
            ldir = self.lyrics_dir_var.get().strip()
            odir = self.output_dir_var.get().strip()

            if not vdir:
                messagebox.showwarning("缺少目录", "请选择视频目录")
                return
            if not ldir:
                messagebox.showwarning("缺少目录", "请选择歌词目录")
                return

            cmd = [python_exe, "-u", str(SYNC_SCRIPT),
                   "--batch", "--video-dir", vdir, "--lyrics-dir", ldir]
            if odir:
                cmd += ["--output-dir", odir]

        cmd += ["--audio-source", audio_source]
        if self.no_vad_var.get():
            cmd += ["--no-vad"]
        if self.no_burn_var.get():
            cmd += ["--no-burn"]
        if self.refresh_var.get():
            cmd += ["--refresh-transcript"]

        # 启动前检查 subtitle_sync.py 是否存在
        if not SYNC_SCRIPT.exists():
            messagebox.showerror(
                "缺少脚本",
                f"未找到核心脚本:\n{SYNC_SCRIPT}\n\n请确保 subtitle_sync.py 与本程序在同一目录。"
            )
            return

        self.running = True
        self.start_btn.config(text="⏳  处理中...", state="disabled",
                              bg=C["dim"], activebackground=C["dim"])
        self._set_status("处理中...", C["yellow"])
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_context = {
            "run_id": run_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "status": "starting",
            "mode": mode,
            "audio_source": audio_source,
            "required_modules": sorted(required_modules),
            "selected_python": selected_python,
            "python_candidates": python_probes,
            "command": cmd,
            "log_path": str(LATEST_LOG),
            "run_log_path": str(LOG_DIR / f"run-{run_id}.log"),
        }
        self._log(
            f"\n{'─'*52}\n▶ 开始处理\n{'─'*52}\n"
            f"Python: {python_exe}\n"
            f"自动日志: {LATEST_LOG}\n"
        )

        threading.Thread(
            target=self._run_cmd,
            args=(cmd, run_context),
            daemon=True,
        ).start()

    def _run_cmd(self, cmd, run_context):
        no_burn = "--no-burn" in cmd
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        run_log_path = Path(run_context["run_log_path"])
        exit_code = None
        status = "failed"

        def update_diagnostics(**changes):
            run_context.update(changes)
            _write_json(LATEST_DIAGNOSTIC, run_context)
            _write_json(run_log_path.with_suffix(".json"), run_context)

        try:
            with (
                open(LATEST_LOG, "w", encoding="utf-8") as latest_log,
                open(run_log_path, "w", encoding="utf-8") as run_log,
            ):
                emit_lock = threading.Lock()

                def emit(text):
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
                    f"识别音源: {run_context['audio_source']}\n"
                    f"命令: {subprocess.list2cmdline(cmd)}\n"
                    f"{'─'*60}\n"
                )
                emit(header)
                update_diagnostics(status="running")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    bufsize=1,
                )
                def pump_output():
                    if proc.stdout is not None:
                        with proc.stdout:
                            for line in proc.stdout:
                                emit(line)

                reader = threading.Thread(target=pump_output, daemon=True)
                reader.start()
                started_waiting = time.monotonic()
                next_heartbeat = 30
                while proc.poll() is None:
                    reader.join(timeout=1)
                    elapsed = int(time.monotonic() - started_waiting)
                    if elapsed >= next_heartbeat:
                        emit(f"\n⏳ 仍在处理，已运行 {elapsed} 秒...\n")
                        next_heartbeat += 30
                reader.join(timeout=5)
                proc.wait()
                exit_code = proc.returncode
                if proc.returncode == 0:
                    status = "completed"
                    emit(f"\n{'─'*52}\n✅ 全部完成！\n{'─'*52}\n")
                    self._set_status("✅ 完成", C["green"])
                    message = "SRT 与对轴报告已生成！" if no_burn else "字幕烧入完成！"
                    self.after(0, lambda: messagebox.showinfo("完成", message))
                else:
                    status = "failed"
                    emit(f"\n❌ 脚本退出，返回码: {proc.returncode}\n")
                    self._set_status("❌ 出错", C["red"])
        except Exception as e:
            status = "launch_failed"
            error_text = f"\n❌ 启动失败: {e}\n"
            self._log(error_text)
            try:
                with open(LATEST_LOG, "a", encoding="utf-8") as handle:
                    handle.write(error_text)
                with open(run_log_path, "a", encoding="utf-8") as handle:
                    handle.write(error_text)
            except OSError:
                pass
            run_context["error"] = str(e)
            self._set_status("❌ 出错", C["red"])
        finally:
            update_diagnostics(
                status=status,
                exit_code=exit_code,
                finished_at=datetime.now().astimezone().isoformat(),
            )
            self.running = False
            self.after(0, lambda: self.start_btn.config(
                text="▶  开始处理", state="normal",
                bg=C["accent"], activebackground=C["accent2"]))

    def _log(self, text):
        def _append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", text)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(0, _append)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _set_status(self, text, color):
        self.after(0, lambda: self.status_label.config(text=text, fg=color))


if __name__ == "__main__":
    app = App()
    app.mainloop()
