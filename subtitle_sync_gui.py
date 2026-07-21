#!/usr/bin/env python3
"""
subtitle_sync_gui.py
====================
subtitle_sync 的图形界面版本  v2.0
现代深色主题  ·  自动 CJK 字体适配  ·  中日双语支持
"""

import sys
import os
import shutil
import threading
import subprocess
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
    APP_DIR = Path(__file__).parent

SYNC_SCRIPT = APP_DIR / "subtitle_sync.py"


def _find_python() -> str:
    """找到可用于运行 subtitle_sync.py 的 Python 解释器。
    
    注意：Windows 的 Microsoft Store 版 Python 会在 PATH 中放置存根
    （如 %LOCALAPPDATA%\\Microsoft\\WindowsApps\\python.exe），
    这些存根不是真正的 .exe，执行会报 WinError 193，必须跳过。
    """
    if getattr(sys, "frozen", False):
        for name in ["python", "python3", "py"]:
            found = shutil.which(name)
            if not found:
                continue
            # ★ 跳过 WindowsApps 商店存根（会导致 WinError 193）
            if "WindowsApps" in found:
                continue
            # ★ 验证候选可用（运行 --version 确认）
            try:
                r = subprocess.run(
                    [found, "--version"],
                    capture_output=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                if r.returncode == 0:
                    return found
            except Exception:
                continue

        # 回退：尝试系统目录下的 Python Launcher
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for candidate in [
            os.path.join(windir, "py.exe"),
            os.path.join(windir, "System32", "py.exe"),
        ]:
            if os.path.isfile(candidate):
                return candidate

        raise RuntimeError(
            "未找到可用的 Python 环境。\n\n"
            "请从 https://python.org 下载安装 Python 3.10+，\n"
            "并确保安装时勾选 'Add Python to PATH'。\n\n"
            "注意：Microsoft Store 版本的 Python 不被支持。"
        )
    else:
        return sys.executable


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

        python_exe = _find_python()

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

            cmd = [python_exe, str(SYNC_SCRIPT),
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

            cmd = [python_exe, str(SYNC_SCRIPT),
                   "--batch", "--video-dir", vdir, "--lyrics-dir", ldir]
            if odir:
                cmd += ["--output-dir", odir]

        cmd += ["--audio-source", self.audio_source_var.get()]
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
        self._log(f"\n{'─'*52}\n▶ 开始处理\n{'─'*52}\n")

        threading.Thread(target=self._run_cmd, args=(cmd,), daemon=True).start()

    def _run_cmd(self, cmd):
        no_burn = "--no-burn" in cmd
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1,
            )
            for line in proc.stdout:
                self._log(line)
            proc.wait()
            if proc.returncode == 0:
                self._log(f"\n{'─'*52}\n✅ 全部完成！\n{'─'*52}\n")
                self._set_status("✅ 完成", C["green"])
                message = "SRT 与对轴报告已生成！" if no_burn else "字幕烧入完成！"
                self.after(0, lambda: messagebox.showinfo("完成", message))
            else:
                self._log(f"\n❌ 脚本退出，返回码: {proc.returncode}\n")
                self._set_status("❌ 出错", C["red"])
        except Exception as e:
            self._log(f"\n❌ 启动失败: {e}\n")
            self._set_status("❌ 出错", C["red"])
        finally:
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
