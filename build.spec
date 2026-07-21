# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for subtitle_sync_gui
打包成单文件夹，包含 GUI + subtitle_sync.py
"""

import sys
from pathlib import Path

_project = Path(SPECPATH)  # SPECPATH = spec 文件所在目录

a = Analysis(
    [str(_project / "subtitle_sync_gui.py")],
    pathex=[str(_project)],
    binaries=[],
    datas=[
        (str(_project / "subtitle_sync.py"), "."),
        (str(_project / "requirements.txt"), "."),
    ],
    hiddenimports=[
        "tkinter",
        "tkinter.font",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除 GUI 不需要的重依赖（它们由系统 Python 提供）
        "torch",
        "torchvision",
        "torchaudio",
        "faster_whisper",
        "demucs",
        "numpy",
        "ctranslate2",
        "onnxruntime",
        "soundfile",
        "av",
        "pandas",
        "sympy",
        "networkx",
        "tokenizers",
        "huggingface_hub",
        "transformers",
        "streamlink",
        "yt_dlp",
        "playwright",
        "biliup",
        "openpyxl",
        "jinja2",
        "sqlalchemy",
        "rich",
        "PIL",
        "cv2",
        "matplotlib",
        "scipy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SubtitleSync",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # ★ 无控制台窗口
    icon=None,
)
