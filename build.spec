# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：打包 PySide6 GUI，并附带可外置更新的核心脚本。"""

import sys
from pathlib import Path

_project = Path(SPECPATH)  # SPECPATH = spec 文件所在目录

a = Analysis(
    [str(_project / "subtitle_sync_gui.py")],
    pathex=[str(_project)],
    binaries=[],
    datas=[
        (str(_project / "subtitle_sync.py"), "."),
        (str(_project / "asr_models.py"), "."),
        (str(_project / "speech_catalog.py"), "."),
        (str(_project / "speech_subtitles.py"), "."),
        (str(_project / "mv_whisperx.py"), "."),
        (str(_project / "subtitle_style.py"), "."),
        (str(_project / "assets" / "fonts"), "assets/fonts"),
        (str(_project / "assets" / "WHISPERX_LICENSE.txt"), "assets"),
        (str(_project / "requirements.txt"), "."),
        (str(_project / "requirements-align.txt"), "."),
        (str(_project / "requirements-speech.txt"), "."),
        *[(str(file), "lyric_alignment") for file in (_project / "lyric_alignment").glob("*.py")],
        (str(_project / "THIRD_PARTY_NOTICES.md"), "."),
    ],
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
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
        "rapidocr",
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
        "tkinter",
        "PyQt5",
        "PyQt6",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Qt 6 on Windows uses the operating system ICU. A build shell may also expose
# an unrelated Poppler/Conda ICU through PATH; PyInstaller can otherwise bundle
# that DLL next to Qt6Core and cause "procedure not found" at startup.
foreign_icu_names = {"icuuc.dll", "icuin.dll"}
a.binaries = [
    entry
    for entry in a.binaries
    if not (
        (
            Path(entry[0]).name.casefold() in foreign_icu_names
            or (
                Path(entry[0]).name.casefold().startswith("icudt")
                and Path(entry[0]).name.casefold().endswith(".dll")
            )
        )
        and "pyside6" not in str(entry[1]).replace("\\", "/").casefold()
    )
]

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
