"""Pinned upstream code and weights. Downloads are atomic and hash checked."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

TIFA_COMMIT = "7b210ca36ff5a2b79187fb1727af58fc977cc301"
TIFA_SOURCE_SHA256 = "0c8bc5e2e9ec6968f2803912bcfae78e39e5739b8869d25c729703aa7411700e"
TIFA_MODEL_SHA256 = "36595b2746f0d1183346fd448c7b99df9d7c51fd403bc45025438c3ebf439396"
MMS_REPO = "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"
MMS_REVISION = "2ab2b5f46539ee284703c281f286b01d2410ee12"


def cache_root() -> Path:
    override = os.environ.get("SUBTITLE_ALIGN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return base / "SubtitleSync" / "lyric-alignment"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and sha256(path) == expected:
        return
    print(f"    下载对齐资源: {path.name}", flush=True)
    # A unique temporary file also prevents interrupted downloads from becoming cache hits.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
        partial = Path(handle.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SubtitleSync/lyric-alignment"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256(partial) != expected:
            raise RuntimeError(f"对齐资源校验失败: {path.name}")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def safe_extract(archive: Path, target: Path) -> None:
    root = target.resolve()
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            normalized = entry.filename.replace("\\", "/")
            resolved = (root / normalized).resolve()
            if not resolved.is_relative_to(root) or ":" in normalized or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {entry.filename}")
        package.extractall(root)


def _install_archive(archive: Path, destination: Path, inner: str, required: tuple[str, ...]) -> Path:
    marker = destination / ".verified.json"
    archive_digest = sha256(archive)
    if marker.is_file():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved["archive"] == archive_digest and all(
                (destination / name).is_file() and sha256(destination / name) == saved["files"][name]
                for name in required
            ):
                return destination
        except (OSError, ValueError, KeyError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="unpack-") as temp:
        temporary = Path(temp)
        safe_extract(archive, temporary)
        unpacked = temporary / inner
        if not all((unpacked / name).is_file() for name in required):
            raise RuntimeError(f"对齐资源不完整: {archive.name}")
        record = {"archive": archive_digest, "files": {name: sha256(unpacked / name) for name in required}}
        (unpacked / ".verified.json").write_text(json.dumps(record), encoding="utf-8")
        # Only replace this exact managed package, never its cache root or user data.
        if destination.exists():
            old = temporary / "previous-package"
            destination.rename(old)
        unpacked.rename(destination)
    return destination


def prepare_tifa() -> tuple[Path, Path]:
    from filelock import FileLock
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "prepare.lock"), timeout=300):
        source_zip = root / "downloads" / f"TIFA-{TIFA_COMMIT}.zip"
        model_zip = root / "downloads" / "TIFA-1.0.zip"
        download(f"https://codeload.github.com/openvpi/TIFA/zip/{TIFA_COMMIT}", source_zip, TIFA_SOURCE_SHA256)
        download("https://github.com/openvpi/TIFA/releases/download/v1.0.0/TIFA-1.0.zip", model_zip, TIFA_MODEL_SHA256)
        source = _install_archive(source_zip, root / f"source-{TIFA_COMMIT}", f"TIFA-{TIFA_COMMIT}", (
            "inference/backend.py", "lib/g2p_encoding.py", "lib/path_traversal.py", "modules/decoding.py", "LICENSE",
        ))
        model = _install_archive(model_zip, root / "TIFA-1.0", "TIFA-1.0", (
            "model.pt", "config.yaml", "vocabulary.json", "dictionaries/japanese_dict_full.txt",
            "dictionaries/ds_cmudict-07b.txt", "assets/LstmG2p-Eng/encoder.onnx", "assets/LstmG2p-Eng/decoder.onnx",
        ))
    return source, model


def prepare_mms() -> Path:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(
        MMS_REPO, revision=MMS_REVISION,
        allow_patterns=["*.json", "model.safetensors"],
    ))
    if not (path / "model.safetensors").is_file():
        raise RuntimeError("MMS 模型权重不完整")
    return path

