# 日语歌曲字幕对轴工具

支持两种处理模式：

- **MV 歌词打轴**：需提供逐行对应的中日歌词。
- **视频自动字幕**：无需原稿，供直播切片等视频使用。

可以生成 SRT 或直接生成嵌入硬字幕的视频，本工具无自动翻译功能。

## 下载 Release 直接使用（推荐）

1. 在本仓库的 **Releases** 页面下载 `SubtitleSync.exe`。
2. 双击 `SubtitleSync.exe` 打开界面，无需安装或解压。
3. 选择“MV 歌词打轴”或“视频自动字幕”，拖入文件后开始。

EXE 仅包含GUI、处理脚本和 Noto 字体。
音频处理依赖安装在 Python 环境中，安装命令见下方对应路线。模型首次使用时会自动下载到本地。

## 从源码运行

安装 Python 3.12，在项目目录执行：

```powershell
py -3.12 -m venv .venv-gui
.\.venv-gui\Scripts\python.exe -m pip install -r requirements-gui.txt
.\.venv-gui\Scripts\python.exe subtitle_sync_gui.py
```

再按使用的路线安装 `.venv-align` 或 `.venv-whisperx`。界面会自动选择对应环境。

## 适合场景与局限

适合：

- 日语歌曲、MV、Live：提供逐行对应的中日歌词，生成中文时间轴。
- 访谈、解说、Vlog、直播切片：自动识别并生成原语言字幕。
- 画面带有日文歌词的 MV：WhisperX 路线可启用 OCR 辅助。

局限：

- MV 歌词打轴面向日语MV，需要完整的日文原词和行数对应的中文；漏词会严重影响精度。
- 和声、多人重叠、混响、重伴奏会影响识别和对齐。尤其是MMS+TIFA路线。
- OCR 需要清晰可读的画面歌词，动态文字、重复短句和提前出现的歌词会影响匹配。
- MMS+TIFA 路线暂不支持 OCR辅助 和 VAD。

MV 可选 Demucs 人声分离；视频自动字幕使用原始音频，默认启用 VAD。

## 打轴方式与模型


| 选项 | 适合内容 | 取舍 |
|---|---|---|
| `歌词声学对齐`（推荐） | 已有逐行对应中日歌词的 MV | MMS+TIFA 路线：轻量准确；FP32 |
| `Large v3` | 日语歌曲、现场、背景声复杂或音质一般的视频 | Whisper 识别 + WhisperX 对齐，准确度优先 |
| `Large v3 Turbo` | 访谈、解说、Vlog 和一般视频 | 接近 Large v3，处理速度更快 |
| `Medium` | 较清晰的语音、硬件资源有限的电脑 | 速度与精度折中 |
| `Small` | 清晰单人语音、快速草稿、CPU 环境 | 最轻量 |

模型能力和相对速度可参考 [OpenAI Whisper 模型表](https://github.com/openai/whisper#available-models-and-languages)，自动下载行为见 [faster-whisper 使用说明](https://github.com/SYSTRAN/faster-whisper#model-conversion)。

## 对轴方式

### 视频自动字幕（WhisperX）

在顶部“处理模式”选择“视频自动字幕”，拖入视频，选择 Whisper 模型和识别语言后开始。默认使用原始音频并启用 VAD。支持自动检测语言，或手动选择 Whisper 支持的语言；对齐模型按语言自动下载并缓存。

```text
视频 → 提取音频 → faster-whisper 识别
     → 释放识别模型 → WhisperX 声学对齐 → 分句 → SRT／烧录视频
```

使用 WhisperX 3.8.6 的 41 个默认对齐语言配置。

运行依赖放在独立环境 `.venv-whisperx`。以下为 CPU 安装方式：

```powershell
py -3.12 -m venv .venv-whisperx
.\.venv-whisperx\Scripts\python.exe -X utf8 -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-whisperx\Scripts\python.exe -X utf8 -m pip install -r requirements-speech.txt
```

GUI 会优先选用这个环境。生成 `.speech.json` 缓存和 `.alignment.json` 对轴报告；“重新识别”会忽略当前视频的字幕缓存。

命令行：

```powershell
.\.venv-whisperx\Scripts\python.exe -X utf8 subtitle_sync.py --task speech --video clip.mp4 --language auto --no-burn
.\.venv-whisperx\Scripts\python.exe -X utf8 subtitle_sync.py --task speech --batch --video-dir ./videos --language en --no-burn
```

### 歌词声学对齐（MMS+TIFA，推荐）

在“MV 歌词打轴”中选择“歌词声学对齐（推荐）”。这条路线使用歌词强制对齐（Forced Alignment）：日文原词直接参与音频对齐，再按行号把时间映射到中文。

在目前测试的日语 MV 打轴场景中，相比原 Whisper Large v3 ASR 路线，该方案对轴更精确，速度更快，模型也更轻量。

该路线依据演唱时间打轴，需要跟随 MV 画面歌词时，可手动调整字幕时间。

支持“中文整块＋双空行＋日文整块”和逐行 `日文|中文` 格式，两种语言的非空行数必须一致。中文用于字幕输出；修改中文翻译可复用声学对齐缓存。

```text
视频 → 提取音频 → 分离人声（可选）
     → 日文读音转换 → MMS 帧概率 + 整曲顺序路径搜索
     → 不超过 24 秒的带上下文窗口 → TIFA 音素边界
     → 按原词行号映射中文 → SRT + 对轴报告
```

首次使用下载固定版本的 MMS、TIFA 和日语读音词典。运行依赖安装到 `.venv-align`：

```powershell
py -3.12 -m venv .venv-align
.\.venv-align\Scripts\python.exe -X utf8 -m pip install -r requirements-align.txt
```

除 SRT 和可选的视频外，还会生成一个 `.alignment.json` 对轴报告。

命令行入口：

```powershell
.\.venv-align\Scripts\python.exe subtitle_sync.py --video song.mp4 --lyrics song.txt --model lyrics-mms-tifa --no-burn
```

MMS 与 TIFA 权重使用 CC BY-NC-SA 4.0，见 [第三方许可](THIRD_PARTY_NOTICES.md)。

### MV WhisperX 对齐

在“MV 歌词打轴”模式中选择任一 Whisper 模型，即使用 WhisperX 路线。需要逐行对应的日文原词和中文字幕，支持现有整块分隔格式，也支持逐行格式：

```text
世界の終わりで|在世界终结之处
生まれた光|诞生的光芒
君を探して|寻找着你
```

处理流程：

```text
视频 → FFmpeg 提取音频 → Demucs 人声分离（可选）
     → Whisper 逐词识别
     → 日文歌词与识别文本建立定位锚点
     → WhisperX 用日文原歌词进行字符级声学对齐
     → 按原词行号映射中文字幕
     → 可选 OCR 扫描画面歌词，校正局部漂移和明显错位边界
     → SRT + 对轴报告 → 可选 FFmpeg 硬字幕烧入
```

定位时没有匹配到的歌词，会连同相邻已定位歌词一起交给 WhisperX 对齐。未完整对齐但有识别锚点的行保留锚点时间，记为 `asr_anchor`；没有定位结果的行写入报告并停止输出新 SRT。对齐缓存保存在 `.alignment.json` 的 `whisperx` 字段中，修改中文翻译可复用，修改日文原词后重新对齐。

运行环境使用上文的 `.venv-whisperx` 与 `requirements-speech.txt`；MMS＋TIFA 仍使用独立的 `.venv-align`。

## 运行环境

- Python 3.12
- FFmpeg
- 使用画面歌词辅助时需要 RapidOCR 与 ONNX Runtime

GPU 运行需要匹配的 PyTorch/CTranslate2 CUDA 运行库。

## 歌词格式

### 逐行日中双语

使用 `|` 或 Tab 分隔：

```text
世界の終わりで|在世界终结之处
生まれた光|诞生的光芒
```

### 中文段＋日文段

中文整块与日文整块之间保留至少**两个完整空行**：

```text
在世界终结之处
诞生的光芒


世界の終わりで
生まれた光
```

**连续两个及以上空行是块分隔符**；中文块在前、日文块在后，逐行对应，非空行数相同。MV 两条路线都需要双语歌词；视频自动字幕无需歌词。

## 界面与命令行

### 界面功能

界面支持：

- 单文件和批量处理
- 视频和歌词拖拽功能
- 语音模型与歌词声学对齐预设及适用场景说明
- Demucs 人声或原始混合音频
- 仅生成 SRT
- 硬字幕字体与字号选择：Noto Sans 中文、Noto Sans 日文、微软雅黑
- 重新对齐或重新识别
- MV WhisperX 路线可选 OCR 画面歌词辅助
- Whisper 路线可启用或禁用 VAD
- 固定显示当前阶段、进度条和已运行时间

GUI 按所选路线检查 Python 环境和依赖，优先使用项目中的 `.venv-align` 或 `.venv-whisperx`。缺少组件时会显示安装命令。

可使用环境变量 `SUBTITLE_SYNC_PYTHON` 指定专用解释器。

### 自动运行诊断

GUI 会把每次运行的完整输出、实际选中的 Python、候选环境检查结果、命令、退出码和时间自动保存到：

```text
logs/latest.log
logs/latest.json
logs/run-YYYYMMDD-HHMMSS.log
logs/run-YYYYMMDD-HHMMSS.json
```

EXE 的日志位于程序同目录的 `logs/`；源码运行时位于项目的 `logs/`。

### MV 单文件

MMS+TIFA，仅生成 SRT：

```powershell
.\.venv-align\Scripts\python.exe subtitle_sync.py --video song.mp4 --lyrics song.txt --model lyrics-mms-tifa --no-burn
```

WhisperX，使用 Large v3：

```powershell
.\.venv-whisperx\Scripts\python.exe subtitle_sync.py --video song.mp4 --lyrics song.txt --model large-v3 --no-burn
```

常用参数：

- `--no-burn`：只生成字幕和报告；去掉后烧录视频。
- `--audio-source original`：使用原始混合音频；默认 `vocals` 分离人声。
- `--refresh-transcript`：重新对齐或重新识别。
- `--no-vad`：Whisper 识别保留所有音频区间。
- `--ocr-assist`：MV WhisperX 路线启用 OCR，默认每 0.5 秒扫描一帧。
- `--min-match-score 60`：MV WhisperX 路线的文本定位锚点最低相似度。

OCR 使用 RapidOCR，模型随 Python 包安装，扫描结果保存在 `.ocr.json`。

### 批量处理

视频和歌词文件需要同名、扩展名不同：

```text
videos/song_a.mp4
lyrics/song_a.txt
```

```powershell
.\.venv-align\Scripts\python.exe subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics --model lyrics-mms-tifa --no-burn
```

## 输出文件

假设输出视频为 `song_subtitled.mp4`：

| 文件 | 用途 |
|---|---|
| `song_subtitled.mp4` | 烧入硬字幕的视频；使用 `--no-burn` 时不生成 |
| `song_subtitled.srt` | 字幕文件 |
| `song_subtitled.alignment.json` | 对轴报告；MMS+TIFA 和 MV WhisperX 也在此保存对齐缓存 |
| `song_subtitled.whisper.json` | MV WhisperX 路线的原始识别片段与逐词时间缓存 |
| `song_subtitled.speech.json` | 视频自动字幕的识别与对齐缓存 |
| `song_subtitled.ocr.json` | 开启画面歌词辅助时生成的 OCR 扫描缓存 |

报告中的时间来源：

- MMS+TIFA：`refined` 使用局部精对齐结果，`coarse_only` 使用 MMS 整曲对齐结果。
- MV WhisperX：`whisperx` 为字符级声学对齐，`asr_anchor` 为识别文本锚点时间，`unresolved` 为未定位行。
- 视频自动字幕：`aligned_segments` 和 `asr_segments` 分别统计声学对齐片段与保留识别时间的片段。

## 项目结构

```text
Subtitle-tool/
├─ subtitle_sync.py          核心处理与命令行入口
├─ subtitle_sync_gui.py      PySide6 图形界面
├─ asr_models.py             语音模型与权重版本
├─ subtitle_style.py         字幕字体预设与压制样式
├─ assets/fonts/             内置 Noto Sans CJK 字体与许可
├─ lyric_alignment/          歌词声学对齐核心、模型资源与输出流程
├─ requirements.txt          基础识别、OCR 和视频处理依赖
├─ requirements-gui.txt      源码运行与打包 GUI 的依赖
├─ requirements-align.txt    MMS+TIFA 运行依赖
├─ requirements-speech.txt   WhisperX 两种模式的运行依赖
├─ speech_catalog.py         识别语言与对齐模型目录
├─ speech_subtitles.py       视频自动识别与 WhisperX 对齐
├─ mv_whisperx.py            MV 原歌词的 WhisperX 逐行对齐
├─ build.spec                PyInstaller 打包配置
├─ tests/                    对轴、缓存和 GUI 测试
├─ tools/benchmark_asr.py     独立转录文字与时间误差评测
├─ tools/benchmark_lyrics.py  歌词声学对齐多曲回归与逐句误差评测
├─ dist/                     打包输出（不提交到 Git）
│  └─ SubtitleSync.exe        可单独发布的 Windows 图形界面程序
├─ LICENSE                   MIT License
├─ THIRD_PARTY_NOTICES.md     第三方模型与许可
└─ README.md                 使用说明
```
## 打包说明

```powershell
.\.venv-gui\Scripts\python.exe -m PyInstaller build.spec
```

输出为 `dist/SubtitleSync.exe`。EXE 内置 PySide6、处理脚本、字体和许可文件；音频处理在独立 Python 环境中运行。外置更新核心脚本时，需要同时更新对应模块：`lyric_alignment/` 或 `speech_catalog.py`、`speech_subtitles.py`、`mv_whisperx.py`，以及共用的 `asr_models.py`、`subtitle_style.py`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖双语行映射、声学对齐、WhisperX 字符时间映射、字幕分句、缓存、对轴报告、Python 环境选择和 GUI 控件。

## 许可证

项目代码采用 [MIT License](LICENSE)。第三方代码、模型、词典与字体的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
