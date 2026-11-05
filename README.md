# 日语歌曲字幕对轴工具

从日语歌曲视频提取音频，通过语音识别或歌词强制对齐确定时间轴，并把逐行中文字幕生成 SRT 或烧入视频。

本工具用于把已有中文字幕对到音频时间轴，没有自动翻译功能，亦不是完整的字幕编辑器。

## 下载 Release 直接使用（推荐）

普通使用者无需下载源码或安装 PySide6：

1. 在本仓库的 **Releases** 页面下载 `SubtitleSync.exe`。
2. 双击 `SubtitleSync.exe` 打开界面，无需安装或解压。
3. 拖入视频和歌词，选择模型与处理方式后开始。

Release 中的 EXE 已包含图形界面和核心脚本，但不包含 Whisper、Demucs、OCR 运行库和语音模型。电脑仍需安装 Python 3.10+ 及识别依赖。第一次使用某个语音模型时会自动联网下载；之后会直接使用本机缓存。

如果运行环境不完整，程序会在处理前列出缺少的组件和安装命令。日常使用不需要通过命令行启动界面。

## 从源码运行

需要修改程序或自行打包时，再使用源码方式：

```powershell
pip install -r requirements.txt
pip install -r requirements-gui.txt
python subtitle_sync_gui.py
```

`requirements.txt` 是识别和视频处理依赖；`requirements-gui.txt` 只用于从源码运行或打包图形界面。

## 适合场景与局限

适合：

- 日语歌曲、MV、Live 等音乐视频。推荐提供行数对应的中文与日文歌词。
- 访谈、解说、Vlog、动画对白等非音乐视频。语音清楚时通常比歌曲更容易识别。
- 已经有逐行中文字幕，只需要生成 SRT 时间轴或烧入视频。
- 画面本身带有日文歌词或硬字幕的视频，可选用 OCR 画面歌词辅助校准。

局限：

- **自动对轴结果仅供人工复核，即使使用 `Large v3`，目前算法下个别字幕仍可能有 0–5 秒误差。**
- 必须提供准备好的中文字幕或歌词。
- 日语外的语言的歌词解析与匹配效果尚未经过实测。
- 歌唱、和声、多人重叠、现场噪声、混响和重伴奏都会降低识别与文本锚点覆盖率。
- 没有匹配到的歌词行会在相邻锚点之间补时间，无法达到逐帧精度。
- OCR 只能利用视频里确实存在的可读文字；无画面歌词、纯装饰文字、严重旋转或故意提前/延后的歌词可能无效或产生误差。

非音乐视频通常直接选择“原始音频”即可；“分离人声”主要用于伴奏较重的歌曲，MV等场景推荐分离人声。

## 语音模型

界面提供模型与打轴方式预设，不需要填写模型名称或识别参数：

| 选项 | 适合内容 | 取舍 |
|---|---|---|
| `Large v3`（默认） | 日语歌曲、现场、背景声复杂或音质一般的视频 | 精度优先，处理较慢 |
| `歌词声学对齐`（实验性） | 已有逐行对应的中日歌词的 MV | 歌词强制对齐路线：MMS 整曲声学定位 + TIFA 局部音素精对齐；FP32 |
| `Large v3 Turbo` | 访谈、解说、Vlog 和一般视频 | 接近 Large v3，处理明显更快 |
| `Medium` | 较清晰的语音、硬件资源有限的电脑 | 速度与精度折中 |
| `Small` | 清晰单人语音、快速草稿、CPU 环境 | 最轻量，不建议用于歌曲和嘈杂音频 |

第一次使用某个模型时，faster-whisper 会从 Hugging Face 自动下载并保存在本机缓存中，之后无需重复下载。切换模型会使当前输出旁的 `.whisper.json` 转录缓存失效并重新识别；模型权重缓存不会被删除。

模型能力和相对速度可参考 [OpenAI Whisper 模型表](https://github.com/openai/whisper#available-models-and-languages)，自动下载行为见 [faster-whisper 使用说明](https://github.com/SYSTRAN/faster-whisper#model-conversion)。

## 对轴方式

### 歌词强制对齐（Forced Alignment，实验性）

新增实验性的歌词强制对齐（Forced Alignment）路线，在“打轴方式”中选择“歌词声学对齐”即可使用。日文原词直接参与音频对齐，再按原词行号把时间映射到中文。

在目前测试的日语 MV 打轴场景中，相比本工具原有的 Whisper Large v3 ASR 路线，该方案对轴更精确，模型也更轻量。在测试中字幕起点与终点相对人工标准轴的平均绝对误差约为0.2秒，已达到可用级别。

但该模式目前仅依据音频确定时间轴，无法自动校正 MV 画面字幕与实际演唱之间的错位，暂不支持 OCR 辅助和 VAD。如果要让字幕适配MV字幕，仍需要手动调整。

歌词格式沿用“中文整块＋双空行＋日文整块”，两块非空行数必须一致。中文只用于最后的字幕输出；修改中文翻译可复用声学对齐缓存。纯中文输入和中日行数不一致时会停止并提示检查。

```text
视频 → 提取音频 → 分离人声（可选）
     → 日文读音转换 → MMS 帧概率 + 整曲顺序路径搜索
     → 不超过 24 秒的带上下文窗口 → TIFA 音素边界
     → 按原词行号映射中文 → SRT + 对轴报告
```

首次使用自动下载固定版本的 MMS、TIFA 和日语读音词典，后续从本机缓存加载。依赖单独安装到 `.venv-align`。基础环境已有 PyTorch 和 Demucs 时：

```powershell
py -3.12 -m venv --system-site-packages .venv-align
.\.venv-align\Scripts\python.exe -X utf8 -m pip install -r requirements-align.txt
python subtitle_sync_gui.py
```

依赖清单包含 PyTorch 和 Demucs，已有兼容版本会直接复用。GUI 会检查依赖并优先选用 `.venv-align`。

新模式只生成一个 `.alignment.json`，同时保存对齐结果、音素明细、缓存及待复核行。

命令行入口：

```powershell
.\.venv-align\Scripts\python.exe subtitle_sync.py --video song.mp4 --lyrics song.txt --model lyrics-mms-tifa --no-burn
```

本模式仍处于实验阶段，部分句子的起止时间仍可能有偏差，生成后需要人工复核。模型权重为非商业许可，见 [第三方许可](THIRD_PARTY_NOTICES.md)。

### Whisper 文本匹配

推荐使用逐行双语歌词：

```text
世界の終わりで|在世界终结之处
生まれた光|诞生的光芒
君を探して|寻找着你
```

处理流程：

```text
视频 → FFmpeg 提取音频 → Demucs 人声分离（可选）
     → Whisper 逐词识别
     → 日文歌词与识别文本进行顺序受限匹配
     → 高分匹配作为时间锚点，漏识别行在相邻锚点间局部补齐
     → 可选 OCR 扫描画面歌词，校正局部漂移和明显错位边界
     → SRT + 对轴报告 → 可选 FFmpeg 硬字幕烧入
```

有日文原文时，程序会在 Whisper 独立识别完成后比较日文歌词与识别文字。没有日文原文时，程序优先使用中文字幕中的英文/数字短语建立局部锚点；可用锚点不足时才使用低置信度时间比例回退。

## 运行环境

- Python 3.10+
- FFmpeg（推荐加入系统 `PATH`）
- 使用画面歌词辅助时需要 RapidOCR 与 ONNX Runtime
- 首次使用所选模型时需要联网自动下载
- 8GB以上的运行内存（使用large v3时）

Release 用户和源码用户都需要安装识别依赖：

```powershell
pip install -r requirements.txt
```

有 NVIDIA 显卡时需要安装与本机 CUDA 环境匹配的 PyTorch/CTranslate2 运行库。没有可用 GPU 时，Whisper 使用 CPU `int8`，歌词声学对齐使用 FP32。速度虽然会慢一些，但也属于能接受的范畴。

## 歌词格式

### 逐行日中双语（精确模式，推荐）

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

**连续两个及以上空行会被当作明确的块分隔符**；单个空行只用于歌词段落排版并会被忽略。没有双空行时，程序继续通过日文假名位置兼容旧歌词文件。两段行数相同时会按行配对。行数不一致时，程序会给出警告。

### 纯中文（不推荐）

```text
在世界终结之处
诞生的光芒
```

精度很差。歌词中存在足够英文/数字短语时会使用这些内容作为低置信度顺序锚点，其余行仅在相邻锚点间补齐；否则只能利用声音片段边界估算时间。如非实在没办法尽量不要使用。

## 界面与命令行

### 界面功能

界面支持：

- 单文件和批量处理
- 视频和歌词拖拽功能
- 语音模型与歌词声学对齐预设及适用场景说明
- Demucs 人声或原始混合音频
- 仅生成 SRT
- 强制重新运行 Whisper
- 可选的 OCR 画面歌词辅助校准
- 启用或禁用 VAD
- 固定显示当前阶段、进度条和已运行时间

GUI 启动处理前会扫描项目虚拟环境、`PATH` 和 Windows Python Launcher 注册的解释器，并逐一检查 Python 版本及 `faster_whisper`、`rapidfuzz`、`demucs` 等模块。选择 Demucs 人声时，缺少 `demucs` 的解释器不会被采用；开启画面歌词辅助时还会检查 `rapidocr` 与 `onnxruntime`。没有合格环境时会在提取音频前停止并列出缺失模块。

可使用环境变量 `SUBTITLE_SYNC_PYTHON` 指定专用解释器。

### 自动运行诊断

GUI 会把每次运行的完整输出、实际选中的 Python、候选环境检查结果、命令、退出码和时间自动保存到：

```text
logs/latest.log
logs/latest.json
logs/run-YYYYMMDD-HHMMSS.log
logs/run-YYYYMMDD-HHMMSS.json
```

从 `release/SubtitleSync.exe` 启动时，日志位于 `release/logs/`。排查问题时无需复制 GUI 中的输出，直接检查 `latest.log` 和 `latest.json` 即可。

### 单文件

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt
```

先检查字幕和报告，不烧录视频：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --no-burn
```

`--no-burn` 只跳过最后的 FFmpeg 视频重编码；首次运行仍需进行音频提取、可选的 Demucs 人声分离和 Whisper 识别，因此不会立刻完成。识别结果写入 `.whisper.json` 后，只要视频、VAD、音源和识别配置不变，更换歌词或调整对轴算法都会直接复用缓存。GUI 会持续显示当前阶段和已运行时间，完整输出另存到日志文件。

使用原始混合音频识别，跳过 Demucs：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --audio-source original
```

使用速度更快的 Turbo 模型：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --model large-v3-turbo
```

忽略已有转录缓存并重新识别：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --refresh-transcript
```

视频本身带有歌词或硬字幕时，开启画面文字辅助：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --ocr-assist
```

该模式默认每 0.5 秒扫描一次全画面，首次运行会多花数分钟，并生成 `.ocr.json` 画面文字缓存；之后可直接复用。只有检测到足够多的高分歌词锚点才会修改时间轴，否则自动保留 Whisper 结果。RapidOCR 的模型随 Python 包安装，无需让用户填写模型参数，支持情况见 [RapidOCR 模型列表](https://rapidai.github.io/RapidOCRDocs/main/model_list/)。

调整文本锚点最低相似度：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --min-match-score 60
```

### 批量处理

视频和歌词文件需要同名、扩展名不同：

```text
videos/song_a.mp4
lyrics/song_a.txt
```

```powershell
python subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics
```

## 输出文件

假设输出视频为 `song_subtitled.mp4`：

| 文件 | 用途 |
|---|---|
| `song_subtitled.mp4` | 烧入硬字幕的视频；使用 `--no-burn` 时不生成 |
| `song_subtitled.srt` | 始终生成的字幕文件 |
| `song_subtitled.alignment.json` | 每行时间、匹配文字、文本/OCR 分数、Whisper 词概率、状态和待检查行 |
| `song_subtitled.whisper.json` | Whisper 原始片段与逐词时间缓存 |
| `song_subtitled.ocr.json` | 开启画面歌词辅助时生成的 OCR 扫描缓存 |

`alignment.json` 中的主要状态：

- `matched`：文本匹配分数较高
- `low_confidence`：达到锚点阈值，但建议复核
- `latin_anchor`：纯中文歌词中匹配到的英文/数字弱锚点，仍建议复核
- `interpolated`：Whisper 漏识别，时间由相邻锚点局部补齐
- `timing_fallback`：没有日文原文，只能按声音边界估算

修改歌词、对轴算法或相似度阈值时可以直接复用 `.whisper.json`，无需重新运行 Demucs 和 Whisper。OCR 扫描结果也会单独复用 `.ocr.json`。输入视频、模型、VAD、音源或识别配置变化时，相应缓存会自动失效。

## 项目结构

```text
Subtitle-tool/
├─ subtitle_sync.py          核心处理与命令行入口
├─ subtitle_sync_gui.py      PySide6 图形界面
├─ asr_models.py             语音模型与权重版本
├─ lyric_alignment/          歌词声学对齐核心、模型资源与输出流程
├─ requirements.txt          识别和视频处理依赖
├─ requirements-gui.txt      源码运行与打包 GUI 的依赖
├─ requirements-align.txt    可选歌词声学对齐运行依赖
├─ build.spec                PyInstaller 打包配置
├─ tests/                    对轴、缓存和 GUI 测试
├─ tools/benchmark_asr.py     独立转录文字与时间误差评测
├─ tools/benchmark_lyrics.py  歌词声学对齐多曲回归与逐句误差评测
├─ dist/                     打包输出（不提交到 Git）
│  └─ SubtitleSync.exe       可单独发布的 Windows 图形界面程序
├─ LICENSE                   MIT License
├─ THIRD_PARTY_NOTICES.md     第三方模型与许可
└─ README.md                 使用说明
```
## 打包说明

```powershell
pip install -r requirements-gui.txt
pyinstaller build.spec
```

打包后的 `SubtitleSync.exe` 内置 PySide6 界面及处理脚本，但不内置语音识别、Demucs、RapidOCR 运行库和模型。目标机器仍需安装可用的 Python 3.10+ 及运行依赖；歌词声学对齐使用 `requirements-align.txt`。EXE 会优先使用同目录的 `subtitle_sync.py`；更新外置脚本时应一并更新 `asr_models.py` 和 `lyric_alignment/`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖文本匹配、漏识别局部补齐、重复副歌顺序、纯中文回退、缓存校验、对轴报告、Python 环境选择和 GUI 日志精简规则。

## 许可证

本项目自身代码采用 [MIT License](LICENSE)。发布源码、Release 和派生版本时应保留许可证与版权声明。第三方模型与资源分别遵守其上游许可；MMS 日语卡拉 OK 对齐与 TIFA 预训练权重使用 CC BY-NC-SA 4.0，不属于本项目 MIT 授权范围，详见 [第三方许可](THIRD_PARTY_NOTICES.md)。
