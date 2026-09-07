# 日语歌曲字幕对轴工具

从日语歌曲视频提取音频，识别日文歌声，并把逐行中文字幕生成 SRT 或烧入视频。

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
- 先自动生成初稿，再结合 `alignment.json` 和播放器进行人工复核。

局限：

- **自动对轴结果仅供人工复核，不能直接交付；即使使用 `Large v3`，目前算法下个别字幕仍可能有 0–5 秒误差。**
- 不会把日语自动翻译成中文，必须提供准备好的中文字幕或歌词。
- 识别语言固定为日语。Whisper 本身支持其他语言，但其他语言的歌词解析与匹配效果尚未经过实测。
- 歌唱、和声、多人重叠、现场噪声、混响和重伴奏都会降低识别与文本锚点覆盖率。
- 没有匹配到的歌词行会在相邻锚点之间补时间，不能保证达到逐帧精度；锚点覆盖率低时必须复核。
- OCR 只能利用视频里确实存在的可读文字；无画面歌词、纯装饰文字、严重旋转或故意提前/延后的歌词可能无效或产生误差。
- 不处理说话人区分、字幕断句改写、翻译、特效字幕或复杂排版。

非音乐视频通常直接选择“原始音频”即可；“分离人声”主要用于伴奏较重的歌曲，MV等场景推荐分离人声。

## 语音模型

界面只提供四个预设，不需要填写模型名称或识别参数：

| 选项 | 适合内容 | 取舍 |
|---|---|---|
| `Large v3`（默认） | 日语歌曲、现场、背景声复杂或音质一般的视频 | 精度优先，速度最慢，资源占用最高 |
| `Large v3 Turbo` | 访谈、解说、Vlog 和一般视频 | 接近 Large v3，处理明显更快 |
| `Medium` | 较清晰的语音、硬件资源有限的电脑 | 速度与精度折中 |
| `Small` | 清晰单人语音、快速草稿、CPU 环境 | 最轻量，不建议用于歌曲和嘈杂音频 |

第一次使用某个模型时，faster-whisper 会从 Hugging Face 自动下载并保存在本机缓存中，之后无需重复下载。切换模型会使当前输出旁的 `.whisper.json` 转录缓存失效并重新识别；模型权重缓存不会被删除。

模型能力和相对速度可参考 [OpenAI Whisper 模型表](https://github.com/openai/whisper#available-models-and-languages)，自动下载行为见 [faster-whisper 使用说明](https://github.com/SYSTRAN/faster-whisper#model-conversion)。

## 对轴方式

推荐使用逐行双语歌词：

```text
世界の終わりで|在世界终结之处
生まれた光|诞生的光芒
君を探して|寻找着你
```

处理流程：

```text
视频 → FFmpeg 提取音频 → Demucs 人声分离（可选）
     → faster-whisper 所选模型逐词识别
     → 日文歌词与识别文本进行顺序受限匹配
     → 高分匹配作为时间锚点，漏识别行在相邻锚点间局部补齐
     → 可选 OCR 扫描画面歌词，校正局部漂移和明显错位边界
     → SRT + 对轴报告 → 可选 FFmpeg 硬字幕烧入
```

有日文原文时，程序会在 Whisper 独立识别完成后比较日文歌词与识别文字，不会把整段歌词作为识别提示。没有日文原文时，程序优先使用中文字幕中的英文/数字短语建立局部锚点；可用锚点不足时才使用低置信度时间比例回退。

## 运行环境

- Python 3.10+
- FFmpeg（推荐加入系统 `PATH`）
- 使用画面歌词辅助时需要 RapidOCR 与 ONNX Runtime（已列入 `requirements.txt`）
- 首次使用所选模型时需要联网自动下载
- 模型越大，对内存或显存的要求越高

Release 用户和源码用户都需要安装识别依赖：

```powershell
pip install -r requirements.txt
```

有 NVIDIA 显卡时需要安装与本机 CUDA 环境匹配的 PyTorch/CTranslate2 运行库。没有可用 GPU 时会使用 CPU `int8`。

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

**连续两个及以上空行会被当作明确的块分隔符**；单个空行只用于歌词段落排版并会被忽略。没有双空行时，程序继续通过日文假名位置兼容旧歌词文件。两段行数相同时会按行配对。行数不一致时，程序会给出警告，缺少日文原文的行将通过相邻文本锚点补齐并列入待检查项。

### 纯中文（低置信度回退）

```text
在世界终结之处
诞生的光芒
```

纯中文无法与日语识别文本直接比较。歌词中存在足够英文/数字短语时会使用这些内容作为低置信度顺序锚点，其余行仅在相邻锚点间补齐；否则只能利用声音片段边界估算时间。

## 界面与命令行

### 界面功能

界面支持：

- 单文件和批量处理
- 视频和歌词拖拽功能
- 四档语音模型预设及适用场景说明
- Demucs 人声或原始混合音频
- 仅生成 SRT
- 强制重新运行 Whisper
- 可选的 OCR 画面歌词辅助校准
- 启用或禁用 VAD
- 固定显示当前阶段、进度条和已运行时间

界面只显示阶段、匹配摘要和输出路径。Whisper 的逐段识别文字仍会保存在完整日志中，不再占满窗口。

GUI 启动处理前会扫描项目虚拟环境、`PATH` 和 Windows Python Launcher 注册的解释器，并逐一检查 Python 版本及 `faster_whisper`、`rapidfuzz`、`demucs` 等模块。选择 Demucs 人声时，缺少 `demucs` 的解释器不会被采用；开启画面歌词辅助时还会检查 `rapidocr` 与 `onnxruntime`。没有合格环境时会在提取音频前停止并列出缺失模块。

可使用环境变量 `SUBTITLE_SYNC_PYTHON` 指定专用解释器。指定环境仍须通过版本和依赖检查。

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
├─ requirements.txt          识别和视频处理依赖
├─ requirements-gui.txt      源码运行与打包 GUI 的依赖
├─ build.spec                PyInstaller 打包配置
├─ tests/                    对轴、缓存和 GUI 测试
├─ release/                  本地 Release 准备目录
│  ├─ SubtitleSync.exe       可单独发布的 Windows 图形界面程序
│  ├─ subtitle_sync.py       可选的外置核心脚本
│  └─ requirements.txt       识别环境依赖清单
├─ LICENSE                   MIT License
└─ README.md                 使用说明
```
## 打包说明

```powershell
pip install -r requirements-gui.txt
pyinstaller build.spec
```

打包后的 `SubtitleSync.exe` 内置 PySide6 界面，但不内置 Whisper、Demucs、RapidOCR 运行库和模型。目标机器仍需安装可用的 Python 3.10+ 及 `requirements.txt` 中的依赖。EXE 会优先使用同目录的 `subtitle_sync.py`，方便更新核心算法。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖文本匹配、漏识别局部补齐、重复副歌顺序、纯中文回退、缓存校验、对轴报告、Python 环境选择和 GUI 日志精简规则。

## 许可证

本项目采用 [MIT License](LICENSE)。发布源码、Release 和派生版本时应保留许可证与版权声明。
