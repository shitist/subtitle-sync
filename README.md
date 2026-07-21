# 日语歌曲中文字幕自动对轴工具

从日语歌曲视频提取音频，识别日文歌声，并把逐行中文字幕生成 SRT 或烧入视频。

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
     → faster-whisper large-v3 逐词识别
     → 日文歌词与识别文本进行顺序受限匹配
     → 高分匹配作为时间锚点，漏识别行在相邻锚点间局部补齐
     → SRT + 对轴报告 → 可选 FFmpeg 硬字幕烧入
```

有日文原文时，程序会在 Whisper 独立识别完成后比较日文歌词与识别文字，不会把整段歌词作为识别提示。没有日文原文时，程序优先使用中文字幕中的英文/数字短语建立局部锚点；可用锚点不足时才使用低置信度时间比例回退。

## 环境要求

- Python 3.10+
- FFmpeg（推荐加入系统 `PATH`）
- 首次使用 `large-v3` 时需要下载约 3 GB 模型

安装 Python 依赖：

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

中文整块与日文整块之间保留至少两个完整空行：

```text
在世界终结之处
诞生的光芒


世界の終わりで
生まれた光
```

连续两个及以上空行会被当作明确的块分隔符；单个空行只用于歌词段落排版并会被忽略。没有双空行时，程序继续通过日文假名位置兼容旧歌词文件。两段行数相同时会按行配对。行数不一致时，程序会给出警告，缺少日文原文的行将通过相邻文本锚点补齐并列入待检查项。

### 纯中文（低置信度回退）

```text
在世界终结之处
诞生的光芒
```

纯中文无法与日语识别文本直接比较。歌词中存在足够英文/数字短语时会使用这些内容作为低置信度顺序锚点，其余行仅在相邻锚点间补齐；否则只能利用声音片段边界估算时间。

## 使用方法

### 图形界面

```powershell
python subtitle_sync_gui.py
```

界面支持：

- 单文件和批量处理
- Demucs 人声或原始混合音频
- 仅生成 SRT，不烧录视频
- 强制重新运行 Whisper
- 启用或禁用 VAD

GUI 启动处理前会扫描项目虚拟环境、`PATH` 和 Windows Python Launcher 注册的解释器，并逐一检查 Python 版本及 `faster_whisper`、`rapidfuzz`、`demucs` 等模块。选择 Demucs 人声时，缺少 `demucs` 的解释器不会被采用；没有合格环境时会在提取音频前停止并列出缺失模块。

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

`--no-burn` 只跳过最后的 FFmpeg 视频重编码；首次运行仍需进行音频提取、可选的 Demucs 人声分离和 Whisper 识别，因此不会立刻完成。识别结果写入 `.whisper.json` 后，只要视频、歌词、VAD 和音源设置不变，后续调整对轴会直接复用缓存。GUI 在长步骤中每 30 秒写入一次运行心跳，并实时刷新已经产生的 Whisper 片段。

使用原始混合音频识别，跳过 Demucs：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --audio-source original
```

忽略已有转录缓存并重新识别：

```powershell
python subtitle_sync.py --video song.mp4 --lyrics song.txt --refresh-transcript
```

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
| `song_subtitled.alignment.json` | 每行时间、匹配文字、文本分数、Whisper 词概率、状态和待检查行 |
| `song_subtitled.whisper.json` | Whisper 原始片段与逐词时间缓存 |

`alignment.json` 中的主要状态：

- `matched`：文本匹配分数较高
- `low_confidence`：达到锚点阈值，但建议复核
- `latin_anchor`：纯中文歌词中匹配到的英文/数字弱锚点，仍建议复核
- `interpolated`：Whisper 漏识别，时间由相邻锚点局部补齐
- `timing_fallback`：没有日文原文，只能按声音边界估算

修改歌词、对轴算法或相似度阈值时可以直接复用 `.whisper.json`，无需重新运行 Demucs 和 Whisper。输入视频、VAD、音源或识别配置变化时缓存会自动失效。

## 打包说明

```powershell
pyinstaller build.spec
```

打包后的 `SubtitleSync.exe` 是轻量 GUI 启动器，不内置 Whisper、Demucs 和模型。目标机器仍需安装可用的 Python 3.10+ 及 `requirements.txt` 中的依赖。EXE 会优先使用同目录的 `subtitle_sync.py`，方便更新核心算法。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖文本匹配、漏识别局部补齐、重复副歌顺序、纯中文回退、缓存校验和对轴报告。
