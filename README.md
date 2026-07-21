# 日语歌曲中文字幕自动对轴工具

自动将纯文本中文歌词与日语歌曲视频对轴并烧入硬字幕。

---

## 工作原理

```
MP4 视频
  ↓  FFmpeg        提取音频
  ↓  Demucs        人声分离（去除背景音乐，大幅提升识别准确率）
  ↓  Whisper       日语识别，获取逐句时间轴
  ↓  对轴算法      时间轴映射到中文歌词
  ↓  FFmpeg        硬字幕烧入
  ↓
带中文字幕的 MP4
```

---

## 环境要求

### 系统依赖（需手动安装）

| 工具 | 说明 | 下载 |
|------|------|------|
| **Python 3.10+** | 运行脚本 | https://python.org |
| **FFmpeg** | 音视频处理 | https://ffmpeg.org/download.html |

> ⚠️ **FFmpeg 安装后需添加到系统 PATH**，在 PowerShell 中运行 `ffmpeg -version` 确认。

### Python 依赖

```powershell
pip install -r requirements.txt
```

> 首次运行时 Whisper 会自动下载 large-v3 模型（约 **3GB**），请确保磁盘空间充足。

### GPU 加速（可选，推荐）

有 NVIDIA 显卡时处理速度可提升 **5~10 倍**，请到 https://pytorch.org/get-started/locally/ 获取对应 CUDA 版本的安装命令。

---

## 使用方法

### 单文件处理

```powershell
# 基本用法
python subtitle_sync.py --video song.mp4 --lyrics song.txt

# 指定输出文件
python subtitle_sync.py --video song.mp4 --lyrics song.txt --output out.mp4

# 同时保留 .srt 字幕文件
python subtitle_sync.py --video song.mp4 --lyrics song.txt --keep-srt
```

### 批量处理

视频文件和歌词文件需**同名**（扩展名不同）：

```
videos/
  song_a.mp4
  song_b.mp4

lyrics/
  song_a.txt    ← 对应 song_a.mp4
  song_b.txt    ← 对应 song_b.mp4
```

```powershell
# 批量处理，输出到 videos/output/
python subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics

# 指定输出目录
python subtitle_sync.py --batch --video-dir ./videos --lyrics-dir ./lyrics --output-dir ./output
```

---

## 歌词文件格式

UTF-8 编码的纯文本文件，**每行一句**（空行自动跳过）：

```
这条街的风景
今天也很美丽
你的笑容
在我脑海中回荡
```

> **关键**：歌词行数应与日语原文一一对应。行数无需精确匹配——工具会自动按比例合并/拆分片段。

---

## 对轴算法说明

| 情况 | 处理方式 |
|------|---------|
| Whisper 片段数 == 歌词行数 | 直接 1:1 映射（最理想） |
| Whisper 片段数 > 歌词行数 | 按比例合并相邻片段 |
| Whisper 片段数 < 歌词行数 | 将歌词均分到各片段内 |

---

## 常见问题

**Q: 时间轴偏差较大怎么办？**  
A: 可以用 `--keep-srt` 保留 .srt 文件，用 [Aegisub](https://aegisub.org/) 手动微调后再用 FFmpeg 烧入。

**Q: Whisper 识别不准怎么办？**  
A: 工具已启用 VAD（语音活动检测）过滤静音段。如仍有问题，检查人声分离效果——可在 `work_dir` 中找到 Demucs 输出的 `vocals.wav` 手动试听。

**Q: 支持哪些视频格式？**  
A: 支持 `.mp4 .mkv .avi .mov .flv .webm .m4v`（批量模式下自动识别）。

**Q: 字体显示问题（方块字）？**  
A: 默认使用微软雅黑，Windows 系统已自带。Linux/macOS 用户需修改脚本中的 `FontName` 为本地可用字体。

---

## 输出说明

| 文件 | 说明 |
|------|------|
| `{原文件名}_subtitled.mp4` | 带硬字幕的视频（默认输出） |
| `{原文件名}.srt` | SRT 字幕文件（使用 `--keep-srt` 时） |
