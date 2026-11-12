# 第三方许可

Subtitle Sync 项目代码采用 [MIT License](LICENSE)。

## 代码与运行库

| 资源 | 来源 | 许可 |
|---|---|---|
| PySide6 / Qt | [Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html) | LGPL-3.0 / GPL；各组件许可随 PySide6 安装包提供 |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD 3-Clause |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) | Apache-2.0 |
| faster-whisper | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) | MIT |
| WhisperX 3.8.6 | [m-bain/whisperX](https://github.com/m-bain/whisperX/blob/v3.8.6/LICENSE) | BSD 2-Clause；本地全文见 [assets/WHISPERX_LICENSE.txt](assets/WHISPERX_LICENSE.txt) |
| Demucs | [facebookresearch/demucs](https://github.com/facebookresearch/demucs/blob/main/LICENSE) | MIT |
| RapidOCR | [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE) | Apache-2.0 |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | MIT |
| num2words | [savoirfairelinux/num2words](https://github.com/savoirfairelinux/num2words/blob/master/COPYING) | LGPL-2.1 |
| TIFA 代码 | [openvpi/TIFA](https://github.com/openvpi/TIFA/blob/7b210ca36ff5a2b79187fb1727af58fc977cc301/LICENSE) | MIT |
| g2pflow 0.3.1 | [openvpi/g2pflow](https://github.com/openvpi/g2pflow/tree/v0.3.1) | 主代码 MIT；安装包内其他代码和词典包含 Apache-2.0、CC BY-SA 3.0、CC BY-SA 4.0，见包内许可文件 |
| FFmpeg | [FFmpeg 许可](https://ffmpeg.org/legal.html) | LGPL-2.1+；启用 GPL 组件的构建采用 GPL，具体见所用二进制的构建信息 |

## 模型权重

### MMS+TIFA 歌词声学对齐

| 资源 | 来源 | 许可 |
|---|---|---|
| MMS 日语卡拉 OK 对齐权重 | [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) | CC BY-NC-SA 4.0 |
| TIFA-1.0 预训练权重 | [TIFA v1.0.0](https://github.com/openvpi/TIFA/releases/tag/v1.0.0) | CC BY-NC-SA 4.0 |

MMS 与 TIFA 权重要求署名、非商业性使用和相同方式共享。许可全文：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

### Whisper 与 WhisperX 路线

| 资源 | 来源 | 许可 |
|---|---|---|
| Whisper 代码与权重 | [openai/whisper](https://github.com/openai/whisper#license)；运行时使用 [SYSTRAN 转换权重](https://huggingface.co/Systran) | MIT |
| 日语对齐权重 | [jonatasgrosman/wav2vec2-large-xlsr-53-japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) | Apache-2.0 |
| 中文对齐权重 | [jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn) | Apache-2.0 |
| 英语对齐权重 | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | Apache-2.0 |
| 土耳其语对齐权重 | [mpoyraz/wav2vec2-xls-r-300m-cv7-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish) | CC BY 4.0 |
| 越南语对齐权重 | [nguyenvulebinh/wav2vec2-base-vi-vlsp2020](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vi-vlsp2020) | CC BY-NC 4.0 |
| 瑞典语对齐权重 | [KBLab/wav2vec2-large-voxrex-swedish](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish) | CC0 1.0 |

其余语言的模型仓库列在 [speech_catalog.py](speech_catalog.py) 的 `ALIGN_MODELS` 中，各模型的许可见对应 Hugging Face 模型卡。程序按所选语言加载模型。

## 词典与字体

| 资源 | 来源 | 许可 |
|---|---|---|
| 日语词典 | [UniDic](https://clrd.ninjal.ac.jp/unidic/) | 以下载词典附带的版权及许可文件为准 |
| 英语读音模型与发音词典 | TIFA-1.0 下载包中的 LstmG2p-Eng、CMU 字典 | 以原始下载包及 [上游 G2P 资源说明](https://github.com/openvpi/TIFA/blob/main/docs/G2P.md) 为准 |
| Noto Sans CJK 字体 | [Noto CJK Sans 2.004](https://github.com/notofonts/noto-cjk/releases/tag/Sans2.004) | SIL Open Font License 1.1；本地全文见 [assets/fonts/OFL.txt](assets/fonts/OFL.txt) |
