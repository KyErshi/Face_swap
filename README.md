# 🎭 AI 换脸

基于 insightface 的高质量 AI 换脸工具，支持图片/视频换脸、GFPGAN 增强、Web UI 和命令行批量处理。

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 📷 **图片换脸** | 上传源人脸 + 目标图，一键换脸 |
| 🎬 **视频换脸** | 逐帧处理，保留原音频 |
| 🖼️ **批量处理** | 目录批量换脸 |
| 🔍 **人脸检测** | 可视化人脸位置、年龄、性别 |
| ✨ **GFPGAN 增强** | 可选面部超分辨率修复 |
| 🌐 **Web UI** | 友好的 Gradio 图形界面 |
| 🖥️ **命令行** | 适合批量/脚本处理 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 大部分依赖已预装，首次运行时会自动下载 insightface 模型（~300MB）。

### 2. 启动 Web UI

```bash
python main.py
```

浏览器访问: http://127.0.0.1:7860

### 3. 命令行使用

**单张图片换脸:**
```bash
python main.py swap source.jpg target.jpg -o result.jpg
```

**视频换脸 (前30秒):**
```bash
python main.py video source.jpg video.mp4 -o output.mp4 --seconds 30
```

**批量图片换脸:**
```bash
python main.py batch source.jpg ./input_dir/ -o ./output_dir/
```

**人脸检测:**
```bash
python main.py detect photo.jpg -o detected.jpg
```

**查看所有选项:**
```bash
python main.py --help
```

## 🖥️ 命令行参数

### `swap` — 图片换脸

| 参数 | 说明 |
|------|------|
| `source` | 源图像（提供人脸特征） |
| `target` | 目标图像（被换脸） |
| `-o` | 输出路径 (默认: output.jpg) |
| `--enhance` | 开启 GFPGAN 增强 |
| `--source-face` | 源图中第几张人脸 (默认 0) |
| `--target-face` | 目标图中第几张人脸 (默认 0) |
| `--cpu` | 强制使用 CPU |
| `--no-blend` | 禁用无缝融合 |

### `video` — 视频换脸

| 参数 | 说明 |
|------|------|
| `source` | 源图像 |
| `video` | 目标视频 |
| `-o` | 输出路径 (默认: output_video.mp4) |
| `--seconds` | 处理前 N 秒 (0=全部) |
| `--no-audio` | 不保留原音频 |
| `--enhance` | GFPGAN 增强 |

### `batch` — 批量处理

| 参数 | 说明 |
|------|------|
| `source` | 源图像 |
| `input_dir` | 输入目录 |
| `-o` | 输出目录 (默认: ./output) |

### `detect` — 人脸检测

| 参数 | 说明 |
|------|------|
| `image` | 输入图像 |
| `-o` | 输出标注图像 |

### `web` — 启动 UI

| 参数 | 说明 |
|------|------|
| `--host` | 监听地址 (默认: 127.0.0.1) |
| `--port` | 端口号 (默认: 7860) |
| `--share` | 生成公网链接 |

## ⚙️ 技术架构

```
ai-face-swap/
├── main.py                      统一入口
├── cli.py                       命令行接口
├── app.py                       Gradio Web UI
├── requirements.txt             依赖清单
├── README.md                    文档
└── face_swapper/                核心库
    ├── __init__.py              包入口
    ├── engine.py                换脸引擎 (检测+特征+替换+混合)
    ├── detector.py              人脸检测器 (多后端)
    ├── enhancer.py              GFPGAN 面部增强
    ├── video_processor.py       视频逐帧处理 + 音频合并
    └── utils.py                 图像工具函数
```

## 🧠 技术栈

- **[insightface](https://github.com/deepinsight/insightface)** — 人脸检测/识别/换脸
  - 检测模型: `buffalo_l`
  - 换脸模型: `inswapper_128.onnx`
- **[GFPGAN](https://github.com/TencentARC/GFPGAN)** — 面部超分辨率修复 (可选)
- **[Gradio](https://gradio.app)** — Web UI 框架
- **[OpenCV](https://opencv.org)** — 图像/视频处理
- **[FFmpeg](https://ffmpeg.org)** — 音视频封装

## 📋 支持的文件格式

- **图片**: JPG, JPEG, PNG, BMP, TIFF, WebP
- **视频**: MP4, AVI, MOV, MKV, WMV, FLV, WebM

## 💡 使用建议

1. **源照片**: 正脸、光照均匀、表情自然、无遮挡 → 效果最佳
2. **目标照片**: 角度和光照尽量匹配源照片
3. **视频处理**: 建议先截取 5-10 秒测试，确认效果后再处理全片
4. **GFPGAN 增强**: 提高清晰度的同时会大幅增加处理时间（每帧多 0.5-2 秒）
5. **GPU**: 有 NVIDIA GPU 时自动启用 CUDA 加速，CPU 也可运行但慢 5-10 倍

## ⚠️ 伦理声明

本工具仅供合法、道德的使用，例如：
- 个人娱乐创作
- 影视特效制作
- 学术研究

**禁止用于:**
- 未经同意的虚假内容制作
- 欺诈、诽谤、冒充他人
- 任何违反法律法规的用途

## 📄 许可

MIT License
