"""
AI 换脸引擎 — Face Swapper

基于 insightface 的高质量人脸替换库。
支持图片/视频换脸，可选 GFPGAN 增强，Gradio Web UI，CLI 批处理。
"""

from .engine import FaceSwapper, get_engine
from .detector import FaceDetector, FaceBox
from .enhancer import FaceEnhancer
from .video_processor import VideoProcessor
from . import utils

__all__ = [
    "FaceSwapper",
    "get_engine",
    "FaceDetector",
    "FaceBox",
    "FaceEnhancer",
    "VideoProcessor",
    "utils",
]

__version__ = "1.0.0"
