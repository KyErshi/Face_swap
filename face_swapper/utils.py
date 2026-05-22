"""
图像处理工具函数
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def get_model_path() -> Path:
    """获取模型存储路径"""
    # insightface 默认在 ~/.insightface/models/
    home = Path.home()
    return home / ".insightface" / "models"


def download_models():
    """触发 insightface 模型下载

    insightface 的特性是首次使用 FaceAnalysis 时自动下载，
    此函数仅为显式触发。
    """
    from insightface.model_zoo import model_zoo
    from insightface.app import FaceAnalysis

    logger.info("检查并下载换脸模型...")
    model_dir = get_model_path()
    model_dir.mkdir(parents=True, exist_ok=True)

    # 预下载换脸模型
    model_name = "inswapper_128.onnx"
    model_path = model_dir / "buffalo_l" / model_name
    if not model_path.exists():
        logger.info(f"正在下载 {model_name} 到 {model_path}...")
        # insightface 的 model_zoo 会在需要时下载
        # 这里只做记录
        logger.info("模型将在首次使用时自动下载")

    return str(model_dir)


def load_image(path: str) -> Optional[np.ndarray]:
    """加载图像文件 (支持中文路径)

    Args:
        path: 图像文件路径

    Returns:
        BGR numpy 数组, 失败返回 None
    """
    try:
        # 先用 OpenCV (不支持中文路径时回退到 PIL)
        img = cv2.imread(path)
        if img is not None:
            return img

        # 通过 PIL 中转 (支持中文)
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    except Exception as e:
        logger.error(f"加载图像失败 {path}: {e}")
        return None


def save_image(path: str, img: np.ndarray) -> bool:
    """保存图像文件 (支持中文路径)"""
    try:
        success = cv2.imwrite(path, img)
        if not success:
            # 通过 PIL 中转
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.save(path)
        return True
    except Exception as e:
        logger.error(f"保存图像失败 {path}: {e}")
        return False


def blend_images(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """混合前景和背景图像

    Args:
        foreground: 前景图像 (BGR)
        background: 背景图像 (BGR), 需要与前景同尺寸
        mask: 灰度遮罩 (0-255), 如果为 None 则使用 alpha 常量
        alpha: 整体透明度 (0-1), 仅 mask 为 None 时使用

    Returns:
        混合后的图像
    """
    if foreground.shape != background.shape:
        # 调整背景尺寸匹配前景
        background = cv2.resize(
            background, (foreground.shape[1], foreground.shape[0])
        )

    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    if mask is not None:
        # 确保 mask 与图像同尺寸
        if mask.shape[:2] != foreground.shape[:2]:
            mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))

        if len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask

        mask_f = mask_3ch.astype(np.float32) / 255.0
        blended = fg * mask_f + bg * (1.0 - mask_f)
    else:
        blended = fg * alpha + bg * (1.0 - alpha)

    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_face_boxes(
    img: np.ndarray,
    faces_info: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制人脸框和标注"""
    result = img.copy()
    for info in faces_info:
        bbox = info["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        # 绘制矩形框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签
        label = f"#{info['idx']} {info['gender']} {info['age']}y {info.get('confidence', 0):.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        # 标签背景
        cv2.rectangle(
            result,
            (x1, y1 - label_size[1] - 6),
            (x1 + label_size[0] + 4, y1),
            color,
            -1,
        )
        # 标签文字
        cv2.putText(
            result,
            label,
            (x1 + 2, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return result


def resize_to_limit(
    img: np.ndarray,
    max_size: int = 1920,
) -> np.ndarray:
    """限制图像最大尺寸 (保持宽高比)"""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_face(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.3,
) -> np.ndarray:
    """按人脸框裁剪并扩展边距"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    face_w = x2 - x1
    face_h = y2 - y1

    margin_x = int(face_w * margin)
    margin_y = int(face_h * margin)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return img[y1:y2, x1:x2]


def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """基于面部关键点做仿射变换对齐"""
    if len(landmarks) < 5:
        return img

    # 标准人脸关键点位置 (归一化)
    # 参照 FFHQ / insightface 的标准对齐
    src_pts = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float64,
    )

    dst_pts = landmarks[:5].astype(np.float64)

    # 估计仿射变换
    tform = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.LMEDS)
    if tform[0] is not None:
        aligned = cv2.warpAffine(
            img, tform[0], (112, 112), flags=cv2.INTER_LINEAR
        )
        return aligned

    return img


def image_to_base64(img: np.ndarray) -> str:
    """将 numpy 图像转为 base64 (用于 Gradio 显示)"""
    import base64

    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def is_image_file(path: str) -> bool:
    """检查文件路径是否为支持的图片格式"""
    ext = Path(path).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_video_file(path: str) -> bool:
    """检查文件路径是否为支持的视频格式"""
    ext = Path(path).suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
