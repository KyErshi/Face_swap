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


def color_transfer(
    source: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """颜色迁移 — 将 source 的配色风格转移到 target

    基于 Reinhard et al. 2001 的 LAB 空间颜色迁移算法。
    通过匹配 LAB 三个通道的均值和标准差，使 source 的色调/饱和度/亮度拟合 target。

    Args:
        source: 源图像 (BGR)，其颜色将被修改
        target: 目标图像 (BGR)，提供参考颜色统计
        mask: 可选 — 仅对 mask 区域计算统计 (灰度图, 0-255)

    Returns:
        颜色迁移后的 BGR 图像 (仅 source 区域被修改)
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = src_lab.copy()

    for c in range(3):
        if mask is not None:
            # 仅在 mask 区域计算统计
            mask_f = mask.astype(np.float32) / 255.0
            src_mean = np.sum(src_lab[..., c] * mask_f) / (np.sum(mask_f) + 1e-6)
            src_std = np.sqrt(
                np.sum(mask_f * (src_lab[..., c] - src_mean) ** 2) / (np.sum(mask_f) + 1e-6)
            )
            tgt_mean = np.sum(tgt_lab[..., c] * mask_f) / (np.sum(mask_f) + 1e-6)
            tgt_std = np.sqrt(
                np.sum(mask_f * (tgt_lab[..., c] - tgt_mean) ** 2) / (np.sum(mask_f) + 1e-6)
            )
        else:
            src_mean, src_std = src_lab[..., c].mean(), src_lab[..., c].std()
            tgt_mean, tgt_std = tgt_lab[..., c].mean(), tgt_lab[..., c].std()

        # 标准化: (src - src_mean) / src_std * tgt_std + tgt_mean
        result[..., c] = (src_lab[..., c] - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    # 仅修改源图中有 mask 的区域 (保留背景)
    if mask is not None:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        output = (source.astype(np.float32) * (1 - mask_3ch) + result_bgr.astype(np.float32) * mask_3ch)
        return np.clip(output, 0, 255).astype(np.uint8)

    return result_bgr


def transfer_skin_texture(
    result_img: np.ndarray,
    source_img: np.ndarray,
    source_face_roi: tuple,
    strength: float = 0.35,
    sigma: float = 2.0,
    landmarks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """迁移源人脸皮肤纹理到换脸结果

    只在 LAB 空间的 L 通道操作，确保不改变颜色。
    高斯差分提取源图皮肤纹理，用局部对比度自适应调节强度，
    并在皮肤区域使用柔和遮罩。

    Args:
        result_img: 换脸后的 BGR 图像
        source_img: 原始源 BGR 图像
        source_face_roi: (x1, y1, x2, y2) 人脸框
        strength: 基础纹理强度
        sigma: 高斯模糊 sigma (越小纹理越细，越大纹理越粗)
        landmarks: 关键点 (用于排除眼/嘴区域), shape (N,2)

    Returns:
        叠加纹理后的 BGR 图像
    """
    try:
        x1, y1, x2, y2 = source_face_roi
        rh, rw = result_img.shape[:2]
        sh, sw = source_img.shape[:2]
        margin = int(max(x2 - x1, y2 - y1) * 0.2)

        # 各自按各自的尺寸裁切 (源图和结果图可能尺寸不同)
        sx1 = max(0, x1 - margin)
        sy1 = max(0, y1 - margin)
        sx2 = min(sw, x2 + margin)
        sy2 = min(sh, y2 + margin)
        rx1 = max(0, x1 - margin)
        ry1 = max(0, y1 - margin)
        rx2 = min(rw, x2 + margin)
        ry2 = min(rh, y2 + margin)
        roi_h_r, roi_w_r = ry2 - ry1, rx2 - rx1
        if roi_w_r <= 20 or roi_h_r <= 20:
            return result_img

        src_roi = source_img[sy1:sy2, sx1:sx2].astype(np.float32)
        res_roi = result_img[ry1:ry2, rx1:rx2].astype(np.float32)
        # 统一以结果图尺寸为准
        if src_roi.shape[:2] != (roi_h_r, roi_w_r):
            src_roi = cv2.resize(src_roi, (roi_w_r, roi_h_r),
                                 interpolation=cv2.INTER_LANCZOS4)

        # 转到 LAB 空间，只操作 L 通道
        src_lab = cv2.cvtColor(src_roi.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(float)
        res_lab = cv2.cvtColor(res_roi.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(float)

        src_l = src_lab[..., 0]
        res_l = res_lab[..., 0]

        # 多尺度纹理提取 (细纹理 + 中等纹理)
        blur1 = cv2.GaussianBlur(src_l, (0, 0), sigma)
        texture_fine = src_l - blur1  # 细纹理 (sigma=2)

        blur2 = cv2.GaussianBlur(src_l, (0, 0), sigma * 2)
        texture_medium = blur1 - blur2  # 中等纹理

        # 合并纹理: 细纹理权重高，中等纹理权重低
        texture = texture_fine * 0.7 + texture_medium * 0.3

        # 自适应强度: 在纹理强的区域降低强度 (防止过锐)
        local_contrast = cv2.GaussianBlur(np.abs(texture), (0, 0), 5)
        adaptive_strength = strength / (1.0 + local_contrast / 10.0)
        adaptive_strength = np.clip(adaptive_strength, 0, strength)

        # 皮肤遮罩: 排除眼睛和嘴巴区域
        skin_mask = np.ones((roi_h_r, roi_w_r), dtype=float)

        if landmarks is not None and len(landmarks) >= 5:
            lmk = landmarks.astype(np.int32)
            # 排除眼睛区域 (用椭圆)
            for pt_idx, scale in [(0, 2.5), (1, 2.5)]:  # 左右眼
                if pt_idx < len(lmk):
                    cx = lmk[pt_idx][0] - rx1
                    cy = lmk[pt_idx][1] - ry1
                    if 0 <= cx < roi_w_r and 0 <= cy < roi_h_r:
                        r = int(max(roi_w_r, roi_h_r) * 0.04)
                        cv2.ellipse(skin_mask, (cx, cy),
                                    (int(r * 1.5), r), 0, 0, 360, 0.15, -1)
            # 排除嘴巴区域
            for pt_idx in [3, 4]:  # 左右嘴角
                if pt_idx < len(lmk):
                    cx = lmk[pt_idx][0] - rx1
                    cy = lmk[pt_idx][1] - ry1
                    if 0 <= cx < roi_w_r and 0 <= cy < roi_h_r:
                        cv2.ellipse(skin_mask, (cx, cy),
                                    (int(roi_w_r * 0.08), int(roi_h_r * 0.04)),
                                    0, 0, 360, 0.2, -1)

        # 边缘降权
        center_weight = cv2.GaussianBlur(
            np.ones((roi_h_r, roi_w_r), dtype=float), (0, 0),
            min(roi_h_r, roi_w_r) * 0.15
        )
        center_weight = (center_weight - center_weight.min()) / (center_weight.max() - center_weight.min() + 1e-6)
        center_weight = center_weight * 0.3 + 0.7

        skin_mask = skin_mask * center_weight
        skin_mask = skin_mask.clip(0, 1)

        # 对 L 通道施加纹理
        enhanced_l = res_l + texture * adaptive_strength * skin_mask
        enhanced_l = enhanced_l.clip(0, 255)

        # 合成回 LAB → BGR
        res_lab[..., 0] = enhanced_l
        enhanced_bgr = cv2.cvtColor(res_lab.clip(0, 255).astype(np.uint8),
                                     cv2.COLOR_LAB2BGR)

        # 通过皮肤遮罩混合 (只改变皮肤区域)
        skin_mask_3ch = np.stack([skin_mask] * 3, axis=-1)
        result = result_img.copy().astype(float)
        result[ry1:ry2, rx1:rx2] = (
            enhanced_bgr.astype(float) * skin_mask_3ch
            + res_roi * (1.0 - skin_mask_3ch)
        )
        return result.clip(0, 255).astype(np.uint8)

    except Exception as e:
        logger.warning(f"皮肤纹理迁移失败: {e}")
        return result_img


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
