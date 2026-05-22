"""
核心换脸引擎
基于 insightface 模型实现人脸检测 -> 特征提取 -> 人脸替换 -> 后处理混合
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Any
from urllib.request import urlretrieve

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insightface_model

from .detector import FaceDetector
from .enhancer import FaceEnhancer
from .utils import (
    download_models,
    blend_images,
    get_model_path,
)

logger = logging.getLogger(__name__)

# 模型常量
# insightface 模型提供者
INSIGHTFACE_PROVIDERS = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "CPUExecutionProvider",
]

DETECTION_MODEL = "buffalo_l"  # insightface 内置检测+识别模型
SWAPPER_MODEL = "inswapper_128.onnx"  # 换脸模型
# inswapper 官方下载地址
SWAPPER_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
)


class FaceSwapper:
    """AI 换脸引擎

    完整流程:
    1. 初始化 FaceAnalysis (检测+关键点+特征)
    2. 加载换脸模型 (inswapper)
    3. detect_faces -> 获取源/目标所有人脸
    4. swap(source_img, target_img, source_face_idx, target_face_idx)
    5. 可选: enhance -> GFPGAN 增强
    """

    def __init__(
        self,
        det_threshold: float = 0.5,
        use_gpu: bool = True,
        model_dir: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        self.det_threshold = det_threshold
        self.use_gpu = use_gpu and self._check_gpu()
        self.providers = providers or (
            INSIGHTFACE_PROVIDERS if self.use_gpu else ["CPUExecutionProvider"]
        )

        # 模型存储目录
        if model_dir is None:
            self.model_dir = get_model_path()
        else:
            self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # insightface 会在 model_dir 的父级新建 models/ 子目录
        # 所以我们需要在 model_dir 的父级操作
        self._insightface_root = self.model_dir.parent

        self._initialized = False
        self._face_analysis: Optional[FaceAnalysis] = None
        self._swapper: Any = None
        self._enhancer: Optional[FaceEnhancer] = None

        logger.info(f"FaceSwapper 初始化 (GPU={'开启' if self.use_gpu else '关闭'})")

    @staticmethod
    def _check_gpu() -> bool:
        """检查 GPU 是否可用"""
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
                return True
        except Exception:
            pass
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if any("CUDA" in p or "TensorRT" in p for p in providers):
                return True
        except Exception:
            pass
        logger.info("未检测到 GPU，使用 CPU")
        return False

    def _ensure_swapper_model(self) -> Path:
        """确保换脸模型存在，不存在则下载

        Returns:
            模型完整路径
        """
        # 搜索可能的路径
        candidates = [
            self.model_dir / SWAPPER_MODEL,
            self.model_dir / "models" / SWAPPER_MODEL,
            self.model_dir / "buffalo_l" / SWAPPER_MODEL,
            self._insightface_root / "models" / SWAPPER_MODEL,
            self._insightface_root / "models" / "models" / SWAPPER_MODEL,
            Path.home() / ".insightface" / "models" / SWAPPER_MODEL,
            Path.home() / ".insightface" / "models" / "models" / SWAPPER_MODEL,
        ]

        for p in candidates:
            if p.exists():
                logger.info(f"找到换脸模型: {p}")
                return p

        # 下载到标准位置
        target = self.model_dir / SWAPPER_MODEL
        logger.info(f"下载换脸模型: {SWAPPER_URL}")
        logger.info(f"保存到: {target}")
        try:
            urlretrieve(SWAPPER_URL, str(target))
            logger.info("换脸模型下载完成")
            return target
        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise

    def initialize(self):
        """延迟初始化 — 加载模型（首次调用耗时）"""
        if self._initialized:
            return

        try:
            # 1. 初始化人脸分析器 (检测+关键点+特征)
            logger.info("加载人脸分析模型 (buffalo_l)...")
            self._face_analysis = FaceAnalysis(
                name=DETECTION_MODEL,
                root=str(self.model_dir),
                providers=self.providers,
            )
            self._face_analysis.prepare(
                ctx_id=0 if self.use_gpu else -1,
                det_thresh=self.det_threshold,
            )
            logger.info("人脸分析模型加载完成")

            # 2. 加载换脸模型
            swapper_path = self._ensure_swapper_model()
            self._swapper = get_insightface_model(
                str(swapper_path),
                providers=self.providers,
            )
            logger.info("换脸模型加载完成")

            self._initialized = True

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise RuntimeError(f"换脸引擎初始化失败: {e}")

    def detect_faces(self, img: np.ndarray) -> List[Any]:
        """检测图像中所有人脸

        Args:
            img: BGR 图像 (OpenCV 格式)

        Returns:
            人脸对象列表，每个包含 bbox/landmarks/embedding/age/gender
        """
        self.initialize()

        if img is None:
            return []

        # 确保 RGB 格式 (insightface 使用 RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self._face_analysis.get(img_rgb)
        logger.debug(f"检测到 {len(faces)} 张人脸")
        return faces

    def swap(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_face_idx: int = 0,
        target_face_idx: int = 0,
        enhance: bool = False,
        blend: bool = True,
    ) -> np.ndarray:
        """执行换脸

        Args:
            source_img: 源图像 (BGR) — 提供人脸特征
            target_img: 目标图像 (BGR) — 被换脸的对象
            source_face_idx: 源图中第几张人脸 (默认第一张)
            target_face_idx: 目标图中第几张人脸 (默认第一张)
            enhance: 是否进行 GFPGAN 增强
            blend: 是否使用无缝融合

        Returns:
            换脸后的 BGR 图像
        """
        self.initialize()

        # 检测人脸
        source_faces = self.detect_faces(source_img)
        target_faces = self.detect_faces(target_img)

        if len(source_faces) == 0:
            raise ValueError("源图像未检测到人脸")
        if len(target_faces) == 0:
            raise ValueError("目标图像未检测到人脸")

        if source_face_idx >= len(source_faces):
            raise ValueError(
                f"源图第 {source_face_idx} 张人脸不存在 (共 {len(source_faces)} 张)"
            )
        if target_face_idx >= len(target_faces):
            raise ValueError(
                f"目标图第 {target_face_idx} 张人脸不存在 (共 {len(target_faces)} 张)"
            )

        source_face = source_faces[source_face_idx]
        target_face = target_faces[target_face_idx]

        logger.info(
            f"执行换脸: 源={source_face_idx} "
            f"(性别={'男' if source_face.sex == 1 else '女'}, "
            f"年龄≈{source_face.age}) "
            f"→ 目标={target_face_idx} "
            f"(性别={'男' if target_face.sex == 1 else '女'}, "
            f"年龄≈{target_face.age})"
        )

        # 执行换脸
        result = self._swapper.get(target_img, target_face, source_face)

        # 可选: 无缝融合 (减少边界伪影)
        if blend:
            result = self._blend_face(result, target_img, target_face)

        # 可选: GFPGAN 增强
        if enhance:
            result = self.enhance(result, target_faces=[target_face])

        return result

    def swap_all_faces(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        enhance: bool = False,
        blend: bool = True,
    ) -> np.ndarray:
        """用源图中所有人脸替换目标图中对应数量的第一张人脸

        如果目标图有多张人脸，依次用源图的人脸替换；
        如果源图人脸少于目标图，剩余目标人脸保持不变
        """
        self.initialize()

        source_faces = self.detect_faces(source_img)
        target_faces = self.detect_faces(target_img)

        if len(source_faces) == 0:
            raise ValueError("源图像未检测到人脸")
        if len(target_faces) == 0:
            raise ValueError("目标图像未检测到人脸")

        result = target_img.copy()
        num_swaps = min(len(source_faces), len(target_faces))

        for i in range(num_swaps):
            logger.info(f"换脸: 源#{i} → 目标#{i}")
            result = self._swapper.get(result, target_faces[i], source_faces[i])
            if blend:
                result = self._blend_face(result, target_img, target_faces[i])

        if enhance:
            result = self.enhance(result, target_faces=target_faces)

        return result

    def _blend_face(
        self,
        swapped_img: np.ndarray,
        original_img: np.ndarray,
        face: Any,
    ) -> np.ndarray:
        """用无缝融合改善人像边界"""
        try:
            bbox = face.bbox.astype(np.int32).flatten()
            x1, y1, x2, y2 = bbox[:4]

            # 稍微扩大融合区域
            h, w = original_img.shape[:2]
            margin = int(max(x2 - x1, y2 - y1) * 0.1)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            # ROI
            roi_src = swapped_img[y1:y2, x1:x2]
            roi_dst = original_img[y1:y2, x1:x2]

            # 创建遮罩
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            # 在遮罩中绘制面部区域
            landmarks = getattr(face, 'landmarks_2d', None)
            if landmarks is not None:
                landmarks = landmarks.astype(np.int32)
            if landmarks is not None and len(landmarks) >= 5:
                # 用关键点创建凸包
                hull = cv2.convexHull(landmarks)
                cv2.fillConvexPoly(mask, hull, 255)
            else:
                # 用 bbox 中心椭圆
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                cv2.ellipse(mask, (cx - x1, cy - y1), axes, 0, 0, 360, 255, -1)

            # 高斯模糊遮罩以实现柔和过渡
            mask = cv2.GaussianBlur(mask, (21, 21), 11)

            # 加权混合
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            blended_roi = (
                roi_src.astype(np.float32) * mask_3ch
                + roi_dst.astype(np.float32) * (1.0 - mask_3ch)
            ).astype(np.uint8)

            result = swapped_img.copy()
            result[y1:y2, x1:x2] = blended_roi
            return result

        except Exception as e:
            logger.warning(f"混合失败，返回原始换脸结果: {e}")
            return swapped_img

    def enhance(
        self,
        img: np.ndarray,
        target_faces: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """使用 GFPGAN 增强人脸质量"""
        if self._enhancer is None:
            self._enhancer = FaceEnhancer()
        return self._enhancer.enhance(img, faces=target_faces)

    def get_face_info(self, img: np.ndarray) -> List[dict]:
        """获取图像中的所有人脸信息 (用于 UI 展示)"""
        faces = self.detect_faces(img)
        info_list = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(np.int32).flatten()
            info_list.append(
                {
                    "idx": i,
                    "bbox": bbox[:4].tolist(),
                    "age": face.age,
                    "gender": "男" if face.sex == 1 else "女",
                    "confidence": float(face.det_score),
                }
            )
        return info_list


# 单例模式 — 全局共享引擎实例
_global_engine: Optional[FaceSwapper] = None


def get_engine(
    det_threshold: float = 0.5,
    use_gpu: bool = True,
    **kwargs,
) -> FaceSwapper:
    """获取全局共享的换脸引擎实例 (单例)"""
    global _global_engine
    if _global_engine is None:
        _global_engine = FaceSwapper(
            det_threshold=det_threshold,
            use_gpu=use_gpu,
            **kwargs,
        )
    return _global_engine
