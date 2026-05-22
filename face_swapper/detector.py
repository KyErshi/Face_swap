"""
人脸检测封装
提供多种检测器后端: insightface (默认) / OpenCV Haar / OpenCV DNN
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """人脸检测器 (抽象多后端)

    提供统一接口，支持:
    - insightface (高精度，有 landmarks)
    - OpenCV DNN (轻量)
    - OpenCV Haar Cascade (CPU 极速)
    """

    def __init__(self, backend: str = "insightface"):
        self.backend = backend
        self._detector: Optional[BaseDetector] = None

        if backend == "insightface":
            self._detector = InsightFaceDetector()
        elif backend == "opencv_dnn":
            self._detector = OpenCVDNNDetector()
        elif backend == "haar":
            self._detector = HaarDetector()
        else:
            raise ValueError(f"未知检测器后端: {backend}")

        logger.info(f"人脸检测器初始化: {backend}")

    def detect(self, img: np.ndarray) -> List[FaceBox]:
        """检测图像中的人脸

        Args:
            img: BGR 图像

        Returns:
            FaceBox 列表 (每个包含 bbox, confidence, 可选的 landmarks)
        """
        if img is None or img.size == 0:
            return []
        return self._detector.detect(img)


class FaceBox:
    """人脸检测框"""

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        landmarks: Optional[List[Tuple[int, int]]] = None,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.landmarks = landmarks or []

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def __repr__(self) -> str:
        return (
            f"FaceBox({self.x1},{self.y1},{self.x2},{self.y2}, "
            f"conf={self.confidence:.3f})"
        )


class BaseDetector(ABC):
    """检测器抽象基类"""

    @abstractmethod
    def detect(self, img: np.ndarray) -> List[FaceBox]:
        ...


class InsightFaceDetector(BaseDetector):
    """insightface 检测器 (需已加载模型)"""

    def __init__(self):
        # 由引擎的 FaceAnalysis 负责，此处仅包装
        self._engine_ref = None

    def detect(self, img: np.ndarray) -> List[FaceBox]:
        # 实际由 FaceSwapper 的 FaceAnalysis 完成
        # 此方法预留为独立调用接口
        logger.warning("InsightFaceDetector 应通过 FaceSwapper.detect_faces() 使用")
        return []


class OpenCVDNNDetector(BaseDetector):
    """OpenCV DNN 人脸检测器

    使用 OpenCV 预训练的 Caffe 模型 (SSD).
    模型文件会自动从 OpenCV 包加载.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._net = None
        self._load_model()

    def _load_model(self):
        """加载 DNN 模型"""
        try:
            # OpenCV 4.5.1+ 内置了人脸检测模型
            # 使用 OpenCV 的 FaceDetectorYN
            self._net = cv2.FaceDetectorYN.create(
                model=None,  # 使用默认模型
                config=None,
                input_size=(320, 320),
                score_threshold=self.confidence_threshold,
            )
            logger.info("OpenCV DNN 人脸检测器加载成功")
        except Exception as e:
            logger.warning(f"OpenCV DNN 加载失败, 回退到 Haar: {e}")
            self._net = None

    def detect(self, img: np.ndarray) -> List[FaceBox]:
        if self._net is None:
            return []

        h, w = img.shape[:2]
        self._net.setInputSize((w, h))
        _, faces = self._net.detect(img)

        results = []
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2, conf = face[:4].astype(np.int32), face[4]
                if conf > self.confidence_threshold:
                    results.append(FaceBox(x1, y1, x2, y2, float(conf)))
        return results


class HaarDetector(BaseDetector):
    """Haar Cascade CPU 快速检测器"""

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path:
            self._cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # 使用 OpenCV 内置的 frontalface 模型
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        if self._cascade.empty():
            logger.error("Haar Cascade 加载失败")
        else:
            logger.info("Haar Cascade 人脸检测器加载成功")

    def detect(
        self,
        img: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[FaceBox]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        results = []
        for x, y, w, h in faces:
            results.append(FaceBox(x, y, x + w, y + h, confidence=1.0))
        return results
