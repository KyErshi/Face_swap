"""
人脸质量增强器
使用 GFPGAN 进行面部修复，提升换脸后的图像质量
"""

import logging
from typing import Optional, List, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceEnhancer:
    """人脸增强器

    使用 GFPGAN 模型进行面部超分辨率/修复。
    可选 Real-ESRGAN 作为通用图像增强。

    延迟加载: 只在第一次调用 enhance() 时下载/加载模型
    """

    def __init__(
        self,
        version: str = "1.4",
        upscale: float = 1.0,  # 1.0 = 不放大, 仅修复
        use_gpu: bool = True,
    ):
        self.version = version
        self.upscale = upscale
        self.use_gpu = use_gpu
        self._gfpgan = None
        self._initialized = False
        logger.info(f"FaceEnhancer 初始化 (v{version}, upscale={upscale})")

    def _initialize(self):
        """延迟加载 GFPGAN 模型"""
        if self._initialized:
            return

        try:
            from gfpgan import GFPGANer

            # GFPGAN 会从 HuggingFace 自动下载模型
            # 模型: GFPGANv1.4.pth
            # 背景模型: RestoreFormer.pth 或直接使用 GFPGAN
            self._gfpgan = GFPGANer(
                model_path=None,  # 使用默认路径
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,  # 不增强背景
            )
            self._initialized = True
            logger.info("GFPGAN 模型加载成功")

        except ImportError:
            logger.warning("GFPGAN 未安装，增强功能不可用")
            self._initialized = False
        except Exception as e:
            logger.warning(f"GFPGAN 加载失败: {e}")
            self._initialized = False

    def enhance(
        self,
        img: np.ndarray,
        faces: Optional[List[Any]] = None,
        weight: float = 0.8,
    ) -> np.ndarray:
        """对图像进行面部增强

        Args:
            img: BGR 输入图像
            faces: 可选 — 已知的人脸对象列表，用于加速检测
            weight: GFPGAN 融合权重 (0-1), 越高增强效果越强

        Returns:
            增强后的 BGR 图像
        """
        self._initialize()

        if not self._initialized or self._gfpgan is None:
            logger.warning("GFPGAN 不可用，返回原图")
            return img

        try:
            # GFPGAN 处理
            _, _, enhanced = self._gfpgan.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight,
            )

            if enhanced is not None:
                logger.info("面部增强完成")
                return enhanced
            else:
                return img

        except Exception as e:
            logger.error(f"增强处理失败: {e}")
            return img

    def enhance_face_region(
        self,
        img: np.ndarray,
        bbox: tuple,
    ) -> np.ndarray:
        """仅增强人脸区域 (局部增强)

        Args:
            img: BGR 图像
            bbox: (x1, y1, x2, y2) 人脸框

        Returns:
            增强后完整图像 (仅人脸区域被修改)
        """
        self._initialize()

        if not self._initialized or self._gfpgan is None:
            return img

        x1, y1, x2, y2 = map(int, bbox)

        # 扩大区域
        h, w = img.shape[:2]
        margin = int(max(x2 - x1, y2 - y1) * 0.2)
        x1_e = max(0, x1 - margin)
        y1_e = max(0, y1 - margin)
        x2_e = min(w, x2 + margin)
        y2_e = min(h, y2 + margin)

        # 裁剪面部区域
        face_roi = img[y1_e:y2_e, x1_e:x2_e]

        # 增强
        enhanced_roi = self.enhance(face_roi)

        # 拼回原图
        result = img.copy()
        result[y1_e:y2_e, x1_e:x2_e] = cv2.resize(
            enhanced_roi, (x2_e - x1_e, y2_e - y1_e)
        )

        return result
