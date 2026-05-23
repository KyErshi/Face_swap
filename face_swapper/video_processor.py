"""
视频换脸处理器
支持: 逐帧换脸、音频保留、进度回调、批量处理
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Callable, Union, Any

import cv2
import numpy as np
from tqdm import tqdm

from .engine import FaceSwapper

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频换脸处理器

    流程:
    1. 打开视频 -> 获取 fps/总帧数/尺寸
    2. 读取源图人脸
    3. 逐帧: 读取 -> 换脸 -> 写入临时视频
    4. 合并音轨 -> 输出最终视频
    """

    def __init__(
        self,
        swapper: FaceSwapper,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        temp_dir: Optional[str] = None,
    ):
        self.swapper = swapper
        self.progress_callback = progress_callback
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def process_video(
        self,
        source_img: np.ndarray,
        video_path: str,
        output_path: str,
        source_face_idx: int = 0,
        target_face_idx: int = 0,
        enhance: bool = False,
        enhance_source: bool = True,
        blend: bool = True,
        color_match: bool = True,
        skin_texture: bool = False,
        temporal_smooth: bool = True,
        max_frames: int = 0,  # 0 = 全部帧
        start_frame: int = 0,
        end_frame: int = 0,  # 0 = 到结束
        keep_audio: bool = True,
        target_fps: Optional[float] = None,
    ) -> str:
        """执行视频换脸

        Args:
            source_img: 源人脸图像 (BGR)
            video_path: 输入视频路径
            output_path: 输出视频路径
            source_face_idx: 源图中第几张人脸
            target_face_idx: 目标视频中第几张人脸 (仅用于检测顺序)
            enhance: 是否 GFPGAN 增强每帧
            blend: 是否无缝融合
            max_frames: 最大处理帧数 (0=全部)
            start_frame: 起始帧号
            end_frame: 结束帧号 (0=视频末尾)
            keep_audio: 是否保留源视频音频
            target_fps: 目标帧率 (None=保持原帧率)

        Returns:
            输出视频路径
        """
        # 检测源人脸 (一次性)
        source_faces = self.swapper.detect_faces(source_img)
        if len(source_faces) == 0:
            raise ValueError("源图像未检测到人脸")
        if source_face_idx >= len(source_faces):
            raise ValueError(
                f"源图第 {source_face_idx} 张人脸不存在 (共 {len(source_faces)} 张)"
            )
        source_face = source_faces[source_face_idx]

        # 保存原始 embedding (用于后续融合)
        original_src_embedding = getattr(source_face, 'normed_embedding', None)

        # 源人脸预增强: GFPGAN → 眼部多尺度细节增强
        if enhance_source:
            try:
                from .enhancer import FaceEnhancer
                from .utils import crop_face
                enhancer = FaceEnhancer()

                # --- 1. 先 GFPGAN 增强全脸 ---
                bbox_arr = source_face.bbox.astype(np.int32).flatten()
                src_face_crop = crop_face(source_img, bbox_arr)
                self._src_crop_video = src_face_crop
                # 保存 crop 原点偏移 (用于后续精确对齐)
                self._crop_ox = max(0, bbox_arr[0] - int((bbox_arr[2]-bbox_arr[0])*0.3))
                self._crop_oy = max(0, bbox_arr[1] - int((bbox_arr[3]-bbox_arr[1])*0.3))
                enhanced_crop = src_face_crop.copy()
                if src_face_crop is not None and src_face_crop.size > 0:
                    gfpgan_result = enhancer.enhance(src_face_crop)
                    if gfpgan_result is not None:
                        enhanced_crop = gfpgan_result
                        self._crop_enhanced_video = enhanced_crop

                # --- 2. 在 GFPGAN 结果上做眼部多尺度细节增强 ---
                src_lmk = self.swapper._get_landmarks(source_face)
                if src_lmk is not None and len(src_lmk) >= 2:
                    src_eyes = src_lmk[:2].astype(np.int32)
                    eye_dist = float(np.linalg.norm(
                        src_eyes[0].astype(float) - src_eyes[1].astype(float)
                    ))
                    eye_r = max(int(eye_dist * 0.20), 10)
                    for ex, ey in src_eyes:
                        x1 = max(0, ex - eye_r)
                        y1 = max(0, ey - eye_r)
                        x2 = min(enhanced_crop.shape[1], ex + eye_r)
                        y2 = min(enhanced_crop.shape[0], ey + eye_r)
                        roi = enhanced_crop[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        # 2a. CLAHE
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(3, 3))
                        eg = clahe.apply(gray)
                        # 2b. USM 锐化
                        blur = cv2.GaussianBlur(eg, (0, 0), 1.5)
                        usm = cv2.addWeighted(eg, 1.8, blur, -0.8, 0)
                        # 2c. DoG 细节提取
                        fine = cv2.GaussianBlur(usm, (0, 0), 0.5)
                        coarse = cv2.GaussianBlur(usm, (0, 0), 3.0)
                        detail = cv2.subtract(fine, coarse)
                        detail = cv2.addWeighted(detail, 0, detail, 2.5, 0)
                        # 2d. 合成
                        base = usm.astype(float)
                        final_gray = np.clip(base + detail.astype(float) * 0.4, 0, 255).astype(np.uint8)
                        # 2e. 写回 V 通道
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(float)
                        hsv[..., 2] = hsv[..., 2] * 0.3 + final_gray.astype(float) * 0.7
                        enhanced_crop[y1:y2, x1:x2] = cv2.cvtColor(
                            hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
                        )
                    self._crop_enhanced_video = enhanced_crop  # 保存储最终增强 crop
                    logger.info("视频眼部多尺度细节增强完成 (CLAHE+USM+DoG)")

                    # --- 3. 贴回原图再检测 (保留上下文) ---
                    face_w = bbox_arr[2] - bbox_arr[0]
                    face_h = bbox_arr[3] - bbox_arr[1]
                    margin_x = int(face_w * 0.3)
                    margin_y = int(face_h * 0.3)
                    px1 = max(0, bbox_arr[0] - margin_x)
                    py1 = max(0, bbox_arr[1] - margin_y)
                    px2 = min(source_img.shape[1], bbox_arr[2] + margin_x)
                    py2 = min(source_img.shape[0], bbox_arr[3] + margin_y)
                    paste_h, paste_w = py2 - py1, px2 - px1

                    enhanced_full = source_img.copy().astype(np.float32)
                    paste = cv2.resize(enhanced_crop, (paste_w, paste_h))
                    mask_paste = np.zeros((paste_h, paste_w), dtype=np.float32)
                    cv2.ellipse(mask_paste, (paste_w//2, paste_h//2),
                                (paste_w//2-4, paste_h//2-4), 0, 0, 360, 1.0, -1)
                    mask_paste = cv2.GaussianBlur(mask_paste, (31, 31), 15)
                    mask_paste = np.clip(mask_paste, 0, 1)
                    mask_3ch = np.stack([mask_paste] * 3, axis=-1)
                    roi = enhanced_full[py1:py2, px1:px2].copy()
                    blended_roi = paste.astype(np.float32) * mask_3ch + roi * (1.0 - mask_3ch)
                    enhanced_full[py1:py2, px1:px2] = blended_roi
                    enhanced_full = np.clip(enhanced_full, 0, 255).astype(np.uint8)

                    # 在完整图上检测 + embedding 融合
                    enhanced_faces = self.swapper.detect_faces(enhanced_full, retry_lower_threshold=False)
                    if len(enhanced_faces) > 0:
                        best = max(enhanced_faces, key=lambda f: f.det_score if hasattr(f, 'det_score') else 1.0)
                        enhanced_emb = getattr(best, 'normed_embedding', None)
                        if original_src_embedding is not None and enhanced_emb is not None:
                            fused = (original_src_embedding + enhanced_emb).astype(np.float32)
                            fused /= np.linalg.norm(fused)
                            best.normed_embedding = fused
                            logger.info("视频源人脸预增强完成，融合双 embedding")
                        else:
                            logger.info("视频源人脸预增强完成 (无法融合)")
                        source_face = best
                    else:
                        # 回退: 裁剪图
                        enhanced_faces = self.swapper.detect_faces(enhanced_crop, retry_lower_threshold=False)
                        if len(enhanced_faces) > 0:
                            best = max(enhanced_faces, key=lambda f: f.det_score if hasattr(f, 'det_score') else 1.0)
                            enhanced_emb = getattr(best, 'normed_embedding', None)
                            if original_src_embedding is not None and enhanced_emb is not None:
                                fused = (original_src_embedding + enhanced_emb).astype(np.float32)
                                fused /= np.linalg.norm(fused)
                                best.normed_embedding = fused
                            source_face = best
                            logger.info("视频源人脸预增强完成 (裁剪回退)")
            except Exception as e:
                logger.warning(f"视频源人脸预增强失败，使用原始特征: {e}")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = target_fps or fps

        logger.info(
            f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧, "
            f"处理范围: {start_frame}-{end_frame or total_frames}"
        )

        # 修正结束帧
        if end_frame <= 0 or end_frame > total_frames:
            end_frame = total_frames

        # 临时无音频视频
        temp_video = os.path.join(
            self.temp_dir, f"temp_swapped_{os.getpid()}.mp4"
        )
        temp_video_avi = temp_video.replace(".mp4", ".avi")

        try:
            # 写入 AVI 格式 (OpenCV 对 AVI 支持最稳定)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(
                temp_video_avi,
                fourcc,
                actual_fps,
                (width, height),
            )

            if not writer.isOpened():
                # 回退: 直接写入 mp4
                writer = cv2.VideoWriter(
                    temp_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    actual_fps,
                    (width, height),
                )
                temp_video_avi = temp_video  # 不使用 avi 路径

            # 跳到起始帧
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 逐帧处理
            frame_idx = start_frame
            processed = 0
            max_proc = max_frames if max_frames > 0 else (end_frame - start_frame)

            # 进度条
            pbar = tqdm(
                total=min(max_proc, end_frame - start_frame),
                desc="视频换脸",
                unit="帧",
            )

            # 时域平滑状态
            prev_face_bbox: Optional[np.ndarray] = None  # (cx, cy, w, h)
            # 检测间隔优化: 每 N 帧检测一次，中间帧复用上次的人脸框
            DETECT_INTERVAL = 15  # 每 15 帧检测一次人脸 (0.5s @ 30fps)
            cached_target_face = None
            last_detect_frame = -999

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # 对帧执行换脸
                try:
                    detect_ran = False
                    # 检测当前帧中的人脸 (间隔优化 + 运动回退)
                    should_detect = (cached_target_face is None
                                     or (frame_idx - last_detect_frame) >= DETECT_INTERVAL)
                    # 如果缓存的人脸框位置与上一帧差距过大，强制重检测
                    if not should_detect and cached_target_face is not None and prev_face_bbox is not None:
                        cx0, cy0 = prev_face_bbox[:2]
                        b = cached_target_face.bbox.astype(np.int32).flatten()
                        cx1 = (b[0] + b[2]) // 2
                        cy1 = (b[1] + b[3]) // 2
                        move = np.sqrt((cx1 - cx0)**2 + (cy1 - cy0)**2)
                        if move > prev_face_bbox[2] * 0.3:  # 移动超过人脸宽30%
                            should_detect = True

                    if should_detect:
                        target_faces = self.swapper.detect_faces(frame)
                        last_detect_frame = frame_idx
                        detect_ran = True
                    else:
                        target_faces = [cached_target_face]
                        detect_ran = False

                    if len(target_faces) > 0:
                        # 时域追踪: 选择最接近上一帧的人脸
                        if detect_ran:
                            if temporal_smooth and prev_face_bbox is not None:
                                prev_cx, prev_cy = prev_face_bbox[:2]
                                best_idx = 0
                                best_dist = float("inf")
                                for i, f in enumerate(target_faces):
                                    b = f.bbox.astype(np.int32).flatten()
                                    cx = (b[0] + b[2]) // 2
                                    cy = (b[1] + b[3]) // 2
                                    dist = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = i
                                tgt_idx = best_idx
                            else:
                                tgt_idx = min(target_face_idx, len(target_faces) - 1)
                            # 缓存本次检测到的人脸
                            cached_target_face = target_faces[tgt_idx]
                        else:
                            tgt_idx = 0  # 只有缓存的单张人脸

                        # 更新追踪位置
                        if temporal_smooth and detect_ran:
                            b = cached_target_face.bbox.astype(np.int32).flatten()
                            prev_face_bbox = np.array([
                                (b[0] + b[2]) // 2, (b[1] + b[3]) // 2,
                                b[2] - b[0], b[3] - b[1],
                            ])

                        result = self.swapper._swapper.get(
                            frame, cached_target_face, source_face
                        )

                        if blend:
                            result = self.swapper._blend_face(
                                result, frame, target_faces[tgt_idx]
                            )

                        if skin_texture:
                            try:
                                from .utils import transfer_skin_texture
                                bbox = target_faces[tgt_idx].bbox.astype(np.int32).flatten()
                                src_lmk = self.swapper._get_landmarks(source_face)
                                result = transfer_skin_texture(
                                    result, source_img,
                                    source_face_roi=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                    strength=0.3, sigma=2.0,
                                    landmarks=src_lmk,
                                )
                            except Exception as e:
                                logger.warning(f"帧 {frame_idx} 皮肤纹理失败: {e}")

                        if color_match:
                            try:
                                from .utils import color_transfer
                                face = target_faces[tgt_idx]
                                lmk = self.swapper._get_landmarks(face)
                                if lmk is not None:
                                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                                    hull = cv2.convexHull(lmk.astype(np.int32))
                                    cv2.fillConvexPoly(mask, hull, 255)
                                    mask = cv2.GaussianBlur(mask, (31, 31), 15)
                                    result = color_transfer(result, frame, mask=mask)
                            except Exception as e:
                                logger.warning(f"帧 {frame_idx} 颜色迁移失败: {e}")

                        # 双眼部+全脸细节增强 (CLAHE/USM/DoG，整图无色块)
                        try:
                            tgt_face = target_faces[tgt_idx]
                            tgt_lmk = self.swapper._get_landmarks(tgt_face)
                            if tgt_lmk is not None and len(tgt_lmk) >= 2:
                                # 面部轮廓遮罩
                                face_mask = np.zeros(result.shape[:2], dtype=np.uint8)
                                if len(tgt_lmk) >= 33:
                                    cv2.fillConvexPoly(face_mask, tgt_lmk[:33].astype(np.int32), 255)
                                elif len(tgt_lmk) >= 5:
                                    cv2.fillConvexPoly(face_mask, cv2.convexHull(tgt_lmk.astype(np.int32)), 255)
                                else:
                                    b = tgt_face.bbox.astype(np.int32).flatten()
                                    cv2.ellipse(face_mask, ((b[0]+b[2])//2,(b[1]+b[3])//2),
                                                ((b[2]-b[0])//2,(b[3]-b[1])//2), 0, 0, 360, 255, -1)
                                face_mask_f = cv2.GaussianBlur(face_mask, (41, 41), 20).astype(float) / 255.0
                                # 眼部遮罩
                                eyes = tgt_lmk[:2].astype(np.int32)
                                eye_dist = float(np.linalg.norm(eyes[0].astype(float) - eyes[1].astype(float)))
                                eye_r = max(int(eye_dist * 0.28), 14)
                                eye_mask = np.zeros(result.shape[:2], dtype=np.float32)
                                for ex, ey in eyes:
                                    yy, xx = np.ogrid[:result.shape[0], :result.shape[1]]
                                    dist = np.sqrt((xx - ex)**2 + (yy - ey)**2)
                                    m = np.clip(1.0 - dist / eye_r, 0, 1)
                                    m = cv2.GaussianBlur(m, (0, 0), eye_r * 0.3)
                                    eye_mask = np.maximum(eye_mask, m)
                                eye_mask = np.clip(eye_mask, 0, 1)
                                # 图像处理
                                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                                # 全脸轻度
                                cfa = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                                eg_f = cfa.apply(gray)
                                usm_f = cv2.addWeighted(eg_f, 1.5, cv2.GaussianBlur(eg_f, (0, 0), 1.8), -0.5, 0)
                                # 眼部强力
                                ce = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(3, 3))
                                eg_e = ce.apply(gray)
                                usm_e = cv2.addWeighted(eg_e, 1.8, cv2.GaussianBlur(eg_e, (0, 0), 1.5), -0.8, 0)
                                fine = cv2.GaussianBlur(usm_e, (0, 0), 0.5)
                                coarse = cv2.GaussianBlur(usm_e, (0, 0), 3.0)
                                detail = cv2.subtract(fine, coarse)
                                eye_final = np.clip(usm_e.astype(float) + detail.astype(float) * 0.5, 0, 255).astype(np.uint8)
                                # 混合
                                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(float)
                                ov = hsv[..., 2].copy()
                                hsv[..., 2] = ov * (1 - face_mask_f * 0.30) + usm_f.astype(float) * face_mask_f * 0.30
                                blend_e = ov * (1 - eye_mask * 0.55) + eye_final.astype(float) * eye_mask * 0.55
                                hsv[..., 2] = hsv[..., 2] * (1 - eye_mask) + blend_e * eye_mask
                                result = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
                        except Exception:
                            pass

                        if enhance:
                            result = self.swapper.enhance(result)
                    else:
                        result = frame  # 无人脸，跳过

                except Exception as e:
                    logger.warning(f"帧 {frame_idx} 处理失败: {e}")
                    result = frame

                writer.write(result)

                frame_idx += 1
                processed += 1
                pbar.update(1)

                # 回调
                if self.progress_callback:
                    self.progress_callback(processed, max_proc)

                # 达到最大帧数
                if max_frames > 0 and processed >= max_frames:
                    break

            pbar.close()
            writer.release()
            cap.release()
            logger.info(f"视频换脸完成: 处理 {processed} 帧")

            # 始终通过 ffmpeg 转为浏览器兼容的 MP4
            self._ensure_output_dir(output_path)
            source_for_ffmpeg = temp_video_avi if os.path.exists(temp_video_avi) else temp_video

            if keep_audio:
                success = self._reencode_video(source_for_ffmpeg, output_path, audio_video=video_path)
            else:
                success = self._reencode_video(source_for_ffmpeg, output_path, audio_video=None)

            if not success:
                logger.error("ffmpeg 全部失败，尝试返回原始文件")
                if os.path.exists(output_path):
                    os.remove(output_path)
                if os.path.exists(source_for_ffmpeg):
                    os.rename(source_for_ffmpeg, output_path)

            return output_path

        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            raise
        finally:
            cap.release()
            # 清理临时文件
            for tmp in [temp_video, temp_video_avi]:
                if os.path.exists(tmp) and tmp != output_path:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def _reencode_video(
        self,
        input_video: str,
        output_video: str,
        audio_video: Optional[str] = None,
    ) -> bool:
        """用 ffmpeg 将视频重新编码为浏览器兼容的 H.264 MP4

        Args:
            input_video: 源视频路径 (OpenCV 生成)
            output_video: 输出路径
            audio_video: 可选 — 从此文件中提取音频流

        Returns:
            是否成功
        """
        import subprocess

        # 所有输入放前面，输出选项放后面
        # 注意: -c:v 等编码选项必须在所有 -i 之后，否则 ffmpeg 会误用为解码器
        cmd = ["ffmpeg", "-i", input_video]

        if audio_video and os.path.exists(audio_video):
            cmd.extend(["-i", audio_video])

        # 输出选项
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-crf", "18",
        ])

        if audio_video and os.path.exists(audio_video):
            cmd.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        else:
            cmd.extend(["-an"])

        cmd.extend(["-y", output_video])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info(f"ffmpeg 重新编码完成: {output_video}")
                return True
            else:
                logger.error(f"ffmpeg 失败:\n{result.stderr}")
                return False
        except Exception as e:
            logger.error(f"ffmpeg 异常: {e}")
            return False

    def _merge_audio(
        self,
        input_video: str,
        temp_video: str,
        output_video: str,
    ) -> str:
        """重新编码并合并音频为浏览器可播放的 MP4"""
        self._ensure_output_dir(output_video)

        # 尝试带音频的重新编码
        if self._reencode_video(temp_video, output_video, audio_video=input_video):
            return output_video

        # 回退 1: 无声版本
        no_audio_out = output_video.replace(".mp4", "_no_audio.mp4")
        if self._reencode_video(temp_video, no_audio_out, audio_video=None):
            # 复制回最终路径
            if os.path.exists(output_video):
                os.remove(output_video)
            os.rename(no_audio_out, output_video)
            return output_video

        # 回退 2: 直接使用 OpenCV 原片
        logger.warning("ffmpeg 全部失败，返回 OpenCV 原始视频 (部分浏览器可能无法播放)")
        if os.path.exists(output_video):
            os.remove(output_video)
        os.rename(temp_video, output_video)
        return output_video

    @staticmethod
    def _ensure_output_dir(path: str):
        """确保输出目录存在"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: str) -> dict:
        """获取视频元信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            if cap.get(cv2.CAP_PROP_FPS) > 0
            else 0,
        }
        cap.release()
        return info
