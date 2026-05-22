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
        blend: bool = True,
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

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # 对帧执行换脸
                try:
                    # 检测当前帧中的人脸
                    target_faces = self.swapper.detect_faces(frame)

                    if len(target_faces) > 0:
                        # 处理所有检测到的人脸或指定的人脸
                        tgt_idx = min(target_face_idx, len(target_faces) - 1)
                        result = self.swapper._swapper.get(
                            frame, target_faces[tgt_idx], source_face
                        )

                        if blend:
                            result = self.swapper._blend_face(
                                result, frame, target_faces[tgt_idx]
                            )

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

        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",        # 浏览器兼容的色彩格式
            "-movflags", "+faststart",     # moov 放前面，支持流式播放
            "-crf", "18",                  # 高质量 (0-51, 越小越好)
        ]

        if audio_video and os.path.exists(audio_video):
            cmd.extend(["-i", audio_video])
            cmd.extend(["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
        else:
            cmd.extend(["-an"])  # 无音频

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
