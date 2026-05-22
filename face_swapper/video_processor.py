"""
视频换脸处理器
支持: 逐帧换脸、音频保留、进度回调、批量处理
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Callable, Union

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

        try:
            # 视频写入器
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                temp_video,
                fourcc,
                actual_fps,
                (width, height),
            )

            if not writer.isOpened():
                # 回退到 avi
                temp_video = temp_video.replace(".mp4", ".avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(
                    temp_video,
                    fourcc,
                    actual_fps,
                    (width, height),
                )

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

            # 合并音频
            if keep_audio:
                output_path = self._merge_audio(video_path, temp_video, output_path)
            else:
                # 直接移动
                self._ensure_output_dir(output_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_video, output_path)

            return output_path

        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            raise
        finally:
            # 清理临时文件
            cap.release()
            if os.path.exists(temp_video) and temp_video != output_path:
                try:
                    os.remove(temp_video)
                except OSError:
                    pass

    def _merge_audio(
        self,
        input_video: str,
        temp_video: str,
        output_video: str,
    ) -> str:
        """使用 ffmpeg 合并视频和音频"""
        self._ensure_output_dir(output_video)

        # 先尝试用 ffmpeg-python
        try:
            import ffmpeg

            input_video_obj = ffmpeg.input(temp_video)
            input_audio_obj = ffmpeg.input(input_video)

            # 音频流从原视频，视频流从换脸结果
            stream = ffmpeg.output(
                input_video_obj["v"],
                input_audio_obj["a"],
                output_video,
                vcodec="libx264",
                acodec="aac",
                preset="fast",
                **{"b:v": "8M"},
                loglevel="quiet",
            )
            stream.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)

            logger.info(f"音频合并完成: {output_video}")
            return output_video

        except Exception as e:
            logger.warning(f"ffmpeg-python 合并失败: {e}，尝试直接调用 ffmpeg...")

        # 回退: 直接调用 ffmpeg 命令行
        try:
            import subprocess

            cmd = [
                "ffmpeg",
                "-i", temp_video,
                "-i", input_video,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-y",
                output_video,
            ]
            subprocess.run(cmd, capture_output=True, timeout=300)
            logger.info(f"音频合并完成: {output_video}")
            return output_video

        except Exception as e:
            logger.warning(f"ffmpeg 不可用, 返回无音频视频: {e}")
            # 无法合并音频，返回无声视频
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
