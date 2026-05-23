"""
AI 换脸 Web UI — Gradio 应用

功能标签页:
1. 图片换脸 — 上传源图+目标图，选择人脸，换脸
2. 视频换脸 — 上传源图+视频，逐帧换脸
3. 批量处理 — 目录批量图片换脸
4. 人脸检测 — 可视化检测结果
5. 设置 — 模型参数调整
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np
import gradio as gr

from face_swapper import FaceSwapper, get_engine
from face_swapper.utils import (
    load_image,
    draw_face_boxes,
    resize_to_limit,
    is_image_file,
    is_video_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 全局引擎
_engine: Optional[FaceSwapper] = None

# 临时文件目录
TEMP_DIR = tempfile.mkdtemp(prefix="face_swap_")


def get_engine_instance() -> FaceSwapper:
    """获取/初始化全局引擎"""
    global _engine
    if _engine is None:
        _engine = get_engine()
        try:
            _engine.initialize()
        except Exception as e:
            logger.error(f"引擎初始化失败: {e}")
            raise
    return _engine


def update_det_threshold(threshold: float):
    """更新全局引擎的人脸检测阈值"""
    engine = get_engine_instance()
    engine.set_det_threshold(threshold)
    return f"✅ 检测阈值已设为 {threshold:.2f}"


def parse_face_idx(selector_value: str | int) -> int:
    """从 Gradio 下拉菜单值解析人脸索引"""
    if isinstance(selector_value, int):
        return selector_value
    if not selector_value or "未" in selector_value or "错误" in selector_value:
        return 0
    try:
        return int(selector_value.split("#")[1].split(" ")[0])
    except (IndexError, ValueError):
        return 0


# ============================================================
#  图片换脸
# ============================================================

def image_swap(
    source_img: Optional[np.ndarray],
    target_img: Optional[np.ndarray],
    source_face_idx: str | int,
    target_face_idx: str | int,
    enable_enhance: bool,
    enable_enhance_source: bool,
    enable_blend: bool,
    enable_color_match: bool,
    enable_preserve_eyes: bool,
    enable_skin_texture: bool,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[np.ndarray, str, str]:
    """图片换脸主函数"""
    if source_img is None:
        raise gr.Error("请上传源图像（提供人脸特征）")
    if target_img is None:
        raise gr.Error("请上传目标图像（被换脸的对象）")

    # 从下拉菜单字符串中解析人脸序号
    src_idx = parse_face_idx(source_face_idx)
    tgt_idx = parse_face_idx(target_face_idx)

    progress(0, desc="初始化引擎...")
    try:
        engine = get_engine_instance()
    except Exception as e:
        raise gr.Error(f"引擎初始化失败: {e}")

    progress(0.3, desc="检测人脸...")
    try:
        result = engine.swap(
            source_img,
            target_img,
            source_face_idx=src_idx,
            target_face_idx=tgt_idx,
            enhance=enable_enhance,
            enhance_source=enable_enhance_source,
            blend=enable_blend,
            color_match=enable_color_match,
            preserve_eyes=enable_preserve_eyes,
            skin_texture=enable_skin_texture,
        )
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        logger.exception("换脸失败")
        raise gr.Error(f"换脸失败: {e}")

    progress(1.0, desc="完成!")
    return result, "✅ 换脸成功!", ""


def detect_faces_ui(
    img: Optional[np.ndarray],
    progress: gr.Progress = gr.Progress(),
) -> Tuple[np.ndarray, str]:
    """人脸检测 + 可视化"""
    if img is None:
        raise gr.Error("请先上传图像")

    progress(0, desc="初始化...")
    engine = get_engine_instance()

    progress(0.3, desc="检测人脸...")
    faces_info = engine.get_face_info(img)

    if not faces_info:
        return img, "⚠️ 未检测到人脸"

    # 绘制
    result = draw_face_boxes(img, faces_info)
    info_str = "\n".join(
        f"#{f['idx']}: {f['gender']}, {f['age']}岁, 置信度={f['confidence']:.2f}"
        for f in faces_info
    )

    progress(1.0, desc="完成")
    return result, f"✅ 检测到 {len(faces_info)} 张人脸:\n{info_str}"


def get_face_selector(img: Optional[np.ndarray]) -> gr.Dropdown:
    """获取人脸选择器选项"""
    if img is None:
        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")

    try:
        engine = get_engine_instance()
        faces = engine.get_face_info(img)
        if not faces:
            return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")

        choices = [
            f"#{f['idx']} ({f['gender']}, {f['age']}岁, {f['confidence']:.2f})"
            for f in faces
        ]
        return gr.Dropdown(choices=choices, value=choices[0])
    except Exception:
        return gr.Dropdown(choices=["错误"], value="错误")


# ============================================================
#  视频换脸
# ============================================================

def video_swap(
    source_img: Optional[np.ndarray],
    target_video: Optional[str],
    source_face_idx: str | int,
    enable_enhance: bool,
    enable_enhance_source: bool,
    enable_blend: bool,
    enable_color_match: bool,
    enable_preserve_eyes: bool,
    enable_skin_texture: bool,
    enable_temporal_smooth: bool,
    keep_audio: bool,
    process_seconds: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], str]:
    """视频换脸"""
    if source_img is None:
        raise gr.Error("请上传源图像")
    if target_video is None:
        raise gr.Error("请上传目标视频")

    # 从下拉菜单字符串中解析人脸序号
    src_idx = parse_face_idx(source_face_idx)

    progress(0, desc="初始化引擎...")
    try:
        engine = get_engine_instance()
    except Exception as e:
        raise gr.Error(f"引擎初始化失败: {e}")

    # 输出路径
    output_path = os.path.join(
        TEMP_DIR, f"output_{os.getpid()}.mp4"
    )

    from face_swapper.video_processor import VideoProcessor

    # 逐帧更新进度
    def on_progress(current, total):
        if total > 0:
            pct = 0.1 + 0.85 * (current / total)
            progress(pct, desc=f"处理帧 {current}/{total}...")

    processor = VideoProcessor(
        swapper=engine,
        temp_dir=TEMP_DIR,
        progress_callback=on_progress,
    )

    # 获取视频信息
    try:
        info = processor.get_video_info(target_video)
        total_frames = int(info["total_frames"])
        fps = info["fps"]

        # 计算帧范围
        end_frame = 0
        max_frames = 0
        if process_seconds > 0:
            max_frames = int(fps * process_seconds)
            max_frames = min(max_frames, total_frames)

        logger.info(
            f"视频: {total_frames}帧, {fps:.2f}fps, 处理{max_frames or total_frames}帧"
        )
    except Exception as e:
        logger.warning(f"读取视频信息失败: {e}")
        max_frames = 0
        end_frame = 0

    progress(0.1, desc="正在换脸 (这可能需要几分钟)...")

    try:
        result_path = processor.process_video(
            source_img=source_img,
            video_path=target_video,
            output_path=output_path,
            source_face_idx=src_idx,
            target_face_idx=0,
            enhance=enable_enhance,
            enhance_source=enable_enhance_source,
            blend=enable_blend,
            color_match=enable_color_match,
            preserve_eyes=enable_preserve_eyes,
            skin_texture=enable_skin_texture,
            temporal_smooth=enable_temporal_smooth,
            max_frames=max_frames,
            keep_audio=keep_audio,
        )

        progress(1.0, desc="完成!")
        return result_path, "✅ 视频换脸完成!"
    except Exception as e:
        logger.exception("视频换脸失败")
        raise gr.Error(f"视频换脸失败: {e}")


# ============================================================
#  构建 UI
# ============================================================

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; }
h1 { text-align: center; margin-bottom: 0.5em; }
.tab-nav { font-size: 1.1em; }
footer { display: none !important; }
/* 确保按钮不被视频播放器遮挡 */
#video-swap-btn { position: relative; z-index: 100; clear: both; }
video { max-width: 100%; }
"""


def build_app() -> gr.Blocks:
    """构建 Gradio 应用"""

    with gr.Blocks(
        css=CSS,
        title="AI 换脸",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🎭 AI 换脸
            *基于 insightface 的高质量换脸工具 - 上传一张人脸照片和一张目标图像/视频即可*
            """
        )

        with gr.Tabs():
            # ========== Tab 1: 图片换脸 ==========
            with gr.TabItem("📷 图片换脸"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ① 源图像 (提供人脸)")
                        src_img_input = gr.Image(
                            label="上传源人脸照片",
                            type="numpy",
                            height=350,
                        )
                        with gr.Row():
                            src_refresh_btn = gr.Button(
                                "🔄 检测人脸", size="sm"
                            )
                            src_face_selector = gr.Dropdown(
                                choices=["上传后检测"],
                                value="上传后检测",
                                label="选择源人脸",
                                interactive=True,
                            )

                    with gr.Column():
                        gr.Markdown("### ② 目标图像 (被换脸)")
                        tgt_img_input = gr.Image(
                            label="上传目标照片",
                            type="numpy",
                            height=350,
                        )
                        with gr.Row():
                            tgt_refresh_btn = gr.Button(
                                "🔄 检测人脸", size="sm"
                            )
                            tgt_face_selector = gr.Dropdown(
                                choices=["上传后检测"],
                                value="上传后检测",
                                label="选择目标人脸",
                                interactive=True,
                            )

                with gr.Accordion("⚙️ 高级设置", open=False):
                    with gr.Row():
                        enable_enhance = gr.Checkbox(
                            label="GFPGAN 增强 (提高清晰度)",
                            value=False,
                        )
                        enable_enhance_source = gr.Checkbox(
                            label="源人脸增强 (提高相似度)",
                            value=True,
                        )
                        enable_blend = gr.Checkbox(
                            label="无缝融合 (减少边界)",
                            value=True,
                        )
                        enable_color_match = gr.Checkbox(
                            label="颜色匹配 (消除肤色色差)",
                            value=True,
                        )
                        enable_preserve_eyes = gr.Checkbox(
                            label="保留眼形", value=True
                        )
                        enable_skin_texture = gr.Checkbox(
                            label="皮肤纹理", value=True
                        )
                    det_threshold_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="人脸检测灵敏度 (越低越灵敏)",
                    )
                    det_threshold_status = gr.Markdown("")
                    det_threshold_slider.change(
                        fn=update_det_threshold,
                        inputs=[det_threshold_slider],
                        outputs=[det_threshold_status],
                    )
                    det_threshold_slider.release(
                        fn=update_det_threshold,
                        inputs=[det_threshold_slider],
                        outputs=[det_threshold_status],
                    )

                swap_btn = gr.Button(
                    "✨ 开始换脸",
                    variant="primary",
                    size="lg",
                )

                with gr.Row():
                    output_img = gr.Image(
                        label="换脸结果",
                        height=400,
                    )
                    detect_img = gr.Image(
                        label="人脸检测",
                        height=400,
                        visible=False,
                    )

                status_msg = gr.Markdown("")

                # 事件绑定
                def on_src_upload(img):
                    if img is None:
                        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")
                    engine = get_engine_instance()
                    faces = engine.get_face_info(img)
                    if not faces:
                        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")
                    choices = [
                        f"#{f['idx']} ({f['gender']}, {f['age']}岁)"
                        for f in faces
                    ]
                    return gr.Dropdown(choices=choices, value=choices[0])

                def on_tgt_upload(img):
                    if img is None:
                        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")
                    engine = get_engine_instance()
                    faces = engine.get_face_info(img)
                    if not faces:
                        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")
                    choices = [
                        f"#{f['idx']} ({f['gender']}, {f['age']}岁)"
                        for f in faces
                    ]
                    return gr.Dropdown(choices=choices, value=choices[0])



                src_img_input.upload(
                    on_src_upload,
                    inputs=[src_img_input],
                    outputs=[src_face_selector],
                )
                tgt_img_input.upload(
                    on_tgt_upload,
                    inputs=[tgt_img_input],
                    outputs=[tgt_face_selector],
                )
                src_refresh_btn.click(
                    on_src_upload,
                    inputs=[src_img_input],
                    outputs=[src_face_selector],
                )
                tgt_refresh_btn.click(
                    on_tgt_upload,
                    inputs=[tgt_img_input],
                    outputs=[tgt_face_selector],
                )

                swap_btn.click(
                    fn=image_swap,
                    inputs=[
                        src_img_input,
                        tgt_img_input,
                        src_face_selector,
                        tgt_face_selector,
                        enable_enhance,
                        enable_enhance_source,
                        enable_blend,
                        enable_color_match,
                        enable_preserve_eyes,
                        enable_skin_texture,
                    ],
                    outputs=[output_img, status_msg],
                )

            # ========== Tab 2: 视频换脸 ==========
            with gr.TabItem("🎬 视频换脸"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### 源人脸照片")
                        v_src_img = gr.Image(
                            label="上传源人脸",
                            type="numpy",
                            height=250,
                        )
                        v_src_face_selector = gr.Dropdown(
                            choices=["上传后检测"],
                            value="上传后检测",
                            label="选择源人脸",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 目标视频")
                        v_tgt_video = gr.Video(
                            label="上传目标视频",
                            height=250,
                        )

                with gr.Accordion("⚙️ 视频设置", open=False):
                    with gr.Row():
                        v_keep_audio = gr.Checkbox(
                            label="保留原音频", value=True
                        )
                        v_enhance = gr.Checkbox(
                            label="GFPGAN 增强 (慢)", value=False
                        )
                        v_blend = gr.Checkbox(
                            label="无缝融合", value=True
                        )
                        v_color_match = gr.Checkbox(
                            label="颜色匹配", value=True
                        )
                        v_temporal_smooth = gr.Checkbox(
                            label="时域平滑 (减少闪烁)", value=True
                        )
                        v_enhance_source = gr.Checkbox(
                            label="源人脸增强", value=True
                        )
                        v_preserve_eyes = gr.Checkbox(
                            label="保留眼形", value=True
                        )
                        v_skin_texture = gr.Checkbox(
                            label="皮肤纹理", value=True
                        )
                    v_process_seconds = gr.Slider(
                        minimum=0,
                        maximum=120,
                        value=10,
                        step=1,
                        label="处理时长 (秒, 0=全部)",
                    )

                with gr.Group():
                    v_swap_btn = gr.Button(
                        "🎥 开始视频换脸",
                        variant="primary",
                        size="lg",
                        elem_id="video-swap-btn",
                    )
                    v_status = gr.Markdown("")

                gr.Markdown("---")
                with gr.Group():
                    gr.Markdown("### 换脸结果")
                    v_output = gr.Video(height=350)

                def on_v_src_upload(img):
                    if img is None:
                        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")
                    engine = get_engine_instance()
                    faces = engine.get_face_info(img)
                    if not faces:
                        return gr.Dropdown(choices=["未检测到人脸"], value="未检测到人脸")
                    choices = [
                        f"#{f['idx']} ({f['gender']}, {f['age']}岁)"
                        for f in faces
                    ]
                    return gr.Dropdown(choices=choices, value=choices[0])

                v_src_img.upload(
                    on_v_src_upload,
                    inputs=[v_src_img],
                    outputs=[v_src_face_selector],
                )

                v_swap_btn.click(
                    fn=video_swap,
                    inputs=[
                        v_src_img,
                        v_tgt_video,
                        v_src_face_selector,
                        v_enhance,
                        v_enhance_source,
                        v_blend,
                        v_color_match,
                        v_preserve_eyes,
                        v_skin_texture,
                        v_temporal_smooth,
                        v_keep_audio,
                        v_process_seconds,
                    ],
                    outputs=[v_output, v_status],
                )

            # ========== Tab 3: 人脸检测 ==========
            with gr.TabItem("🔍 人脸检测"):
                gr.Markdown("上传图像进行人脸检测和识别")
                with gr.Row():
                    with gr.Column():
                        d_img = gr.Image(
                            label="上传图像",
                            type="numpy",
                            height=400,
                        )
                        d_btn = gr.Button(
                            "🔍 检测人脸",
                            variant="primary",
                        )
                    with gr.Column():
                        d_output = gr.Image(
                            label="检测结果",
                            height=400,
                        )
                d_info = gr.Markdown("")

                d_btn.click(
                    fn=detect_faces_ui,
                    inputs=[d_img],
                    outputs=[d_output, d_info],
                )

            # ========== Tab 4: 关于 ==========
            with gr.TabItem("ℹ️ 关于"):
                gr.Markdown(
                    """
                    ## AI 换脸工具

                    ### 技术栈
                    - **人脸检测/识别**: insightface (buffalo_l 模型)
                    - **换脸模型**: inswapper_128
                    - **增强**: GFPGAN (可选)
                    - **界面**: Gradio
                    - **视频处理**: OpenCV + FFmpeg

                    ### 使用提示
                    1. **源照片** — 使用正面、光照均匀、表情自然的照片效果最好
                    2. **目标照片** — 角度和光照与源照片匹配时效果最佳
                    3. **视频处理** — 较慢，建议先短时间测试
                    4. **GFPGAN 增强** — 提高面部清晰度但大幅增加处理时间

                    ### 隐私声明
                    所有处理在本地完成，不会上传任何图像到互联网。
                    """
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
