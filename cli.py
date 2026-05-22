"""
AI 换脸 — 命令行接口

支持:
  - 单张图片换脸
  - 批量图片换脸 (目录)
  - 视频换脸
  - 人脸检测 + 可视化
  - 模型下载
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm

from face_swapper import FaceSwapper, get_engine
from face_swapper.utils import (
    load_image,
    save_image,
    is_image_file,
    is_video_file,
    draw_face_boxes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_swap(args: argparse.Namespace):
    """图片换脸命令"""
    logger.info(f"加载源图: {args.source}")
    source_img = load_image(args.source)
    if source_img is None:
        logger.error(f"无法加载源图: {args.source}")
        sys.exit(1)

    logger.info(f"加载目标图: {args.target}")
    target_img = load_image(args.target)
    if target_img is None:
        logger.error(f"无法加载目标图: {args.target}")
        sys.exit(1)

    engine = get_engine(det_threshold=args.threshold, use_gpu=not args.cpu)

    try:
        result = engine.swap(
            source_img,
            target_img,
            source_face_idx=args.source_face,
            target_face_idx=args.target_face,
            enhance=args.enhance,
            blend=not args.no_blend,
        )
    except ValueError as e:
        logger.error(f"换脸失败: {e}")
        sys.exit(1)

    save_image(args.output, result)
    logger.info(f"结果已保存: {args.output}")


def cmd_batch(args: argparse.Namespace):
    """批量换脸命令"""
    source_img = load_image(args.source)
    if source_img is None:
        logger.error(f"无法加载源图: {args.source}")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    files = [f for f in input_dir.iterdir() if f.is_file() and is_image_file(str(f))]
    if not files:
        logger.error(f"输入目录无图片: {input_dir}")
        sys.exit(1)

    logger.info(f"找到 {len(files)} 张图片，开始批量处理...")

    engine = get_engine(det_threshold=args.threshold, use_gpu=not args.cpu)

    for i, file_path in enumerate(tqdm(files, desc="批量换脸")):
        try:
            target_img = load_image(str(file_path))
            if target_img is None:
                logger.warning(f"跳过: {file_path}")
                continue

            result = engine.swap(
                source_img,
                target_img,
                source_face_idx=args.source_face,
                target_face_idx=args.target_face,
                enhance=args.enhance,
                blend=not args.no_blend,
            )

            output_path = output_dir / f"swapped_{file_path.name}"
            save_image(str(output_path), result)

        except Exception as e:
            logger.warning(f"处理失败 {file_path}: {e}")

    logger.info(f"批量完成! 结果保存到: {output_dir}")


def cmd_video(args: argparse.Namespace):
    """视频换脸命令"""
    source_img = load_image(args.source)
    if source_img is None:
        logger.error(f"无法加载源图: {args.source}")
        sys.exit(1)

    if not os.path.isfile(args.video):
        logger.error(f"视频文件不存在: {args.video}")
        sys.exit(1)

    engine = get_engine(det_threshold=args.threshold, use_gpu=not args.cpu)

    from face_swapper.video_processor import VideoProcessor

    processor = VideoProcessor(swapper=engine)

    max_frames = 0
    if args.seconds > 0:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        max_frames = int(fps * args.seconds)

    logger.info(f"开始视频换脸: {args.video}")
    logger.info(f"输出: {args.output}")

    try:
        result_path = processor.process_video(
            source_img=source_img,
            video_path=args.video,
            output_path=args.output,
            source_face_idx=args.source_face,
            target_face_idx=0,
            enhance=args.enhance,
            blend=not args.no_blend,
            max_frames=max_frames,
            keep_audio=not args.no_audio,
        )
        logger.info(f"视频换脸完成: {result_path}")
    except Exception as e:
        logger.error(f"视频换脸失败: {e}")
        sys.exit(1)


def cmd_detect(args: argparse.Namespace):
    """人脸检测命令"""
    img = load_image(args.image)
    if img is None:
        logger.error(f"无法加载图像: {args.image}")
        sys.exit(1)

    engine = get_engine(det_threshold=args.threshold, use_gpu=not args.cpu)
    faces_info = engine.get_face_info(img)

    if not faces_info:
        logger.info("未检测到人脸")
        return

    logger.info(f"检测到 {len(faces_info)} 张人脸:")
    for f in faces_info:
        logger.info(
            f"  #{f['idx']}: [年龄≈{f['age']}, 性别={f['gender']}, "
            f"置信度={f['confidence']:.3f}] "
            f"框={f['bbox']}"
        )

    if args.output:
        result = draw_face_boxes(img, faces_info)
        save_image(args.output, result)
        logger.info(f"标注图已保存: {args.output}")


def cmd_web(args: argparse.Namespace):
    """启动 Web UI"""
    from app import build_app

    logger.info(f"启动 Web UI (端口={args.port})...")
    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="AI 换脸工具 — 命令行版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单张图片换脸
  python cli.py swap source.jpg target.jpg -o result.jpg

  # 开启 GFPGAN 增强
  python cli.py swap source.jpg target.jpg -o result.jpg --enhance

  # 批量处理
  python cli.py batch source.jpg ./input_dir/ -o ./output_dir/

  # 视频换脸 (前 30 秒)
  python cli.py video source.jpg video.mp4 -o output.mp4 --seconds 30

  # 人脸检测
  python cli.py detect photo.jpg -o detected.jpg

  # 启动 Web UI
  python cli.py web
        """,
    )
    parser.add_argument(
        "--cpu", action="store_true", help="强制使用 CPU (不使用 GPU)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="人脸检测阈值 (0-1, 默认 0.5)"
    )
    parser.add_argument(
        "--enhance", action="store_true", help="使用 GFPGAN 增强输出质量"
    )
    parser.add_argument(
        "--no-blend", action="store_true", help="禁用无缝融合"
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # swap
    p_swap = subparsers.add_parser("swap", help="单张图片换脸")
    p_swap.add_argument("source", help="源图像 (提供人脸)")
    p_swap.add_argument("target", help="目标图像 (被换脸)")
    p_swap.add_argument("-o", "--output", default="output.jpg", help="输出路径")
    p_swap.add_argument("--source-face", type=int, default=0, help="源图中第几张人脸")
    p_swap.add_argument("--target-face", type=int, default=0, help="目标图中第几张人脸")
    p_swap.set_defaults(func=cmd_swap)

    # batch
    p_batch = subparsers.add_parser("batch", help="批量图片换脸")
    p_batch.add_argument("source", help="源图像")
    p_batch.add_argument("input_dir", help="输入目录")
    p_batch.add_argument("-o", "--output-dir", default="./output", help="输出目录")
    p_batch.add_argument("--source-face", type=int, default=0, help="源图中第几张人脸")
    p_batch.add_argument("--target-face", type=int, default=0, help="目标图中第几张人脸")
    p_batch.set_defaults(func=cmd_batch)

    # video
    p_video = subparsers.add_parser("video", help="视频换脸")
    p_video.add_argument("source", help="源图像")
    p_video.add_argument("video", help="目标视频")
    p_video.add_argument("-o", "--output", default="output_video.mp4", help="输出视频路径")
    p_video.add_argument("--source-face", type=int, default=0, help="源图中第几张人脸")
    p_video.add_argument("--seconds", type=int, default=0, help="处理前 N 秒 (0=全部)")
    p_video.add_argument("--no-audio", action="store_true", help="不保留原音频")
    p_video.set_defaults(func=cmd_video)

    # detect
    p_detect = subparsers.add_parser("detect", help="人脸检测")
    p_detect.add_argument("image", help="输入图像")
    p_detect.add_argument("-o", "--output", help="输出标注图像 (可选)")
    p_detect.set_defaults(func=cmd_detect)

    # web
    p_web = subparsers.add_parser("web", help="启动 Web 界面")
    p_web.add_argument("--host", default="127.0.0.1", help="监听地址")
    p_web.add_argument("--port", type=int, default=7860, help="端口号")
    p_web.add_argument("--share", action="store_true", help="生成公共链接 (share=True)")
    p_web.set_defaults(func=cmd_web)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 延迟导入 cv2 (为视频命令)
    global cv2
    import cv2

    args.func(args)


if __name__ == "__main__":
    main()
