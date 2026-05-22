"""
AI 换脸 — 统一入口

用法:
  python main.py            # 启动 Web UI (默认)
  python main.py web        # 启动 Web UI
  python main.py swap ...   # 命令行换脸
  python main.py video ...  # 视频换脸
  python main.py batch ...  # 批量换脸
  python main.py detect ... # 人脸检测
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

BANNER = """
========================================
       AI Face Swap v1.0
   基于 insightface + GFPGAN
========================================
"""


def main():
    print(BANNER)

    # 直接运行不带参数 -> Web UI
    if len(sys.argv) == 1 or sys.argv[1] not in ("swap", "batch", "video", "detect", "web"):
        # 启动 Web UI
        if len(sys.argv) > 1 and sys.argv[1] == "--help":
            print("用法:")
            print("  python main.py             启动 Web UI")
            print("  python main.py web         启动 Web UI")
            print("  python main.py swap ...    图片换脸")
            print("  python main.py video ...   视频换脸")
            print("  python main.py batch ...   批量换脸")
            print("  python main.py detect ...  人脸检测")
            print("  python main.py --help      显示此帮助")
            return

        from app import build_app

        logger.info("启动 Web UI @ http://127.0.0.1:7860")
        app = build_app()
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
        )
    else:
        # 委托给 CLI
        from cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
