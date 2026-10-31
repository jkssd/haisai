#!/usr/bin/env python3
"""命令行入口（免安装）: python pose.py <cmd> [args...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from target_pose.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
