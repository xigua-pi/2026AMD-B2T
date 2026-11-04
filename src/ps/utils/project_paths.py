"""定位科研目录，使脚本不依赖开发机器的绝对盘符。"""
from pathlib import Path


def project_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "research_manifest.json").exists():
            return path
    raise FileNotFoundError("未找到 research_manifest.json，请保留科研目录结构")
