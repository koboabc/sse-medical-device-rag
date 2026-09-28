"""环境自检：确认 Python / PyTorch / CUDA 及 RAG 依赖可用性。

用法：
    python scripts/check_env.py

本脚本只做只读探测，不安装任何东西。缺什么会打印对应的安装建议。
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _installed_version(dist: str) -> str | None:
    """问包元数据要版本号，不看 import 能不能过。未安装返回 None。"""
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:  # noqa: BLE001
        return None


def _probe(mod: str) -> tuple[bool, str]:
    """返回 (能否 import, 说明)。

    刻意区分「未安装」和「装了但 import 失败」—— 后者在本机真实存在：
    base 环境里的 transformers==2.1.1 装了却导入即崩，笼统报成「未安装」
    会把排查方向带偏。
    """
    installed = _installed_version(mod)
    try:
        importlib.import_module(mod)
    except ImportError as e:
        if installed:
            return False, f"{installed} 已安装但 import 失败: {e}"
        return False, "未安装"
    except Exception as e:  # noqa: BLE001
        return False, f"import 报错: {type(e).__name__}: {e}"
    return True, installed or "已安装(版本号未暴露)"


def _module_version(mod: str, attr: str = "__version__") -> str | None:
    """返回模块版本号；未安装或不可导入返回 None。"""
    try:
        m = importlib.import_module(mod)
    except Exception:
        return None
    return getattr(m, attr, None) or _installed_version(mod)


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return Check(
        "Python",
        ok,
        f"{platform.python_version()}  ({sys.executable})",
        "需要 3.10+；本机 3.11.4 满足" if ok else "版本过低，建议 3.10+",
    )


def check_torch() -> Check:
    ver = _module_version("torch")
    if ver is None:
        return Check(
            "PyTorch",
            False,
            "未安装",
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
            "   （本机无 NVIDIA GPU，务必用 CPU 版，勿装 cu121/cu124）",
        )
    return Check("PyTorch", True, ver)


def check_cuda() -> Check:
    try:
        import torch
    except Exception:
        return Check("CUDA", False, "无法判定（PyTorch 未安装）", "先装 CPU 版 torch")
    if not torch.cuda.is_available():
        return Check(
            "CUDA",
            False,
            "不可用 —— 本机无 NVIDIA GPU，无 CUDA",
            "预期结果。embedding 走 CPU，生成走 API；不要尝试在本地跑 LLM",
        )
    return Check(
        "CUDA",
        True,
        f"{torch.version.cuda} / {torch.cuda.device_count()} 卡 / "
        f"{torch.cuda.get_device_name(0)}",
    )


def check_gpu_hardware() -> Check:
    """不依赖 torch，直接问系统要显卡型号。"""
    if sys.platform != "win32":
        return Check("GPU 硬件", False, "非 Windows，跳过 WMI 探测", "")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name -join '; '"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return Check("GPU 硬件", False, f"探测失败: {e}", "")
    has_nv = "nvidia" in out.lower()
    return Check(
        "GPU 硬件",
        has_nv,
        out or "未探测到",
        "" if has_nv else "无 NVIDIA 独显 → 与 CUDA 不可用一致",
    )


def check_disk() -> Check:
    try:
        usage = shutil.disk_usage(__file__)
        free_gb = usage.free / 1024**3
    except Exception as e:  # noqa: BLE001
        return Check("磁盘余量", False, f"探测失败: {e}", "")
    return Check(
        "磁盘余量",
        free_gb > 5,
        f"{free_gb:.1f} GB 可用",
        "半年报 PDF 单份 2-10MB，17 家约 100MB，另需向量索引空间" if free_gb > 5 else "余量偏低",
    )


# 依赖名 -> requirements.txt 中的用途说明
DEPENDENCIES = [
    ("fitz", "pymupdf：PDF 文本抽取"),
    ("pdfplumber", "pdfplumber：表格抽取兜底"),
    ("pandas", "pandas：数据处理"),
    ("numpy", "numpy：数值计算"),
    ("langchain_text_splitters", "langchain-text-splitters：文本切分"),
    ("sentence_transformers", "sentence-transformers：本地 embedding"),
    ("transformers", "transformers：模型加载"),
    ("faiss", "faiss-cpu：向量检索"),
    ("rank_bm25", "rank-bm25：关键词检索"),
    ("jieba", "jieba：中文分词"),
    ("openai", "openai：LLM API 调用"),
    ("dotenv", "python-dotenv：读取 .env"),
    ("streamlit", "streamlit：演示界面"),
]


def check_dependencies() -> list[Check]:
    checks = []
    for mod, desc in DEPENDENCIES:
        ok, detail = _probe(mod)
        checks.append(Check(desc, ok, detail))
    return checks


def main() -> int:
    print("=" * 68)
    print(" 上交所医疗器械财报 RAG —— 环境自检")
    print("=" * 68)

    core = [
        check_python(),
        check_torch(),
        check_cuda(),
        check_gpu_hardware(),
        check_disk(),
    ]

    print("\n【核心环境】")
    for c in core:
        print(f"  [{'OK ' if c.ok else '!! '}] {c.name:<12} {c.detail}")
        if c.hint and not c.ok:
            print(f"        → {c.hint}")

    deps = check_dependencies()
    print("\n【RAG 依赖】")
    for c in deps:
        print(f"  [{'OK ' if c.ok else '-- '}] {c.name:<34} {c.detail}")

    missing = [c for c in deps if not c.ok]
    print("\n" + "-" * 68)
    if missing:
        print(f"缺 {len(missing)} 项依赖。安装：")
        print("    python -m venv .venv && source .venv/Scripts/activate")
        print("    pip install -r requirements.txt")
        print("注意：请勿在 anaconda base 直接装 —— base 里是 transformers==2.1.1，")
        print("      与 sentence-transformers 所需 >=4.34 冲突。")
    else:
        print("依赖齐全。")
    print("-" * 68)

    # 核心项里 CUDA 不可用属预期，不计入失败
    hard_fail = [c for c in core[:2] if not c.ok]
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
