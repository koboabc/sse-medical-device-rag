"""纯 token 统计：不加载模型，只看 batch 内的 padding 浪费有多少。

Transformer 前向的开销 ≈ 与 batch 内总 token 数成正比（padding 也算）。
所以 batch 策略的好坏，直接看「padding 后 token / 真实 token」这个比值。

比较两种策略：
  顺序 batch（当前实现）    : 按 chunk_id 连续切 32 条，pad 到批内最长
  按长度排序 batch（拟改）  : 先按 token 长度排序，批内长度接近，浪费小

用法：
    python scripts/probe_batching.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.indexing.build_index import load_chunks  # noqa: E402
from src.indexing.embedder import Embedder  # noqa: E402


def pad_waste(lengths: np.ndarray, bs: int) -> tuple[float, float]:
    """返回 (有效token, padding后token)。"""
    eff = pad = 0
    for i in range(0, len(lengths), bs):
        chunk = lengths[i:i + bs]
        m = int(chunk.max())
        eff += int(chunk.sum())
        pad += m * len(chunk)
    return eff, pad


def main() -> int:
    df = load_chunks()
    e = Embedder(batch_size=8)
    tok = e.model.tokenizer
    n = len(df)

    t = time.monotonic()
    lens = np.array([
        len(tok(s, truncation=True, max_length=e.max_seq_length)["input_ids"])
        for s in df["content"]
    ], dtype=np.int64)
    print(f"分词 {n} 篇耗时 {time.monotonic() - t:.1f}s")
    print(f"token 总数 {lens.sum():,}  中位 {np.median(lens):.0f} "
          f"p95 {np.percentile(lens, 95):.0f} max {lens.max():,}")
    for thr in (512, 1024, 2048, 4096, 8192):
        c = int((lens > thr).sum())
        print(f"  > {thr:>5} token: {c:>4} 篇  "
              f"占 token 总量 {lens[lens > thr].sum() / lens.sum() * 100:5.1f}%")

    print(f"\n{'batch':>6} {'顺序 有效/填充=倍数':>26} {'排序 有效/填充=倍数':>26}")
    srt = np.sort(lens)[::-1]          # 长的先，批内最接近
    for bs in (8, 16, 32, 64):
        e1, p1 = pad_waste(lens, bs)
        e2, p2 = pad_waste(srt, bs)
        print(f"{bs:>6} {e1:>14,}/{p1:>12,}={p1 / e1:>6.2f}x "
              f"{e2:>14,}/{p2:>12,}={p2 / e2:>6.2f}x")

    # padding 是按「批内最长」算的，长块会拖累同批的短块：
    # 看有多少 token 是被超长表格块拖进 padding 的
    long_mask = lens > 2000
    print(f"\n>2000 token 的块 {int(long_mask.sum())} 篇，"
          f"占真实 token 的 {lens[long_mask].sum() / lens.sum() * 100:.1f}%")

    # 二次项成本：注意力 ~ n^2。用实测点标定 time(n)=a*n+b*n^2
    # 实测：8913 token 单篇 414.6s；典型批 137.9 tok/s
    n_big, t_big = 8913, 414.6
    n_typ, t_typ = 474, 1 / 137.9 * 474
    A = np.array([[n_big, n_big ** 2], [n_typ, n_typ ** 2]])
    b_coef, a_coef = np.linalg.solve(A, np.array([t_big, t_typ]))
    print(f"\n标定 t(n) = {b_coef:.6f}*n + {a_coef:.3e}*n^2  (秒/篇)")

    # 全量成本：批内每条都按「批内最长」算，padding 与注意力都要算进去
    def cost(lengths: np.ndarray, bs: int) -> float:
        c = 0.0
        for i in range(0, len(lengths), bs):
            m = int(lengths[i:i + bs].max())
            c += len(lengths[i:i + bs]) * (b_coef * m + a_coef * m * m)
        return c
    for bs in (8, 16, 32, 64):
        c1, c2 = cost(lens, bs), cost(srt, bs)
        print(f"batch={bs:>3}: 顺序 {c1 / 3600:>5.2f}h   "
              f"排序 {c2 / 3600:>5.2f}h   "
              f"省 {c1 / max(c2, 1e-9):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
