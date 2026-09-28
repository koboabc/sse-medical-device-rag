"""实测 Qwen3-Embedding-0.6B 在本机的编码吞吐，用于估算全量耗时。

结论要看两个量：
  * 每篇的 token 数（决定线性的矩阵乘开销）
  * 批量大小与线程数（决定 CPU 利用率）

用法：
    python scripts/probe_embed_speed.py --n 8 --threads 8
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.indexing.build_index import load_chunks  # noqa: E402
from src.indexing.embedder import Embedder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="抽样条数")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"线程 {torch.get_num_threads()} / batch {args.batch_size}", flush=True)

    df = load_chunks()
    e = Embedder(batch_size=args.batch_size)
    tok = e.model.tokenizer

    def ntok(s: str) -> int:
        return len(tok(s, truncation=True, max_length=e.max_seq_length)["input_ids"])

    typ = df[(df.char_count >= 500) & (df.char_count <= 1000)].sample(
        args.n, random_state=1)
    tks = [ntok(t) for t in typ.content]
    print(f"典型块 {len(tks)} 个，token 数 {tks}（合计 {sum(tks)}）", flush=True)

    t = time.monotonic()
    e.encode_documents(typ.content.tolist(), batch_size=args.batch_size,
                       show_progress=False)
    dt = time.monotonic() - t
    rate = sum(tks) / dt
    print(f"  → {dt:.1f}s  {len(tks) / dt:.2f} 条/秒  {rate:.1f} tok/s", flush=True)

    # 全量估算：用真实 token 分布，而不是拿抽样均值外推
    all_tok = sum(ntok(c) for c in df["content"].sample(
        min(200, len(df)), random_state=7))
    est_tok = all_tok / min(200, len(df)) * len(df)
    print(f"\n全量 token 估算（200 抽样）: {est_tok:,.0f}", flush=True)
    print(f"按 {rate:.1f} tok/s 外推全量: {est_tok / rate / 60:.0f} 分钟 "
          f"= {est_tok / rate / 3600:.1f} 小时", flush=True)

    # 最长块单独测：二次方注意力在这里最贵
    big = df.nlargest(1, "char_count").iloc[0]
    n = ntok(big.content)
    t = time.monotonic()
    e.encode_documents([big.content], batch_size=1, show_progress=False)
    dt = time.monotonic() - t
    print(f"\n最长块 {big.chunk_id} {big.char_count} 字 = {n} token："
          f"{dt:.1f}s（{n / dt:.1f} tok/s）", flush=True)
    print(f"  仅此一块就占全量时间的 {dt / (est_tok / rate) * 100:.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
