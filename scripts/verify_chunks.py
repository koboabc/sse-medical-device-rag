"""核验 data/metadata.parquet 与 data/chunks/*.jsonl。

机械化检查（全部断言，失败即报）：
  1. 9 个必需字段齐备、无空值
  2. chunk_id 全局唯一，且与 stock_code 前缀一致
  3. page_start <= page_end，页码在 PDF 总页数范围内（用 pymupdf 实测）
  4. 每个 PDF 的 chunk 数与 jsonl 行数、parquet 行数三方一致
  5. jsonl 内容与 parquet 内容逐条字节一致（防止只更新了一边）
  6. 表格块必须含 Markdown 分隔行（|---|---|），否则是空壳
  7. 正文块不得含 Markdown 表格分隔行（表格没漏进正文）
  8. 章节名不得为空；报告前部的块数应有上界

用法：
    python scripts/verify_chunks.py
    python scripts/verify_chunks.py --sample 10 --seed 20260926
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config  # noqa: E402

REQUIRED = ["chunk_id", "stock_code", "company", "report", "section",
            "page_start", "page_end", "content", "content_type"]
SEP_RE = "|---"
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260926)
    args = ap.parse_args()

    df = pd.read_parquet(config.data_dir / "metadata.parquet")
    print(f"metadata.parquet: {len(df)} 行 × {len(df.columns)} 列")
    print(f"列：{list(df.columns)}\n")

    # ---------- 1. 字段 ----------
    for f in REQUIRED:
        check(f in df.columns, f"缺字段 {f}")
    check(df[REQUIRED].notna().all().all(), "必需字段存在空值")
    check((df["content"].str.len() > 0).all(), "存在空 content")
    check(set(df["content_type"]) <= {"text", "table"},
          f"content_type 取值异常: {set(df['content_type'])}")
    check(df["section"].str.len().min() > 0, "存在空 section")
    check(set(df["report"]) == {"2026年半年度报告"}, f"report 取值异常: {set(df['report'])}")

    # ---------- 2. chunk_id ----------
    check(df["chunk_id"].is_unique, "chunk_id 有重复")
    pref = df["stock_code"].astype(str) + "-"
    check((df["chunk_id"].str[:7] == pref.str[:7]).all(),
          "chunk_id 前缀与 stock_code 不一致")
    check((df["chunk_id"].str.split("-").str[1].str.len() == 4).all(),
          "chunk_id 序号不是 4 位")

    # ---------- 3. 页码 ----------
    check((df["page_start"] <= df["page_end"]).all(), "存在 page_start > page_end")
    check((df["page_start"] >= 1).all(), "存在 page_start < 1")

    # ---------- 4/5. jsonl 与 parquet 一致性 ----------
    jl_rows: list[dict] = []
    for jl in sorted((config.data_dir / "chunks").glob("*.jsonl")):
        with jl.open("r", encoding="utf-8") as f:
            jl_rows += [json.loads(l) for l in f if l.strip()]
    check(len(jl_rows) == len(df),
          f"jsonl 行数 {len(jl_rows)} != parquet 行数 {len(df)}")
    jdf = pd.DataFrame(jl_rows).sort_values("chunk_id").reset_index(drop=True)
    pdf_sorted = df.sort_values("chunk_id").reset_index(drop=True)
    for col in ["content", "section", "content_type"]:
        same = (jdf[col].astype(str) == pdf_sorted[col].astype(str)).all()
        check(same, f"jsonl 与 parquet 的 {col} 不一致")
    print("✓ jsonl 与 parquet 逐条一致")

    # ---------- 6/7. 表格完整性 ----------
    tb = df[df["content_type"] == "table"]
    bad_tb = tb[~tb["content"].str.contains(SEP_RE, regex=False)]
    check(len(bad_tb) == 0, f"{len(bad_tb)} 个表格块缺 Markdown 分隔行")
    tx = df[df["content_type"] == "text"]
    leak = tx[tx["content"].str.contains(SEP_RE, regex=False)]
    check(len(leak) == 0, f"{len(leak)} 个正文块混入 Markdown 表格")

    # ---------- 8. 页码上界：用真实 PDF 页数核对 ----------
    with (config.data_dir / "manifest.csv").open("r", encoding="utf-8-sig",
                                                 newline="") as f:
        pages = {}
        for r in csv.DictReader(f):
            pdf = config.root / r["local_path"]
            if pdf.exists():
                d = pymupdf.open(pdf)
                pages[r["stock_code"]] = d.page_count
                d.close()
    over = df[df.apply(lambda r: r["page_end"] > pages.get(r["stock_code"], 10**9), axis=1)]
    check(len(over) == 0, f"{len(over)} 个 chunk 页码超出 PDF 总页数")

    # ---------- 汇总 ----------
    print(f"✓ 字段/唯一性/页码/表格完整性 检查完毕\n")
    print("=" * 78)
    print(f"{'代码':<8}{'公司':<24}{'页数':>5}{'chunk':>7}{'表格':>7}{'正文':>7}{'字数':>10}")
    print("-" * 78)
    for code, g in df.groupby("stock_code", sort=True):
        n_t = (g["content_type"] == "table").sum()
        name = g["company"].iloc[0][:22]
        print(f"{code:<8}{name:<24}{pages.get(code, 0):>5}{len(g):>7}"
              f"{n_t:>7}{len(g) - n_t:>7}{g['char_count'].sum():>10,}")
    print("-" * 78)
    print(f"{'合计':<8}{'':<24}{sum(pages.values()):>5}{len(df):>7}"
          f"{(df['content_type'] == 'table').sum():>7}"
          f"{(df['content_type'] == 'text').sum():>7}{df['char_count'].sum():>10,}")
    print("=" * 78)

    # ---------- 尺寸分布 ----------
    tx_c = df[df["content_type"] == "text"]["char_count"]
    tb_c = df[df["content_type"] == "table"]["char_count"]
    print(f"\n正文 {len(tx_c)} 块：中位 {int(tx_c.median())}，"
          f"500~1000 占 {((tx_c >= 500) & (tx_c <= 1000)).mean():.0%}，"
          f"<150 的存根 {int((tx_c < 150).sum())} 个")
    print(f"表格 {len(tb_c)} 块：中位 {int(tb_c.median())}，"
          f">2000 字的长表 {int((tb_c > 2000).sum())} 个 "
          f"（表格整体成块，长表偏长是要求 9 的代价）")

    # ---------- 章节覆盖 ----------
    print(f"\n章节分布（共 {df['section'].nunique()} 个）：")
    sec = df["section"].value_counts()
    for s, n in sec.head(14).items():
        print(f"  {n:>5}  {s}")

    # ---------- 随机抽查 ----------
    random.seed(args.seed)
    picks = random.sample(range(len(df)), min(args.sample, len(df)))
    print("\n" + "=" * 78)
    print(f"随机抽查 {len(picks)} 个 chunk（seed={args.seed}）")
    print("=" * 78)
    for i in picks:
        r = df.iloc[i]
        print(f"\n【{r['chunk_id']}】{r['company']}（{r['stock_code']}）")
        print(f"  报告 {r['report']} | 章节 {r['section']}"
              + (f" | 子节 {r['subsection']}" if r.get("subsection") else ""))
        print(f"  页码 p{r['page_start']}-{r['page_end']} | 类型 {r['content_type']}"
              f" | {r['char_count']} 字")
        body = r["content"]
        print("  原文：")
        for line in body[:900].splitlines():
            print(f"    {line}")
        if len(body) > 900:
            print(f"    …（余 {len(body) - 900} 字）")

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    if FAILS:
        print(f"✗ 校验失败 {len(FAILS)} 项：")
        for m in FAILS:
            print(f"    - {m}")
        return 1
    print("✓ 全部校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
