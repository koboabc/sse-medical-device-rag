"""两份检索评测结果并排比 —— 逐题看「谁把哪道题救回来了、又弄坏了哪道」。

为什么不能只比汇总数字：汇总把 20 道题压成 5 个数，涨跌会互相抵消。本项目
最典型的一次误读就发生在这里 —— 混合路 hit@10=0.05 **低于**单 BM25 的 0.15，
看着像「融合有害」；逐题看才发现是 RRF 把 BM25 的 3 个命中稀释成了 1 个，
而向量路 20 条全是噪声。只看总数会得出「拆掉融合」的结论，看逐题才知道
该修的是向量路本身。

两份结果的 `routes` 名（bm25/vector/hybrid）必须对齐；不同索引变体、不同
消融开关的结果都可以比，差异会打在标题上。

用法：
    python scripts/compare_eval.py A.json B.json
    python scripts/compare_eval.py A.json B.json --routes bm25
    python scripts/compare_eval.py --latest            # 最近两份
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"


def load(p: str | Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def tag_of(d: dict) -> str:
    """一行标签：索引变体 + 消融开关，够区分同一天跑的多份结果。"""
    bits = [Path(d.get("index") or "index").name or "index"]
    bits.append(f"结构化{d.get('structured', '?')}")
    bits.append(f"加权{d.get('weights', '?')}")
    return f"{bits[0]}/{bits[1]}/{bits[2]}"


def main() -> int:
    ap = argparse.ArgumentParser(description="并排比两份评测结果")
    ap.add_argument("a", nargs="?", default=None)
    ap.add_argument("b", nargs="?", default=None)
    ap.add_argument("--latest", action="store_true", help="用最新两份")
    ap.add_argument("--routes", default=None, help="只比某一路（如 bm25）")
    args = ap.parse_args()

    if args.latest or not (args.a and args.b):
        files = sorted(RESULTS.glob("eval_retrieval*.json"), key=lambda p: p.stat().st_mtime)
        if len(files) < 2:
            raise SystemExit(f"结果不足两份（{RESULTS}）")
        args.a, args.b = str(files[-2]), str(files[-1])
    A, B = load(args.a), load(args.b)
    print(f"A = {Path(args.a).name}   [{tag_of(A)}]")
    print(f"B = {Path(args.b).name}   [{tag_of(B)}]")

    names = [n for n in A["routes"] if n in B["routes"]]
    if args.routes:
        names = [n for n in names if n == args.routes]
    if not names:
        raise SystemExit(f"两路名无交集：{list(A['routes'])} vs {list(B['routes'])}")

    for name in names:
        ra, rb = A["routes"][name], B["routes"][name]
        print(f"\n{'=' * 74}\n【{name}】")
        oa, ob = ra["overall"], rb["overall"]
        for k in ("hit@5", "hit@10", "recall@5", "recall@10", "mrr@10"):
            va, vb = oa.get(k), ob.get(k)
            if va is None or vb is None:
                continue
            d = vb - va
            arrow = "→" if abs(d) < 1e-9 else ("↑" if d > 0 else "↓")
            print(f"  {k:10s} {va:.3f} {arrow} {vb:.3f}   {d:+.3f}")
        # 两边都得有才比：旧结果文件里没这个字段，缺字段会被误读成「0.0」
        if "evidence_companies" in oa and "evidence_companies" in ob:
            print(f"  证据覆盖公司  {oa.get('evidence_companies', 0):.1f} → "
                  f"{ob.get('evidence_companies', 0):.1f} /17")

        qa = {r["qid"]: r for r in ra["per_question"]}
        qb = {r["qid"]: r for r in rb["per_question"]}
        fixed = [q for q in qa if q in qb and not qa[q]["hit@10"] and qb[q]["hit@10"]]
        broke = [q for q in qa if q in qb and qa[q]["hit@10"] and not qb[q]["hit@10"]]
        print(f"  救回 {len(fixed)} 题：{fixed or '—'}")
        print(f"  弄坏 {len(broke)} 题：{broke or '—'}")
        for q in fixed + broke:
            a_, b_ = qa[q], qb[q]
            print(f"    {q} [{a_['type']}] rank {a_['first_gold_rank'] or '—'}"
                  f" → {b_['first_gold_rank'] or '—'}"
                  f" | gold 覆盖 {a_['gold_covered']}/{a_['n_gold']}"
                  f" → {b_['gold_covered']}/{b_['n_gold']}")
            if a_.get("structured") or b_.get("structured"):
                print(f"       结构化：A={a_.get('structured')} B={b_.get('structured')}")
            str_a = "、".join(a_["top_ids"][:4])
            str_b = "、".join(b_["top_ids"][:4])
            if str_a != str_b:
                print(f"       A 前4: {str_a}")
                print(f"       B 前4: {str_b}")
        moved = [q for q in qa if q in qb and q not in fixed + broke
                 and qa[q]["first_gold_rank"] != qb[q]["first_gold_rank"]]
        if moved:
            print(f"  名次变化 {len(moved)} 题：" + "，".join(
                f"{q} #{qa[q]['first_gold_rank'] or '—'}→#{qb[q]['first_gold_rank'] or '—'}"
                for q in moved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
