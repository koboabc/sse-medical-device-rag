"""检索评测：hit@5/10、recall@5/10、MRR@10 —— BM25 / 向量 / 混合三路对照。

为什么要三路对照：这一层没有 reranker、没有 query rewrite，能调的东西只有
「用哪路召回 + 融合 + 软加权」。要判断某个改动到底有没有用，就必须先有一张
**同一套题**上的三路基线，否则任何调参都是凭感觉。

几个口径上的决定，都会影响数字，所以写在代码里而不是文档里：

* **命中判定含「内容重复被合并掉的块」**。语料里有 39 条 chunk 正文两两相同
  （同一张表跨页切了两块）。融合时排名靠后的那块会被合并掉，它的 chunk_id
  挂在保留块的 `dup_chunk_ids` 上。若 gold 标的正是被合并的那块，不该算漏召回
  —— 所以判定时把 `dup_chunk_ids` 一并算作覆盖。
* **软加权三路都开**。公司/章节加权是本管线的一部分，不是「某一路专属」；
  关掉它跑出来的基线不对应线上实际行为。要做消融就跑 `--no-weights`。
* **拒答陷阱题不进检索指标**。它们没有 gold，命中与否无意义；这里只统计
  「召回了多少条证据」，供人工判断「模型手上有没有可能编出答案的素材」。
  真正的拒答判定要 LLM 参与，见 scripts/eval_generation.py。
  注意 T02（「17 家公司的员工离职率」）**会触发结构化路由**（它确实是「要一个
  集合」的问法），于是拿到 17 张表 —— 但没有一张含离职率。这反而让拒答测试更严：
  模型得在 17 张表面前说「语料里没有这个指标」，而不是靠证据太少蒙过去。
* **gold 条数 > 评测深度时 recall 上不去是结构性的**。例如条件筛选题 gold 有
  9 条而 Top-K 取 10，recall 天然≤1；这不是召回失败，是深度上限。报告里会把
  gold 条数一并打出来，避免误读。

用法：
    python scripts/eval_retrieval.py                      # 三路，Top-10
    python scripts/eval_retrieval.py --routes bm25         # 只跑 BM25（不载模型）
    python scripts/eval_retrieval.py --show-misses         # 打印没命中的题，便于归因
    python scripts/eval_retrieval.py --no-weights          # 消融：关掉软加权
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.retriever import NO_WEIGHTS, Weights, retrieve  # noqa: E402
from src.retrieval.store import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "datasets" / "qa.jsonl"
RESULTS = ROOT / "evaluation" / "results"

ROUTE_SETS = {"bm25": ("bm25",), "vector": ("vector",), "hybrid": ("bm25", "vector")}


def load_dataset(path: Path) -> list[dict]:
    items = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{ln} 不是合法 JSON：{e}")
    return items


def covered_by(hit, gold: set[str]) -> str | None:
    """这条命中是否覆盖了某个 gold（含内容重复被合并掉的块）。返回命中的 gold id。"""
    if hit.chunk_id in gold:
        return hit.chunk_id
    for cid in hit.dup_chunk_ids:
        if cid in gold:
            return cid
    return None


def eval_one(question: dict, routes: tuple[str, ...], *, top_k: int,
             k_bm25: int, k_vector: int, weights: Weights | None,
             structured: bool | None, st) -> dict:
    t0 = time.monotonic()
    res = retrieve(question["question"], routes=routes, top_k=top_k,
                   k_bm25=k_bm25, k_vector=k_vector, weights=weights,
                   structured=structured, st=st)
    dt = time.monotonic() - t0

    gold = set(question["gold_chunk_ids"])
    covered: list[str] = []          # 按名次顺序被覆盖的 gold
    first_rank: int | None = None
    for rank, h in enumerate(res.hits, 1):
        got = covered_by(h, gold)
        if got and got not in covered:
            covered.append(got)
            if first_rank is None:
                first_rank = rank

    def hit_at(k: int) -> int:
        return int(first_rank is not None and first_rank <= k)

    def recall_at(k: int) -> float:
        if not gold:
            return 0.0
        n = sum(1 for h in res.hits[:k] if covered_by(h, gold))
        return n / len(gold)

    # 证据集实际覆盖了几家公司 —— **不循环**的效果度量。
    # 为什么需要它：条件筛选题的 gold 就是「17 家公司的会计数据表」，而结构化路
    # 取的就是那一组，所以「hit@10 从 0 到 1」在定义上必然发生，不能拿它当效果
    # 证据（那是同义反复）。真正有信息量的是：证据里到齐了几家公司的表 ——
    # 少一家就漏掉一个正确答案，而这一条与 gold 的标注方式无关。
    n_ev = len(res.hits) if res.structured else min(top_k, len(res.hits))
    codes_ev = {st.df["stock_code"].iloc[h.row] for h in res.hits[:n_ev]}

    return {
        "qid": question["qid"], "type": question["type"],
        "question": question["question"],
        "n_gold": len(gold), "n_returned": len(res.hits),
        "gold_covered": len(covered),
        "first_gold_rank": first_rank,
        "hit@5": hit_at(5), "hit@10": hit_at(10),
        "recall@5": round(recall_at(5), 4), "recall@10": round(recall_at(10), 4),
        "mrr@10": round(1.0 / first_rank, 4) if first_rank and first_rank <= 10 else 0.0,
        "recall_counts": res.recall_counts,
        "routes": "/".join(res.routes),
        "structured": res.structured,
        "n_evidence": n_ev,
        "n_companies_in_evidence": len(codes_ev),
        "sec": round(dt, 2),
        "top_ids": [h.chunk_id for h in res.hits],
    }


def summarize(rows: list[dict], gold_key: str = "gold_chunk_ids") -> dict:
    """对一组题算平均。按 gold 条数是否超过评测深度分组，避免把结构上限当召回失败。"""
    if not rows:
        return {}
    n = len(rows)
    return {
        "n": n,
        "hit@5": round(sum(r["hit@5"] for r in rows) / n, 4),
        "hit@10": round(sum(r["hit@10"] for r in rows) / n, 4),
        "recall@5": round(sum(r["recall@5"] for r in rows) / n, 4),
        "recall@10": round(sum(r["recall@10"] for r in rows) / n, 4),
        "mrr@10": round(sum(r["mrr@10"] for r in rows) / n, 4),
        "n_structured": sum(1 for r in rows if r.get("structured")),
        "evidence_companies": round(
            sum(r["n_companies_in_evidence"] for r in rows) / n, 2),
    }


def fmt(name: str, s: dict) -> str:
    if not s:
        return f"  {name:12s} —"
    extra = ""
    if s.get("n_structured"):
        extra = (f"  结构化 {s['n_structured']}/{s['n']} 题，"
                 f"证据覆盖公司均 {s['evidence_companies']:.1f}/17")
    return (f"  {name:12s} n={s['n']:2d}  "
            f"hit@5={s['hit@5']:.3f}  hit@10={s['hit@10']:.3f}  "
            f"recall@5={s['recall@5']:.3f}  recall@10={s['recall@10']:.3f}  "
            f"MRR@10={s['mrr@10']:.3f}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(description="检索评测：三路对照")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--routes", default="bm25,vector,hybrid",
                    help="逗号分隔：bm25 / vector / hybrid，或 all")
    ap.add_argument("--top-k", type=int, default=10, help="评测取前多少条算命中")
    ap.add_argument("--k-bm25", type=int, default=20, help="BM25 路召回深度")
    ap.add_argument("--k-vector", type=int, default=20, help="向量路召回深度")
    ap.add_argument("--no-weights", action="store_true", help="消融：关掉软加权")
    ap.add_argument("--no-structured", action="store_true",
                    help="消融：关掉结构化表组路由（条件筛选题退回相似度 Top-K）")
    ap.add_argument("--only", default=None, help="只跑指定 qid 前缀，如 Q1 或 T")
    ap.add_argument("--show-misses", action="store_true", help="打印未命中的题与返回的 top ids")
    ap.add_argument("--no-save", action="store_true", help="不落盘结果")
    ap.add_argument("--index", default=None,
                    help="索引目录（默认 config.index_dir）。比新旧两套用 "
                         "--index index/header —— chunk_id 行序两套一致，所以同一套题"
                         "可直接对比，只是召回所依据的向量/词频不同")
    args = ap.parse_args()

    items = load_dataset(Path(args.dataset))
    if args.only:
        items = [it for it in items if it["qid"].startswith(args.only)]
    asked = [it for it in items if it["gold_chunk_ids"]]
    traps = [it for it in items if not it["gold_chunk_ids"]]
    print(f"数据集 {Path(args.dataset).name}：{len(items)} 题"
          f"（{len(asked)} 题有 gold + {len(traps)} 题拒答陷阱）")

    names = (list(ROUTE_SETS) if args.routes == "all"
             else [r.strip() for r in args.routes.split(",") if r.strip()])
    for r in names:
        if r not in ROUTE_SETS:
            raise SystemExit(f"未知路名 {r!r}，可选 {list(ROUTE_SETS)}")
    weights = NO_WEIGHTS if args.no_weights else None
    print(f"软加权：{'关（消融）' if args.no_weights else '开（公司 1.5 / 章节 1.2 / 表格 1.15）'}"
          f"｜结构化表组：{'关（消融）' if args.no_structured else '开'}"
          f"｜评测深度 Top-{args.top_k}｜召回深度 BM25 {args.k_bm25} / 向量 {args.k_vector}")

    st = store(args.index)
    print(f"索引：{st.describe()}")
    # 索引变体写进报告文件名与正文：同一套题会跑出多份结果，
    # 光靠时间戳区分不出哪份是「表头索引」，对比时就会拿错。
    variant = args.index or "default"

    all_rows: dict[str, list[dict]] = {}
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "dataset": Path(args.dataset).name, "n_questions": len(asked),
              "index": str(st.index_dir), "index_variant": variant,
              "top_k": args.top_k, "k_bm25": args.k_bm25, "k_vector": args.k_vector,
              "weights": "off" if args.no_weights else "on",
              "structured": "off" if args.no_structured else "on", "routes": {}}

    for name in names:
        routes = ROUTE_SETS[name]
        print(f"\n{'=' * 78}\n跑 {name}（routes={routes}）……")
        rows = []
        for it in asked:
            row = eval_one(it, routes, top_k=args.top_k, k_bm25=args.k_bm25,
                           k_vector=args.k_vector, weights=weights,
                           structured=not args.no_structured, st=st)
            rows.append(row)
            flag = "✅" if row["hit@10"] else "❌"
            mark = "🗂" if row["structured"] else "  "
            print(f"  {flag}{mark} {row['qid']} {row['type'][:5]:5s} "
                  f"gold {row['gold_covered']}/{row['n_gold']} "
                  f"首命中 #{row['first_gold_rank'] or '—'} "
                  f"证据 {row['n_evidence']} 条/{row['n_companies_in_evidence']} 家 "
                  f"({row['sec']}s) {row['question'][:30]}")
        all_rows[name] = rows
        by_type = defaultdict(list)
        for r in rows:
            by_type[r["type"]].append(r)
        report["routes"][name] = {
            "overall": summarize(rows),
            "by_type": {t: summarize(v) for t, v in by_type.items()},
            "per_question": rows,
        }
        # 陷阱题：无 gold，只统计手上拿到多少素材
        tr = []
        for it in traps:
            res = retrieve(it["question"], routes=routes, top_k=args.top_k,
                           k_bm25=args.k_bm25, k_vector=args.k_vector,
                           weights=weights, structured=not args.no_structured, st=st)
            tr.append({"qid": it["qid"], "n_returned": len(res.hits),
                       "top_ids": [h.chunk_id for h in res.hits]})
        report["routes"][name]["traps"] = tr

    print(f"\n{'=' * 78}\n汇总（{len(asked)} 题有 gold，深度 Top-{args.top_k}）")
    for name in names:
        r = report["routes"][name]
        print(f"\n【{name}】")
        print(fmt("全部", r["overall"]))
        for t, s in r["by_type"].items():
            print(fmt(t[:6], s))
        multi = [x for x in r["per_question"] if x["n_gold"] > args.top_k]
        if multi:
            print(f"  ⚠ gold 条数 > Top-{args.top_k} 的题 "
                  f"{[x['qid'] for x in multi]}：recall 上限被深度卡住，不是召回失败")
        print("  拒答陷阱（无 gold，只报拿到几条素材）：" +
              " ｜ ".join(f"{t['qid']} {t['n_returned']} 条" for t in r["traps"]))

    if args.show_misses:
        print(f"\n{'=' * 78}\n未命中（hit@10=0）的题：")
        for name in names:
            bad = [x for x in report["routes"][name]["per_question"] if not x["hit@10"]]
            if not bad:
                print(f"\n【{name}】无 —— 全部命中")
                continue
            print(f"\n【{name}】{len(bad)} 题：")
            for x in bad:
                print(f"  {x['qid']} gold={x['n_gold']} 覆盖={x['gold_covered']}"
                      f"  实际返回：{'、'.join(x['top_ids'][:6])}")

    if not args.no_save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = "" if variant == "default" else f"_{Path(variant).name}"
        p = RESULTS / f"eval_retrieval{tag}_{stamp}.json"
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写出 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
