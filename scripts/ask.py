"""命令行问答入口（本阶段只做召回侧，生成见阶段 2）。

为什么先把「只召回」做成一个正经 CLI：
  * 生成要 API key、要花钱、要等网络。而**召回好不好，看块就知道** ——
    用 `--no-llm` 打印候选块的「公司/章节/页码/类型/字数 + 各路原始分」，
    人眼扫一眼就能判断「检索对不对题」，这是调权重和判标注时唯一可靠的反馈。
  * 评测集里的 gold 是靠它反复核对出来的，所以它必须能复现、能 --json 落盘。

用法：
    python scripts/ask.py --no-llm "联影医疗 2026 上半年营业收入是多少"
    python scripts/ask.py --no-llm -k 10 "哪些公司归母净利润同比下滑？原因是什么"
    python scripts/ask.py --no-llm --company 688271 "营业收入是多少"   # 显式硬过滤
    python scripts/ask.py --no-llm --routes bm25 --json out.jsonl "..."  # 评测用
    python scripts/ask.py --no-llm -i          # 交互模式

生成已接入：**去掉 `--no-llm` 就走完整链**（检索 → 证据 → DeepSeek → 带引用的答案）。
需要 `.env` 里的 `LLM_API_KEY`；缺 key 时会在检索前明确报错，而不是静默只召回。

    python scripts/ask.py "联影医疗 2026 上半年营业收入是多少"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config  # noqa: E402
from src.retrieval.entities import load_companies  # noqa: E402
from src.retrieval.qa import QAResult, answer_question  # noqa: E402
from src.retrieval.retriever import NO_WEIGHTS, Weights, retrieve  # noqa: E402
from src.retrieval.store import store  # noqa: E402


def fmt_hit(st, h, width: int = 180) -> str:
    """一条命中打成多行：首行是元数据，次行是正文摘要。

    正文要露出来 —— 只给「章节+页码」是看不出对不对题的，而这是本阶段
    唯一的判断依据。表格块的换行会被压成 ` / `，免得刷屏。
    """
    r = st.record(h.row)
    sc = " ".join(f"{k}={v:.4g}" for k, v in sorted(h.scores.items())
                  if k in ("bm25", "vector", "rrf", "weight", "final"))
    head = (f"  [{h.rank}] {r['chunk_id']}  {r['company'][:14]} | {r['section']} | "
            f"p{r['page_start']}-{r['page_end']} | {r['content_type']} | "
            f"{r['char_count']}字 | row={h.row}")
    body = r["content"][:width].replace("\n", " / ")
    dup = f"\n       （同内容另见 {', '.join(h.dup_chunk_ids)}）" if h.dup_chunk_ids else ""
    return f"{head}\n       {sc}\n       {body}{'…' if r['char_count'] > width else ''}{dup}"


def print_result(st, res, args) -> None:
    print(f"\n『{res.query}』")
    if args.company:
        print(f"  硬过滤 company={args.company}")
    print(f"  识别公司：{[f'{c.stock_code} {c.security_name}' for c in res.companies] or '无'}")
    print(f"  路由 {list(res.routes)} | 各路召回 {res.recall_counts} | "
          f"合并同内容块 {res.merged_duplicates} | 最终 {len(res.hits)} 条")
    if res.structured:
        # 筛选题的最终条数会明显大于 top_k，不说清楚会以为是 --k 没生效
        cov = res.structured_coverage
        print(f"  🗂 {res.structured}")
        print(f"     覆盖 {cov.get('n_companies', '?')}/17 家公司；"
              f"其余 {len(res.hits) - cov.get('n_rows', 0)} 条是相似度补充")
    for w in res.warnings:
        print(f"  ! {w}")
    if not res.hits:
        print("  （无命中 —— 该问题在本语料里可能没有答案，或查询词全不在词表）")
        return
    for h in res.hits:
        print(fmt_hit(st, h, width=args.width))


def print_answer(r: QAResult) -> None:
    """打印生成结果与**引用出处**。引用编号 [n] 对应上面第 n 条命中。"""
    a = r.answer
    print(f"\n{'=' * 70}\n【最终回答】")
    if a is None:
        print("  （未调用生成）")
        return
    if a.error:
        # 生成失败也要说清楚，且上面已经打印过召回列表 —— 用户至少知道捞到了什么
        print(f"  生成失败：{a.error}")
        return
    print(a.text)
    print(f"\n  引用 {a.citations or '无'}｜未引用 {a.uncited or '无'}｜"
          f"prompt {a.prompt_chars:,} 字符｜耗时 {a.elapsed:.1f}s｜{a.usage or ''}")
    if a.invalid_citations:
        print(f"  ⚠ 回答里出现不存在的证据编号 {a.invalid_citations}"
              f"（证据只有 1~{len(r.evidences)}）—— 模型在编造出处，需逐条回原文核对")
    if a.insufficient:
        print("  （模型判定证据不足或明确拒答 —— 通常说明召回里没有能支撑答案的原文）")
    if not r.cited:
        return
    print("\n【引用出处】")
    for e in r.cited:
        print(f"  [{e.idx}] {e.source_label}｜{e.chunk_id}")


def result_to_dict(st, res, args) -> dict:
    """给评测与复现用的结构化结果。带上 row / chunk_id / 各路原始分。"""
    out = []
    for h in res.hits:
        r = st.record(h.row)
        out.append({**{k: r[k] for k in ("row", "chunk_id", "stock_code", "company",
                                         "section", "subsection", "page_start",
                                         "page_end", "content_type", "char_count")},
                    "rank": h.rank, "scores": h.scores,
                    "dup_chunk_ids": h.dup_chunk_ids})
    return {"query": res.query, "top_k": args.k, "routes": list(res.routes),
            "weights": {f: getattr(args.weights, f) for f in Weights.__dataclass_fields__},
            "company_filter": args.company,
            "companies": [c.stock_code for c in res.companies],
            "hits": out}


def build_weights(args) -> Weights:
    """`--no-weight` → 全 1.0；否则用 retriever 的默认权重。

    把实际生效的权重记进 --json，是为了让评测结果**自带口径**：事后看一份
    数字时，能知道它是加权还是消融跑出来的。
    """
    return NO_WEIGHTS if args.no_weight else Weights()


def run_queries(st, queries: list[str], args) -> int:
    dump = []
    for q in queries:
        if args.no_llm:
            res = retrieve(q, top_k=args.k, routes=args.routes, company=args.company,
                           weights=args.weights, st=st)
            print_result(st, res, args)
            dump.append(result_to_dict(st, res, args))
            continue
        # 完整链：检索 → 证据 → DeepSeek → 带引用的答案（编排在 qa.py）
        r = answer_question(q, top_k=args.k, routes=args.routes,
                            company=args.company, weights=args.weights,
                            with_llm=True, st=st)
        print_result(st, r.retrieval, args)
        print_answer(r)
        dump.append({**result_to_dict(st, r.retrieval, args),
                     "answer": None if r.answer is None else {
                         "text": r.answer.text, "citations": r.answer.citations,
                         "invalid_citations": r.answer.invalid_citations,
                         "uncited": r.answer.uncited,
                         "insufficient": r.answer.insufficient,
                         "prompt_chars": r.answer.prompt_chars,
                         "usage": r.answer.usage,
                         "elapsed": round(r.answer.elapsed, 2),
                         "error": r.answer.error}})
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dump) + "\n",
                     encoding="utf-8")
        print(f"\n已写出 {len(dump)} 条 → {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="上交所医疗器械财报 RAG —— 召回/问答 CLI")
    ap.add_argument("question", nargs="*", help="问题；省略则配合 -i 进入交互")
    ap.add_argument("-k", type=int, default=None, help="最终返回条数（默认取 config）")
    ap.add_argument("--routes", default=None,
                    help="逗号分隔：bm25 / vector / 留空=可用即用")
    ap.add_argument("--company", default=None,
                    help="显式硬过滤，逗号分隔的代码或简称（如 688271 或 联影医疗）")
    ap.add_argument("--no-weight", action="store_true", help="关掉全部软加权（消融）")
    ap.add_argument("--no-llm", action="store_true", help="只召回不生成（阶段 1 唯一可用模式）")
    ap.add_argument("--json", default=None, help="把结果写成 jsonl 路径")
    ap.add_argument("--width", type=int, default=180, help="每个块的正文摘要字数")
    ap.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    args = ap.parse_args()

    if args.routes:
        args.routes = tuple(r.strip() for r in args.routes.split(",") if r.strip())
    args.weights = build_weights(args)

    st = store()
    print(st.describe())
    for w in st.warnings:
        print(f"  ! {w}")

    if not args.no_llm:
        # 先明确报错再开跑：缺 key 时若逐题在 generate() 里失败会刷一屏，
        # 而且看起来像「模型答不上来」，实际是压根没配 key。
        try:
            config.assert_llm_ready()
        except RuntimeError as e:
            print(f"\n[无法生成] {e}")
            print("          只想看召回请加 --no-llm。")
            return 2

    if args.interactive:
        print("交互模式：输入问题回车，空行/Ctrl-C 退出。")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not q:
                return 0
            print_result(st, retrieve(q, top_k=args.k, routes=args.routes,
                                      company=args.company,
                                      weights=build_weights(args)), args)

    if not args.question:
        ap.print_help()
        print(f"\n可用公司（{len(load_companies())} 家）："
              f"{'、'.join(c.security_name for c in load_companies())}", file=sys.stderr)
        return 1
    return run_queries(st, args.question, args)


if __name__ == "__main__":
    raise SystemExit(main())
