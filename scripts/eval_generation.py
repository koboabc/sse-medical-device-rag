"""生成层评测：逐题落盘「问题 / 召回 / 证据 / 回答 / 标准答案 / 是否答对 / 错误原因」。

## 为什么「是否答对」由人判，而不是脚本算

财报问答的错法有两种，任何字符串比对都抓不到：

* **口径错**：「58,802.29 **万元**」答成「58,802.29 **元**」—— 字符几乎全对，
  量级差 10^4（评测集 Q04 专考这一点）。
* **期间错**：答的是「上年同期」的数而不是「本报告期」。

反过来，写死数字比对也会冤枉正确答案：同一指标在证据里同时有本期/上期两个数，
模型答本期就是对。所以本脚本只做两件事：

1. **把机器能判的判掉**：引用是否落在 gold 内（`citation_hit`）、是否拒答
   （`refused`）、有没有编造证据编号（`invalid_citations`）、证据覆盖了几条 gold。
   这些都是客观量，可复现。
2. **把「是否答对」留空**，由人读过回答后写进 `--review` 的 JSON，再 `--report` 出表。

**宁缺不编**：未被判定的题在表里显示「未判」，绝不当成答对，也绝不用规则凑一个
看起来像准确率的数字出来。

## 三级产物

    generation_formal10_<stamp>.json   原始记录（含完整证据正文与回答原文）
    review_formal10.json               人的判定（手写，每题一条）
    formal10_per_question.md           课程要求的逐题表（由上面两份合成）

用法：
    # 1) 不需要 key：先出「召回 / 证据」两栏，回答栏留空
    python scripts/eval_generation.py --no-llm

    # 2) 配好 key 后跑完整链（会调 DeepSeek，10 题约 2~4 分钟）
    python scripts/eval_generation.py

    # 3) 人读过回答后写判定，再出逐题表
    python scripts/eval_generation.py --report \
        --review evaluation/results/review_formal10.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config  # noqa: E402
from src.retrieval.qa import answer_question  # noqa: E402
from src.retrieval.store import store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
DEFAULT_DS = ROOT / "evaluation" / "datasets" / "qa_formal10.jsonl"


def load_dataset(p: Path) -> list[dict]:
    items = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p}:{ln} 不是合法 JSON：{e}")
    return items


def per_route_ids(st, query: str, k_bm25: int, k_vector: int) -> dict:
    """逐路召回结果（题干要求逐题保存 BM25 / 向量 / 混合三路）。

    这里**独立于** answer_question 内部那次检索再跑一遍：answer_question 只返回
    融合后的最终列表，而报告要的是「每一路各自召回了什么」—— 排查「是没召回到
    还是召回到了却被融掉」时，这一栏是唯一依据。纯检索、不调模型，代价可忽略。
    """
    out: dict[str, list[str]] = {}
    out["bm25"] = [h.chunk_id for h in st.search_bm25(query, k=k_bm25)]
    if st.ensure_vector():
        out["vector"] = [h.chunk_id for h in st.search_vector(query, k=k_vector)]
    else:
        out["vector"] = []
    return out


def run_one(item: dict, st, args) -> dict:
    q = item["question"]
    gold = set(item.get("gold_chunk_ids") or [])
    r = answer_question(q, with_llm=not args.no_llm, st=st)
    res, ans = r.retrieval, r.answer

    # 最终证据覆盖了哪些 gold（内容重复被合并掉的块也计入，口径同 eval_retrieval）
    covered: set[str] = set()
    for e, h in zip(r.evidences, res.hits):
        for cid in [e.chunk_id, *getattr(h, "dup_chunk_ids", [])]:
            if cid in gold:
                covered.add(cid)

    cited_ids = [e.chunk_id for e in r.cited] if ans else []
    rec = {
        "qid": item["qid"], "type": item["type"], "question": q,
        "gold_answer": item.get("gold_answer", ""),
        "gold_chunk_ids": sorted(gold),
        "retrieved_per_route": per_route_ids(st, q, args.k_bm25, args.k_vector),
        "final_hit_ids": [h.chunk_id for h in res.hits],
        "structured": res.structured,
        "n_evidence": len(r.evidences),
        "evidence": [{
            "idx": e.idx, "chunk_id": e.chunk_id, "stock_code": e.stock_code,
            "company": e.company, "section": e.section, "subsection": e.subsection,
            "page_label": e.page_label, "content_type": e.content_type,
            "char_count": e.char_count, "rrf": e.rrf, "rank": e.rank,
            "routes": e.routes, "truncated": e.truncated,
            "content": e.content,
        } for e in r.evidences],
        # ---- 机器判定（客观、可复现）----
        "gold_covered_in_evidence": sorted(covered),
        "gold_recall_in_evidence": (round(len(covered) / len(gold), 4)
                                    if gold else None),
        "citation_hit": (bool(gold & set(cited_ids)) if gold else None),
        "cited_ids": cited_ids,
        "refused": (bool(ans.insufficient) if ans else None),
        # ---- 人判（此处留空）----
        "correct": None,
        "error_reason": None,
        "timings": {k: round(v, 2) for k, v in r.timings.items()},
    }
    if ans is not None:
        rec["answer"] = {
            "text": ans.text, "citations": ans.citations,
            "invalid_citations": ans.invalid_citations, "uncited": ans.uncited,
            "insufficient": ans.insufficient, "prompt_chars": ans.prompt_chars,
            "usage": ans.usage, "elapsed": round(ans.elapsed, 2),
            "error": ans.error,
        }
    else:
        rec["answer"] = None
    return rec


def fmt_ids(ids: list[str], limit: int = 5) -> str:
    if not ids:
        return "—"
    s = "、".join(ids[:limit])
    return s + (f" …(共 {len(ids)} 条)" if len(ids) > limit else "")


def write_report(recs: list[dict], review: dict, out: Path) -> None:
    """课程要求的逐题表：问题 / 召回 / 最终证据 / 回答 / 标准答案 / 是否答对 / 错误原因。"""
    lines: list[str] = ["# 正式评测集（10 题）逐题记录", ""]
    lines += [
        f"- 数据集：`evaluation/datasets/qa_formal10.jsonl`（10 题；"
        f"跨公司题 3 道：Q11 / Q13 / Q15；拒答陷阱 2 道：T01 / T03）",
        f"- 索引：`{recs[0].get('_index', '?')}`｜召回深度 BM25 {config.top_k_bm25} / "
        f"向量 {config.top_k_vector}｜最终证据 Top-{config.top_k_final}（条件筛选题走结构化表组时为 17+6）",
        f"- 生成：`{config.llm_model}` @ `{config.llm_base_url}`，temperature "
        f"{config.llm_temperature}",
        "",
        "> **「是否答对」是人读过回答后判的**（`review_formal10.json`），不是脚本算的。",
        "> 财报问答里「58,802.29 元」与「58,802.29 万元」字符全对而量级差一万倍，",
        "> 自动比对判不出来；未判定的题一律显示「未判」，不计入任何比例。",
        "",
        "## 汇总", "",
        "| 题号 | 类型 | 证据含 gold | 引用命中 gold | 是否答对 | 错误原因 |",
        "|---|---|---|---|---|---|",
    ]
    n_judged = n_correct = 0
    for r in recs:
        rv = review.get(r["qid"], {})
        c = rv.get("correct")
        verdict = {True: "✅ 答对", False: "❌ 答错", None: "— 未判"}[c]
        if c is not None:
            n_judged += 1
            n_correct += int(bool(c))
        ev = (f"{len(r['gold_covered_in_evidence'])}/{len(r['gold_chunk_ids'])}"
              if r["gold_chunk_ids"] else "—(陷阱)")
        ch = {True: "✅", False: "❌", None: "—"}[r["citation_hit"]]
        lines.append(f"| {r['qid']} | {r['type']} | {ev} | {ch} | {verdict} | "
                     f"{rv.get('reason', '')} |")
    lines += ["", f"**已判定 {n_judged}/{len(recs)} 题，其中答对 {n_correct} 题。**"
              + ("" if n_judged == len(recs) else
                 "（其余未判定 —— 不推算、不补齐。）"), ""]

    for r in recs:
        rv = review.get(r["qid"], {})
        c = rv.get("correct")
        lines += ["---", "", f"## {r['qid']}（{r['type']}）", "",
                  f"**问题**：{r['question']}", ""]
        if r["structured"]:
            lines += [f"*本次走结构化表组路由*：{r['structured']}", ""]
        lines += [f"**标准答案**：{r['gold_answer']}", "",
                  f"**gold chunk**：{fmt_ids(r['gold_chunk_ids'], 12)}", "",
                  "**召回（各路 Top-5）**", "",
                  f"- BM25：{fmt_ids(r['retrieved_per_route'].get('bm25', []))}",
                  f"- 向量：{fmt_ids(r['retrieved_per_route'].get('vector', []))}",
                  f"- 融合后最终 {len(r['final_hit_ids'])} 条："
                  f"{fmt_ids(r['final_hit_ids'], 8)}", "",
                  f"**最终证据（{r['n_evidence']} 条）**", "",
                  "| # | chunk_id | 公司 | 章节 | 页 | 类型 | 字数 | 召回 |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in r["evidence"][:24]:
            sub = f"{e['section']} > {e['subsection']}" if e["subsection"] else e["section"]
            lines.append(f"| [{e['idx']}] | {e['chunk_id']} | {e['company'][:10]} | "
                         f"{sub[:22]} | {e['page_label']} | {e['content_type']} | "
                         f"{e['char_count']} | {'/'.join(e['routes'])} |")
        if len(r["evidence"]) > 24:
            lines.append(f"| … | 其余 {len(r['evidence']) - 24} 条见 JSON | | | | | | |")
        a = r.get("answer")
        lines += ["", "**回答**", ""]
        if a is None:
            lines += ["> 本次未调用生成（`--no-llm` 或未配置 key）。", ""]
        elif a["error"]:
            lines += [f"> 生成失败：{a['error']}", ""]
        else:
            lines += [a["text"], "",
                      f"*引用 {a['citations']}；未引用 {a['uncited']}；"
                      f"prompt {a['prompt_chars']:,} 字符；耗时 {a['elapsed']}s；"
                      f"token {a['usage'].get('total_tokens', '?')}*", ""]
            if a["invalid_citations"]:
                lines += [f"> ⚠️ 出现不存在的证据编号 {a['invalid_citations']} "
                          f"（证据只有 1~{r['n_evidence']}）—— 模型在编造出处。", ""]
        lines += ["**是否答对**："
                  + {True: "✅ 答对", False: "❌ 答错", None: "— 未判"}[c], "",
                  f"**错误原因**：{rv.get('reason') or '—'}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"逐题表已写出 → {out}（已判定 {n_judged}/{len(recs)}，答对 {n_correct}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成层评测：逐题记录召回/证据/回答/对错")
    ap.add_argument("--dataset", default=str(DEFAULT_DS))
    ap.add_argument("--index", default=None, help="索引目录（默认 config.index_dir）")
    ap.add_argument("--no-llm", action="store_true",
                    help="只跑召回与证据，不调 DeepSeek（无 key 时也能出前两栏）")
    ap.add_argument("--k-bm25", type=int, default=None)
    ap.add_argument("--k-vector", type=int, default=None)
    ap.add_argument("--out", default=None, help="原始记录落盘路径（默认带时间戳）")
    ap.add_argument("--review", default=None, help="人的判定 JSON（qid → {correct, reason}）")
    ap.add_argument("--report", action="store_true", help="由记录 + 判定生成逐题表")
    ap.add_argument("--from-json", default=None, help="--report 时读取的记录 JSON")
    args = ap.parse_args()

    args.k_bm25 = args.k_bm25 or config.top_k_bm25
    args.k_vector = args.k_vector or config.top_k_vector

    if args.report:
        src = Path(args.from_json) if args.from_json else max(
            RESULTS.glob("generation_formal10_*.json"), key=lambda p: p.stat().st_mtime)
        recs = json.loads(src.read_text(encoding="utf-8"))["records"]
        review = (json.loads(Path(args.review).read_text(encoding="utf-8"))
                  if args.review else {})
        write_report(recs, review, RESULTS / "formal10_per_question.md")
        print(f"记录来源 → {src}")
        return 0

    items = load_dataset(Path(args.dataset))
    if not args.no_llm:
        config.assert_llm_ready()       # 缺 key 时在这里明确报错，而不是逐题 401
    st = store(args.index)
    print(f"索引：{st.describe()}")
    print(f"数据集：{Path(args.dataset).name}（{len(items)} 题）｜"
          f"生成：{'关闭（--no-llm）' if args.no_llm else config.llm_model}")

    recs = []
    for it in items:
        rec = run_one(it, st, args)
        rec["_index"] = str(st.index_dir)
        recs.append(rec)
        a = rec.get("answer")
        flag = ("—" if a is None else
                "拒答" if a["insufficient"] else
                f"{len(a['citations'])} 引用")
        print(f"  {rec['qid']} {rec['type'][:4]:4s} 证据 {rec['n_evidence']:2d} 条 "
              f"gold 覆盖 {len(rec['gold_covered_in_evidence'])}/"
              f"{len(rec['gold_chunk_ids'])} 回答 {flag} "
              f"({rec['timings'].get('total', rec['timings'].get('retrieval', 0))}s)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else (
        RESULTS / f"generation_formal10_{datetime.now():%Y%m%d-%H%M%S}.json")
    out.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dataset": Path(args.dataset).name, "index": str(st.index_dir),
        "llm": None if args.no_llm else config.llm_model,
        "k_bm25": args.k_bm25, "k_vector": args.k_vector,
        "records": recs}, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ans = sum(1 for r in recs if r["answer"] and not r["answer"]["error"])
    print(f"\n已写出 → {out}")
    print(f"  拿到回答 {n_ans}/{len(recs)} 题｜"
          f"引用命中 gold {sum(1 for r in recs if r['citation_hit'])} 题｜"
          f"拒答 {sum(1 for r in recs if r['refused'])} 题")
    print("  ⚠️ 本轮**不含**「是否答对」：那要人读过回答后写 --review，再 --report 出表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
