"""问答编排：问题 → 两路召回 → 融合去重 → Top-6 证据 → DeepSeek → 带引用答案。

把「检索」和「生成」**显式拆开**，是因为两者的失败模式完全不同：
  * 检索失败 = 没捞到该捞的块 → 要看召回列表（可能根本没进池子）
  * 生成失败 = 捞到了但答错/编数 → 要看证据与引用是否对得上
返回的对象把两者都留着（`retrieval` 和 `answer`），界面才能分别展示。
生成失败时 `answer` 带 `error` 而不是抛异常 —— 界面还能显示检索结果，
用户至少知道「捞到了什么、为什么没答上来」。

证据数不是恒为 Top-6：条件筛选类问题走结构化表组路由时是
「17 家公司各 1 张 + Top-K 条相似度补充」= 23 条左右（理由见 structured.py）。
所以 prompt 大小由**证据条数 × 单条上限**（`LLM_EVIDENCE_MAX_CHARS`，默认 1800 字）
决定，而不是一个固定的常数 —— 结构性表组每张 300~600 字，不触发单条截断，
整包 prompt 约 2 万字符，在 DeepSeek 的上下文内。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402
from src.retrieval.generator import Answer, Evidence, build_evidence, generate  # noqa: E402
from src.retrieval.retriever import RetrievalResult, Weights, retrieve  # noqa: E402
from src.retrieval.store import Store, store  # noqa: E402


@dataclass
class QAResult:
    question: str
    retrieval: RetrievalResult
    evidences: list[Evidence] = field(default_factory=list)
    answer: Answer | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        ws = list(self.retrieval.warnings)
        if self.answer is not None and self.answer.error:
            ws.append(f"生成失败：{self.answer.error}")
        return ws

    @property
    def cited(self) -> list[Evidence]:
        """回答真正引用到的证据（按编号升序）。界面「引用出处」只渲染这些。"""
        if not self.answer:
            return []
        want = set(self.answer.citations)
        return [e for e in self.evidences if e.idx in want]


def answer_question(question: str, *,
                    k_vector: int | None = None,
                    k_bm25: int | None = None,
                    top_k: int | None = None,
                    routes: tuple[str, ...] | list[str] | None = None,
                    company: list[str] | str | None = None,
                    weights: Weights | None = None,
                    with_llm: bool = True,
                    max_chars_per: int | None = None,
                    st: Store | None = None) -> QAResult:
    """走完整条链。with_llm=False 时只做到证据为止（没有 key 也能用）。"""
    st = st or store()
    t0 = time.monotonic()

    res = retrieve(question, top_k=top_k, k_bm25=k_bm25, k_vector=k_vector,
                   routes=routes, company=company, weights=weights, st=st)
    t_retr = time.monotonic() - t0

    # 只对进 Top-K 的块取正文（record 会复制整段 content，全量取是浪费）
    records = {h.row: st.record(h.row) for h in res.hits}
    ev = build_evidence(res.hits, records, max_chars_per=max_chars_per)

    out = QAResult(question=question, retrieval=res, evidences=ev,
                   timings={"retrieval": t_retr})
    if not with_llm:
        return out

    t1 = time.monotonic()
    out.answer = generate(question, ev)
    out.timings["generate"] = time.monotonic() - t1
    out.timings["total"] = time.monotonic() - t0
    return out


def main() -> int:
    """命令行自检：python -m src.retrieval.qa "问题" [--no-llm]"""
    args = sys.argv[1:]
    with_llm = "--no-llm" not in args
    qs = [a for a in args if not a.startswith("-")] or ["联影医疗 2026 上半年营业收入是多少"]

    st = store()
    print(f"索引：{st.describe()}")
    for q in qs:
        r = answer_question(q, with_llm=with_llm, st=st)
        print(f"\n{'=' * 70}\n问题：{q}")
        print(f"召回：BM25 {r.retrieval.recall_counts.get('bm25', 0)} 条 + "
              f"向量 {r.retrieval.recall_counts.get('vector', 0)} 条"
              f" → 去重合并后 Top {len(r.evidences)}"
              f"（合并掉重复 {r.retrieval.merged_duplicates} 条）"
              f"｜耗时 {r.timings.get('retrieval', 0):.2f}s")
        for e in r.evidences:
            print(f"  [{e.idx}] rrf={e.rrf:.5f} {'/'.join(e.routes):12s} "
                  f"{e.chunk_id}  {e.company[:10]} | {e.section} | {e.page_label} "
                  f"| {e.content_type} {e.char_count}字"
                  f"{' [截断]' if e.truncated else ''}")
        if r.answer is None:
            continue
        print(f"\n答案（{r.answer.elapsed:.1f}s，prompt {r.answer.prompt_chars:,} 字符，"
              f"{r.answer.usage or ''}）：")
        print(r.answer.text or f"  <无内容: {r.answer.error}>")
        if r.answer.invalid_citations:
            print(f"  ⚠ 非法引用编号（证据里不存在）：{r.answer.invalid_citations}")
        if r.answer.uncited:
            print(f"  · 未被引用到的证据：[{']['.join(map(str, r.answer.uncited))}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
