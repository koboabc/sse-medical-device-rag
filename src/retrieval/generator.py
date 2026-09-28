"""生成层：把召回证据交给 DeepSeek，产出**有出处**的答案。

这个模块只做一件事 —— 把「证据」和「问题」拼成 prompt，调 API，把回答和
引用编号解析出来。检索不在这里（见 retriever.py），编排在 qa.py。

三条硬约束，都是为了让答案可核查：

1. **只能依据证据**。系统提示明确禁止用模型自己的知识补数字。
   财报问答里「差一个数量级」和「编一个数」是同一件事：都不可核查。
2. **引用编号必须真实存在**。回答里的 [n] 会被解析出来，越界的编号会被
   标成非法引用（而不是静默丢掉）。UI 据此把 [n] 与证据卡片对上。
3. **证据不足必须说不足**。财报里大量指标是「本期/上期」「合并/母公司」
   多个口径；证据不全时模型必须明说不全，而不是挑一个像的填上。

API Key 只从 `.env` 读（`config.llm_api_key`），本模块任何位置都不打印它，
也不接受把它作为参数传进来。异常信息里同样不含 key。
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

# 证据在 prompt 里的展示名。注意与答案里 [n] 的编号**必须一一对应**：
# 编号就是 evidences 列表的下标 + 1，解析引用时直接按这个约定反查。
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = """你是上交所医疗器械行业财报问答助手，服务于投研人员。

你**只能**依据用户消息里给出的【证据】回答，证据之外的知识一律不得使用。

硬性规则：
1. 每一个事实性陈述后面必须标出证据编号，形如 [1] 或 [2][3]。编号必须真实存在于证据列表中。
2. 证据里没有出现的数字、公司名、结论，一律不得写进答案。禁止用常识、记忆或推算补全。
3. 如果证据不足以回答问题，必须直接说「提供的证据不足以回答该问题」，并指明缺什么
   （例如「证据中没有 X 公司的 Y 指标」）。**不要猜测，不要估算，不要用相近指标替代。**
4. 引用数字时必须带上原文的口径与单位（元/万元、本期/上年同期、合并/母公司），
   并说明来自哪张表或哪一节。
5. 同一指标在证据中出现多个口径时，要把口径说清楚，不要只给一个数。
6. 用简体中文回答。直接回答，不要复述问题，不要罗列与问题无关的证据。
7. 不要输出证据列表本身，也不要自造表格；出处由系统单独展示。"""


@dataclass
class Evidence:
    """一条证据：检索命中 + 完整元数据。UI 的「引用出处」直接渲染它。"""

    idx: int                    # 引用编号，从 1 开始
    row: int
    chunk_id: str
    stock_code: str
    company: str
    report: str
    section: str
    subsection: str
    page_start: int
    page_end: int
    content_type: str
    char_count: int
    content: str                # 原始正文（完整，未截断）
    prompt_text: str            # 实际进 prompt 的正文（超长表格可能被截断）
    truncated: bool
    routes: list[str] = field(default_factory=list)    # 哪几路召回了它
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    rrf: float = 0.0

    @property
    def page_label(self) -> str:
        return (f"p.{self.page_start}" if self.page_start == self.page_end
                else f"p.{self.page_start}-{self.page_end}")

    @property
    def source_label(self) -> str:
        loc = f"{self.section} > {self.subsection}" if self.subsection else self.section
        return f"{self.company}（{self.stock_code}）｜{self.report}｜{loc}｜{self.page_label}"

    def header(self) -> str:
        return (f"[{self.idx}] 公司：{self.company}（{self.stock_code}）"
                f"｜报告：{self.report}\n"
                f"    章节：{self.section}"
                + (f" > {self.subsection}" if self.subsection else "")
                + f"\n    页码:{self.page_label}"
                f"｜chunk_id：{self.chunk_id}"
                f"｜类型：{self.content_type}"
                f"｜召回路径：{'/'.join(self.routes) or '-'}"
                + (f"\n    注意：本段因长度被截断，仅前 {len(self.prompt_text)} 字进入证据。"
                   if self.truncated else ""))


@dataclass
class Answer:
    text: str
    citations: list[int]                 # 回答里实际引用的编号（去重、升序）
    invalid_citations: list[int]         # 越界/不存在的编号 —— 必须暴露，不能吞
    uncited: list[int]                   # 给了但没被任何引用用上的证据编号
    insufficient: bool                   # 是否明确回答「证据不足」
    prompt_chars: int
    usage: dict = field(default_factory=dict)
    elapsed: float = 0.0
    error: str | None = None             # 有值表示没拿到答案（UI 要显示原因）


def clip_for_prompt(text: str, content_type: str, max_chars: int) -> tuple[str, bool]:
    """超长正文按行截断，**保留表头**。

    表格块最长 13758 字，全塞进 prompt 会挤掉别的证据。表格的第一行是表头，
    决定了后面每一列是什么 —— 只留前 N 字而不留表头，剩下的行就是无意义的
    数字串。所以按行拼，且第一行必留。
    """
    if len(text) <= max_chars:
        return text, False
    lines = text.splitlines() or [text]
    head, rest = lines[0], lines[1:]
    if len(head) >= max_chars:                 # 表头本身就超长，退化为硬截断
        return text[:max_chars], True
    out = [head]
    used = len(head)
    for ln in rest:
        if used + len(ln) + 1 > max_chars:
            break
        out.append(ln)
        used += len(ln) + 1
    kept = len(out) - 1
    tail = (f"\n…（本{'表' if content_type == 'table' else '段'}过长，"
            f"已截断：上面是表头的 {kept} 行，全文 {len(text)} 字，"
            f"完整内容见出处 {max_chars} 字以外的部分）")
    return "\n".join(out) + tail, True


def build_evidence(hits, records: dict[int, dict], *,
                   max_chars_per: int | None = None) -> list[Evidence]:
    """把命中列表变成编号证据。records 是 row → store.record(row) 的映射。

    编号 = 列表下标 + 1，与 prompt 里的 [n] 严格对应；调用方不得再排序，
    否则编号与 UI 上的卡片会错位。
    """
    cap = int(max_chars_per or config.llm_evidence_max_chars)
    out: list[Evidence] = []
    for i, h in enumerate(hits, 1):
        r = records[h.row]
        text, tr = clip_for_prompt(r["content"], r["content_type"], cap)
        # h.source 对「两路都召回」的块是 "bm25+vector"，所以要先按 + / 拆开，
        # 否则会拼出 "bm25/bm25+vector/vector" 这种重复的展示。
        routes = sorted(set(h.source.replace("+", "/").split("/"))
                        | {k for k in (h.scores or {}) if k in ("bm25", "vector")})
        out.append(Evidence(
            idx=i, row=h.row, chunk_id=r["chunk_id"], stock_code=r["stock_code"],
            company=r["company"], report=r["report"], section=r["section"],
            subsection=r["subsection"] or "", page_start=r["page_start"],
            page_end=r["page_end"], content_type=r["content_type"],
            char_count=r["char_count"], content=r["content"],
            prompt_text=text, truncated=tr,
            routes=routes, scores=dict(h.scores or {}), rank=h.rank,
            rrf=float(h.score),
        ))
    return out


def build_messages(question: str, evidences: list[Evidence]) -> list[dict]:
    blocks = "\n\n".join(f"{e.header()}\n    正文：\n{e.prompt_text}" for e in evidences)
    user = (f"【问题】\n{question}\n\n"
            f"【证据】（共 {len(evidences)} 条，引用时只能用这些编号）\n\n{blocks}\n\n"
            f"【要求】依据以上证据回答；每条事实后标 [编号]；"
            f"证据不足就直接说明不足，不要猜。")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def parse_citations(text: str, n_evidences: int) -> tuple[list[int], list[int]]:
    """从回答里解析 [n]，返回 (有效引用, 非法引用)。

    非法引用（编号超出证据范围）必须单独返回并暴露到界面上 —— 那意味着
    模型在编造出处，比答案错更值得警惕，绝不能静默过滤掉。
    """
    seen: list[int] = []
    for m in CITATION_RE.finditer(text or ""):
        v = int(m.group(1))
        if v not in seen:
            seen.append(v)
    ok = sorted(v for v in seen if 1 <= v <= n_evidences)
    bad = sorted(v for v in seen if not 1 <= v <= n_evidences)
    return ok, bad


def client():
    """构造 OpenAI 兼容客户端。key 只从 config 来，绝不入参、绝不回显。"""
    config.assert_llm_ready()          # key 缺失时在这里抛，消息里不含 key 本身
    from openai import OpenAI

    return OpenAI(api_key=config.llm_api_key,
                  base_url=config.llm_base_url,
                  timeout=config.llm_timeout)


def generate(question: str, evidences: list[Evidence]) -> Answer:
    """调 DeepSeek 生成答案。失败不抛，返回带 error 的 Answer，让 UI 能显示原因。"""
    if not evidences:
        return Answer(text="", citations=[], invalid_citations=[], uncited=[],
                      insufficient=True, prompt_chars=0,
                      error="没有召回到任何证据，无法回答。")
    messages = build_messages(question, evidences)
    prompt_chars = sum(len(m["content"]) for m in messages)
    t0 = time.monotonic()
    try:
        resp = client().chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
    except Exception as e:  # noqa: BLE001
        # 只回类型与消息文本。openai 的异常里不会带 key，但万一被包装进
        # 某些自定义异常，这里也只取 str(e) 的前 400 字，避免整段 body 外泄。
        return Answer(text="", citations=[], invalid_citations=[], uncited=[],
                      insufficient=False, prompt_chars=prompt_chars,
                      elapsed=time.monotonic() - t0,
                      error=f"{type(e).__name__}: {str(e)[:400]}")

    text = (resp.choices[0].message.content or "").strip()
    ok, bad = parse_citations(text, len(evidences))
    usage = {}
    if getattr(resp, "usage", None):
        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens,
                 "total_tokens": resp.usage.total_tokens}
    return Answer(
        text=text, citations=ok, invalid_citations=bad,
        uncited=[e.idx for e in evidences if e.idx not in ok],
        insufficient=("证据不足" in text or "无法回答" in text),
        prompt_chars=prompt_chars, usage=usage,
        elapsed=time.monotonic() - t0,
    )


if __name__ == "__main__":
    # 自检：不联网，只验 prompt 拼装与引用解析
    from src.retrieval.retriever import retrieve
    from src.retrieval.store import store

    st = store()
    q = sys.argv[1] if len(sys.argv) > 1 else "联影医疗 2026 上半年营业收入"
    res = retrieve(q, st=st)
    recs = {h.row: st.record(h.row) for h in res.hits}
    ev = build_evidence(res.hits, recs)
    print(f"问题：{q}\n召回 {len(ev)} 条：")
    for e in ev:
        print(f"  [{e.idx}] {e.chunk_id} {e.company[:10]} | {e.section} | {e.page_label} "
              f"| {e.content_type} | {e.char_count}字"
              f"{'（截断到 %d 字进 prompt）' % len(e.prompt_text) if e.truncated else ''}")
    msgs = build_messages(q, ev)
    print(f"\nprompt 字符数 {sum(len(m['content']) for m in msgs):,} "
          f"（证据正文 {sum(len(e.prompt_text) for e in ev):,}）")
    print(f"引用解析自检：{parse_citations('营收为 1 亿 [1]，同比 [2] 与 [9]', 6)}")
