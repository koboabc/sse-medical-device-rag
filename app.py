"""上交所医疗器械财报 RAG —— Streamlit 问答页。

流程（页面上按这个顺序展示，每一步都能看到中间结果）：

    用户问题
      → 向量检索 Top-20 ┐
      → BM25 检索 Top-20 ┴→ 合并去重 → Top-6 证据 → DeepSeek → 带引用的回答

设计取舍，都是为了「答案可核查」：
  * **不使用 Reranker、不使用 Query Rewrite**（需求明确排除）。融合只用 RRF，
    它不是模型、可复现、不需要额外显存；加权是规则式的章节/公司偏好，全部
    参数都会显示在侧栏里，能被解释。
  * **检索与生成分开渲染**。生成失败（没 key、超时、限流）时页面照样显示
    召回列表 —— 用户至少能看到「捞到了什么」，而不是一个白屏。
  * **引用编号可追溯**。回答里的 [n] 直接对应「引用出处」里第 n 张卡片；
    越界的编号会被单独标红（那意味着模型在编造出处，比答错更值得警惕）。
  * **启动不重算任何东西**：不解析 PDF、不计算 embedding。BM25 从磁盘读，
    FAISS 惰性装载，0.6B 模型只在第一次走向量路时载入（约 30 秒，之后常驻）。

启动：
    streamlit run app.py            # 监听地址见 .streamlit/config.toml（0.0.0.0）
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import config                      # noqa: E402
from src.retrieval.entities import find_companies  # noqa: E402
from src.retrieval.qa import QAResult, answer_question  # noqa: E402
from src.retrieval.retriever import resolve_company_arg  # noqa: E402
from src.retrieval.store import store              # noqa: E402

EXAMPLES = [
    "联影医疗 2026 年上半年营业收入是多少",
    "心脉医疗的研发投入占营业收入的比例是多少",
    "新华医疗的净利润同比变化及原因",
    "哪些公司 2026 上半年归母净利润同比下滑",
    "联影医疗和心脉医疗哪家毛利率更高",
]

st.set_page_config(page_title="医疗器械财报问答", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# 资源：进程内单例。Streamlit 每次交互都重跑整个脚本，所以**必须**缓存，
# 否则每点一次按钮都会重新读一遍 BM25、重新载一次 0.6B 模型。
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_store():
    """只建一次。不解析 PDF、不算 embedding、不主动载 FAISS/模型。"""
    return store()


def llm_ready() -> bool:
    """只回布尔值。key 本身不打印、不显示、不进页面任何位置。"""
    k = config.llm_api_key or ""
    return bool(k) and not k.startswith("sk-xxx")


def esc(s) -> str:
    return html.escape(str(s), quote=False)


# 召回路径的中文名。页面上出现英文标识符（structured / bm25）对非技术读者是噪声。
ROUTE_CN = {"structured": "结构化表组", "bm25": "BM25", "vector": "向量"}


def routes_cn(routes) -> str:
    return "/".join(ROUTE_CN.get(r, r) for r in routes) or "-"


# --------------------------------------------------------------------------
# 侧栏：参数 + 索引状态
# --------------------------------------------------------------------------
# 注意变量名：**不能**把 store 绑成 `st` —— `import streamlit as st` 就在上面，
# 一覆盖，下一行的 st.sidebar 就会去找 Store.sidebar，整个脚本直接崩。
ST = get_store()

with st.sidebar:
    st.header("检索参数")
    k_vec = st.number_input("向量检索 Top-K", 1, 100, config.top_k_vector, 1)
    k_bm25 = st.number_input("BM25 检索 Top-K", 1, 100, config.top_k_bm25, 1)
    top_k = st.number_input("最终证据数（喂给 LLM）", 1, 20, config.top_k_final, 1)

    st.divider()
    use_vec = st.checkbox("启用向量检索", True,
                          disabled=not (config.index_dir / "faiss" / "faiss.index").exists(),
                          help="索引不存在时会自动降级为只用 BM25")
    use_bm25 = st.checkbox("启用 BM25 检索", True)
    st.caption("不使用 Reranker、不使用 Query Rewrite —— 融合只用 RRF，"
               "加权是规则式的章节/公司偏好。")

    st.divider()
    company_spec = st.text_input("限定公司（可选）",
                                 placeholder="如 688271 或 联影医疗（逗号分隔）",
                                 help="硬过滤，在打分前生效：别的公司根本不会进候选池")
    use_weight = st.checkbox("启用软加权（章节/公司偏好）", True)

    st.divider()
    with_llm = st.toggle("调用 DeepSeek 生成答案", value=llm_ready(),
                         disabled=not llm_ready())
    if not llm_ready():
        st.warning(
            "未检测到 `LLM_API_KEY`，当前只能看检索结果。\n\n"
            "在项目根目录执行（`!` 前缀让命令在本会话里跑，key 不会进入对话记录）：\n"
            "```\n! cp .env.example .env && notepad .env\n```\n"
            "填好 `LLM_API_KEY` 后重启页面即可。key 只从 `.env` 读，代码里不存。",
            icon="🔑")

    st.divider()
    st.header("索引状态")
    st.caption(ST.describe())
    st.caption(f"模型 `{config.llm_model}` @ `{config.llm_base_url}`")

st.title("📊 医疗器械财报问答")
st.caption("数据：17 家上交所医疗器械公司 2026 年半年度报告，4,657 个片段。"
           "答案只依据召回的原文，每条事实都带出处。")

# --------------------------------------------------------------------------
# 【问题输入】
# --------------------------------------------------------------------------
st.subheader("① 问题输入")
with st.form("ask", clear_on_submit=False):
    q = st.text_input("你的问题", placeholder="例：" + EXAMPLES[0],
                      label_visibility="collapsed")
    c1, c2 = st.columns([1, 6])
    go = c1.form_submit_button("提问", type="primary", use_container_width=True)
    c2.caption("也可以点下面的示例：" + " ｜ ".join(f"`{e}`" for e in EXAMPLES[:3]))

picked = None
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"示例 {i + 1}", key=f"ex{i}", use_container_width=True,
                  help=ex):
        picked = ex

question = picked or (q if go and q.strip() else None)


# --------------------------------------------------------------------------
# 执行
# --------------------------------------------------------------------------
def run(question: str) -> QAResult:
    routes = []
    if use_bm25:
        routes.append("bm25")
    if use_vec:
        routes.append("vector")
    if not routes:
        raise ValueError("至少启用一路检索")
    company = resolve_company_arg(company_spec) if company_spec.strip() else None
    return answer_question(question, k_vector=int(k_vec), k_bm25=int(k_bm25),
                           top_k=int(top_k), routes=tuple(routes),
                           company=company, with_llm=bool(with_llm), st=ST)


if question:
    if "result" not in st.session_state or st.session_state.get("q") != question:
        with st.spinner("检索中……首次启用向量路需要载入 0.6B 模型（约 30 秒）"):
            try:
                st.session_state["result"] = run(question)
                st.session_state["q"] = question
            except Exception as e:  # noqa: BLE001
                st.session_state["result"] = None
                st.session_state["err"] = f"{type(e).__name__}: {e}"
    r: QAResult | None = st.session_state.get("result")

    if r is None:
        st.error(f"检索失败：{st.session_state.get('err')}")
        st.stop()

    res, ans = r.retrieval, r.answer

    # ---- 流程概览：让「问题 → 两路检索 → 证据 → 答案」一眼可见 ----
    st.subheader("② RAG 流程")
    steps = [
        ("问题", "1 条"),
        ("BM25 召回", f"{res.recall_counts.get('bm25', 0)} 条"),
        ("向量召回", f"{res.recall_counts.get('vector', 0)} 条"),
        ("合并去重", f"{len(r.evidences)} 条"),
        ("DeepSeek", f"{ans.elapsed:.1f}s" if ans and not ans.error else "未调用"),
        ("回答", f"{len(ans.citations)} 处引用" if ans and not ans.error else "-"),
    ]
    for col, (name, val) in zip(st.columns(len(steps)), steps):
        col.metric(name, val)
    if res.structured:
        # 筛选题的证据集不是「最相似的 K 条」，而是「全部公司的同一张表」。
        # 不说清楚，用户看到 17 条证据会以为检索失控 —— 而它恰恰是这一路的设计。
        cov = res.structured_coverage
        st.info(
            f"**本次走「跨公司条件筛选增强」（结构化表组）**：{res.structured}。\n\n"
            f"「哪些公司……的有哪几家」这类问题要的是**全体公司的同一张表**，"
            f"不是「最相似的 6 条」—— 相似度检索在这里是结构性错配（实测三路 "
            f"hit@10 全为 0）。所以由元数据直接定位表组（不参与打分），"
            f"再补 {len(r.evidences) - cov.get('n_rows', 0)} 条相似度召回。"
            f"表组覆盖 **{cov.get('n_companies', '?')}/17 家公司**"
            f"（{cov.get('n_rows', '?')} 条）—— 漏掉一家就会漏掉一个正确答案。\n\n"
            f"⚠️ 这个 17/17 是**表组覆盖的公司数**（覆盖率），"
            f"**不是本系统的通用准确率** —— 这条增强只对「要一个集合且没点名公司」的"
            f"问题生效，其余题型（单点数值、同比原因、公司对比）走的仍是普通相似度检索。",
            icon="🗂️")
    st.caption(
        (f"召回 {len(res.hits)} 条候选（BM25 {res.recall_counts.get('bm25', 0)} + "
         f"向量 {res.recall_counts.get('vector', 0)}），内容重复合并掉 "
         f"{res.merged_duplicates} 条，"
         + ("结构化表组之外，取 RRF 融合前 "
            f"{len(r.evidences) - res.structured_coverage.get('n_rows', 0)} 条作为补充证据。"
            if res.structured else
            f"取 RRF 融合后前 {len(r.evidences)} 条作为证据。"))
        + f"检索耗时 {r.timings.get('retrieval', 0):.2f}s"
        + (f"，prompt {ans.prompt_chars:,} 字符" if ans else ""))

    for w in r.warnings:
        st.warning(w, icon="⚠️")

    # ---- 【最终回答】 ----
    st.subheader("③ 最终回答")
    if ans is None:
        st.info("未调用生成（侧栏的「调用 DeepSeek 生成答案」关闭，或未配置 key）。"
                "下面是检索到的证据。")
    elif ans.error:
        st.error(f"生成失败：{ans.error}")
    else:
        if ans.insufficient:
            st.warning("模型判定证据不足，或明确拒答 —— 这通常是正确行为，"
                       "说明召回里没有能支撑答案的原文。请核对下面的证据。", icon="🚫")
        st.markdown(ans.text)
        bits = []
        if ans.usage:
            bits.append(f"token {ans.usage.get('prompt_tokens', '?')} + "
                        f"{ans.usage.get('completion_tokens', '?')} = "
                        f"{ans.usage.get('total_tokens', '?')}")
        bits.append(f"耗时 {ans.elapsed:.1f}s")
        bits.append(f"引用 {ans.citations or '无'}")
        st.caption(" ｜ ".join(bits))
        if ans.invalid_citations:
            st.error(
                f"⚠️ 回答里出现了**不存在的证据编号** {ans.invalid_citations}："
                f"证据只有 1~{len(r.evidences)}。这说明模型在编造出处，"
                f"该答案的数字必须逐条回到原文核对。", icon="🚨")

    # ---- 【检索结果】 ----
    st.subheader("④ 检索结果（Top %d 证据）" % len(r.evidences))
    if res.hard_filter:
        st.caption(f"已硬过滤公司：{res.hard_filter}")
    if res.companies:
        st.caption("问题里识别到的公司：" +
                   "，".join(f"{c.security_name}（{c.stock_code}）"
                            for c in res.companies))

    for e in r.evidences:
        # bool() 不能省：ans 为 None（没配 key / 没调生成）时，`ans and ...`
        # 求值成 None，而 expander(expanded=None) 内部会 int(None) 直接崩。
        cited = bool(ans and e.idx in ans.citations)
        head = (f"[{e.idx}] {e.company}（{e.stock_code}）｜{e.section}"
                + (f" > {e.subsection}" if e.subsection else "")
                + f"｜{e.page_label}｜{e.chunk_id}")
        with st.expander(head + ("　✅ 被回答引用" if cited else ""), expanded=cited):
            m = st.columns(6)
            m[0].metric("RRF", f"{e.rrf:.5f}")
            m[1].metric("召回路径", routes_cn(e.routes))
            m[2].metric("该路排名", e.rank)
            m[3].metric("类型", e.content_type)
            m[4].metric("字数", e.char_count)
            m[5].metric("进 prompt", f"{len(e.prompt_text)}字"
                        + ("（截断）" if e.truncated else ""))
            if e.scores:
                st.caption("各路原始分（尺度不同，不能直接比）：" +
                           " ｜ ".join(f"{k}={v:.4f}" for k, v in sorted(e.scores.items())))
            if e.truncated:
                st.info(f"正文 {e.char_count} 字超过单条上限 {config.llm_evidence_max_chars} 字，"
                        f"进 LLM 的版本按行截断（保留表头），下面显示的是**完整原文**。",
                        icon="✂️")
            st.text_area("原文", e.content, height=260, key=f"c{e.idx}",
                         label_visibility="collapsed")

    # ---- 【引用出处】 ----
    st.subheader("⑤ 引用出处")
    if ans is None or ans.error:
        st.caption("本次没有生成答案，无引用。")
    elif not r.cited:
        st.caption("回答没有引用任何证据 —— 要么明确拒答，要么有问题，"
                   "请对照上面的证据自查。")
    else:
        for e in r.cited:
            st.markdown(
                f"**[{e.idx}]** {esc(e.company)}（`{esc(e.stock_code)}`）　"
                f"报告：{esc(e.report)}　章节：{esc(e.section)}"
                + (f" > {esc(e.subsection)}" if e.subsection else "")
                + f"　页码：{esc(e.page_label)}　chunk_id：`{esc(e.chunk_id)}`"
                + f"　召回：{esc(routes_cn(e.routes))}")
            st.text_area("原文", e.content, height=200, key=f"cite{e.idx}",
                         label_visibility="collapsed")
        if ans.uncited:
            st.caption(f"另有未被引用的证据：{[f'[{i}]' for i in ans.uncited]}"
                       f" —— 它们进了上下文但没被答案使用。")

        with st.expander("导出（Markdown，含完整出处与原文）"):
            lines = [f"# {question}", "", ans.text, "", "## 出处"]
            for e in r.cited:
                lines += [f"### [{e.idx}] {e.company}（{e.stock_code}）",
                          f"- 报告：{e.report}",
                          f"- 章节：{e.section}"
                          + (f" > {e.subsection}" if e.subsection else ""),
                          f"- 页码：{e.page_label}",
                          f"- chunk_id：{e.chunk_id}",
                          f"- 召回路径：{'/'.join(e.routes)}", "", "```", e.content, "```", ""]
            st.code("\n".join(lines), language="markdown")

else:
    st.info("在上面输入一个问题，或点一个示例按钮。")

with st.sidebar:
    st.divider()
    st.caption("流程：问题 → 向量 Top-20 + BM25 Top-20 → RRF 融合去重 → "
               "Top-6 证据 → DeepSeek → 带引用回答")
    if st.button("清空结果缓存"):
        st.session_state.pop("result", None)
        st.session_state.pop("q", None)
        st.rerun()
