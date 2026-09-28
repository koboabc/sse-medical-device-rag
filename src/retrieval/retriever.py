"""两路召回 → RRF 融合 → 软加权 → Top-K。

三段各自的理由：

1. **为什么要 RRF（Reciprocal Rank Fusion）而不是加权求和总分**
   BM25 的量纲是「词频×IDF 累加」，可以到 12、也可以到 40；向量那路是余弦，
   恒在 [-1,1]。两者不可比，任何 `a*bm25 + b*cos` 的系数都是拍脑袋，且随查询
   长度漂移。RRF 只用**排名**：`score = Σ w_route / (k + rank)`，天然无量纲，
   也天然抗某一路的分数尺度突变。k=60 是原论文（Cormack 2009）的经验值 ——
   k 越大越「平权」，越小的 k 越奖励单路的第一名。

2. **为什么软加权而不是硬过滤**
   问「联影医疗营业收入」，硬过滤到 688271 是对的；但问「哪些公司净利润同比
   下滑」，硬过滤会把 17 家里的 16 家直接掐死 —— 而问题本身恰恰需要跨公司召回。
   所以公司识别只用来**加权**（识别出且命中 → 分数乘以权重），不删候选。
   只有用户在命令行**显式** `--company` 时才走硬过滤（走 `store.allow_mask`，
   在打分前生效，见 store.py 的注释）。

3. **为什么要按章节加权**
   同一个数字在半年报里往往出现两次：一次在 `第三节 管理层讨论与分析` 的正文
   （「营业收入同比增长 12.34%」），一次在 `第八节 财务报告` 的合并利润表里。
   两者都对，但对「营业收入是多少」这类问题，第八节的表格是**原始凭证**，
   第三节是**转述**。加权把第八节顶上来，答案更可核验。
   这是本项目最直接的动机：`联影医疗 营业收入` 的 BM25 第 1 名是
   `688271-0185`（第六节 股份变动及股东情况），而非报表本体。

权重的默认值都写在 `Weights` 里，全部可通过参数覆盖 —— 评测脚本靠它做消融。

**已知失效模式（双路，2026-09-27 全量实测，4657 条）**：问「联影医疗 2026 年上半年
营业收入是多少」时，BM25 与向量**两路都答不对**，且失败模式同源。答案在
`688271-0009`/`0010`（第二节 > 六、公司主要会计数据和财务指标），实测：
BM25 全库 **#795 / #593**，向量全库 **#541 / #596** —— 深度 20/50/100 都进不了池子，
软加权更不生效（只能在池内重排）。两路的第 1 名都不含「营业收入」：BM25 第 1 是
`688271-0135`（第四节 ESG 正文），向量第 1 是 `688271-0001`（报告前部）。

成因（已定位，非猜测）：**那张会计数据表里没有公司名**，而第三节/第四节的散文
反复写全称「上海联影医疗科技股份有限公司」；`联影`/`医疗` 在 17 份报告里都稀有
（高 IDF），实体词因此压过内容词。向量侧同理 —— 17 家公司的「主要会计数据」表
正文几乎同构且都不含公司名，模型无从分辨哪张是联影的。

**曾经的三条预期都已被实测推翻**，记在这里免得再走一遍：
  * 「以现有权重能把报表本体从第 24 名拉到第 10 名附近」——误记，第 24 名是
    `688271-0063`（第三节正文）；且加权对 #795 的块无效。
  * 「这是向量路该解决的问题（语义匹配强）」——向量路给同一块 #541，同样答不对。
  * 「再加一路表头索引（公司+章节+小节+首行）就能救」——**已试，无效**：联影 378 个
    块的表头几乎一样（同公司、只有 9 个章节），判别力≈0，gold 只从 #795 升到 #301，
    比正文路还差。别再加这类关键词规则。

**已实现并实测的两条修法**（同一套 20 题，见 `evaluation/results/` 与
`scripts/compare_eval.py`）：

  * **A. 索引文本前置元数据表头**（`build_index.py --header` → `index/header/`）。
    BM25 单独：hit@5 0.050→0.300、hit@10 0.150→0.300、MRR@10 0.066→0.139，
    救回 3 题（Q08/Q10/Q15）**弄坏 0 题**；旗舰题的 gold `688271-0009` 在 BM25
    全库从 **#795 升到 #132**（分数 4.759→13.658），`0010` 从 #591 升到 #46。
    注意它仍进不了 Top-20 —— BM25 路的改善是「显著缩小差距」，不是「彻底解决」；
    真正越过第六条阈值的是向量路（对抗集实测 cos 0.3433→0.6659 vs 第 6 名 0.6605）。
  * **B. 跨公司条件筛选增强：走结构化表组**（`structured.py`，免重算）。筛选题
    hit@10 0.000→**1.000**、MRR@10 0→0.669，证据覆盖 **17/17 家公司**，其余 15 题
    一条未变。判据是「问题要一个集合（哪些/几家）且没点名公司」——点名了就退回
    相似度路。
    ⚠ **别把 17/17 当准确率**：那是「表组覆盖了几家公司的表」，命中定义与 gold
    同源（口径说明见 README §12.4 与 structured.py 顶部），且这条路只负责把表摆上
    桌、不负责读表算数。它**不改善**其余题型：另外 15 题一条未变就是证据。

  ⚠ 读 A 的数字时要带上口径：那是 **BM25 单路**的对照（向量索引当时还在重算）。
  重算完成后要重跑三路对照，别拿半张表当结论。

用法：
    from src.retrieval.retriever import retrieve

    res = retrieve("联影医疗 2026 上半年营业收入是多少")
    res.hits[0].row, res.hits[0].scores["final"]

命令行自检（只用 BM25 也能跑）：
    python src/retrieval/retriever.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402
from src.retrieval.entities import Company, find_companies, load_companies  # noqa: E402
from src.retrieval.store import Hit, Store, store  # noqa: E402
from src.retrieval.structured import (  # noqa: E402
    coverage,
    is_screening,
    structured_rows,
)

RRF_K = 60
# 两路的相对权重。向量路建好之前它不存在，权重表里留好位置即可。
ROUTE_WEIGHTS: dict[str, float] = {"bm25": 1.0, "vector": 1.0}

# 章节偏好：关键词 → 章节**前缀**（用前缀匹配而不是全等，免得跟标题里的
# 标点变动较劲：`第八节 财务报告` 这种带节号的前缀稳定）。
# 每组第一个是主章节（拿满 `Weights.section`），其余是次章节（拿 `section_secondary`）。
# 按顺序匹配，**首个命中的组生效**，所以数值组要放在泛化的「原因」组前面。
TOPIC_SECTIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # 具体数值 → 报表本体（第八节），其次公司简介里的主要财务指标（第二节）
    (("营业收入", "营收", "营业成本", "毛利", "净利润", "归母", "扣非", "研发投入",
      "研发费用", "销售费用", "管理费用", "财务费用", "现金流", "总资产", "净资产",
      "负债", "每股收益", "货币资金", "应收账款", "存货", "资产负债率"),
     ("第八节", "第二节")),
    # 变动原因 / 业务经营 → 管理层的讨论与分析（第三节），报表是佐证
    (("原因", "为什么", "为何", "变动", "增长", "下滑", "下降", "同比", "业务",
      "行业", "竞争", "主营", "经营情况", "战略"),
     ("第三节", "第八节")),
    # 股东与股本
    (("股东", "股份变动", "质押", "持股", "解禁", "回购", "增持", "减持", "股本"),
     ("第六节",)),
    # 治理 / 环境 / 社会 / 人力
    (("治理", "环境", "社会", "员工", "薪酬", "董事会", "监事", "高管", "环保"),
     ("第四节",)),
    # 重大事项。（分红方案不在第八节的报表里，而在本节的利润分配段落 ——
    # 所以「分红」归这里，别放进上面的数值组。）
    (("诉讼", "仲裁", "担保", "关联交易", "承诺", "募集资金", "重大合同", "处罚",
      "分红", "利润分配", "派息"),
     ("第五节",)),
    # 名词解释
    (("释义", "简称", "名词"),
     ("第一节",)),
)

# 表格块的加成只给「问具体数字」的场景。表格块最长 13758 字，进 prompt 要截断，
# 无差别顶上来会挤掉正文块，所以这个权重刻意给小。
NUMERIC_HINTS = ("多少", "是多少", "金额", "占比", "比例", "率", "同比", "几", "数值")

# 「要解释」的显式线索。问「净利润为什么下滑」时，问题里既有数值词（净利润）
# 又有归因词（为什么/原因），按 TOPIC_SECTIONS 的顺序会被数值组先抢走，
# 于是把答案顶到报表里 —— 而原因写在第三节的文字里。所以归因线索单独前置判定。
EXPLAIN_HINTS = ("原因", "为什么", "为何", "说明", "解释", "reason")



@dataclass(frozen=True)
class Weights:
    """软加权的全部旋钮。全 1.0 即「不加权」，评测脚本用它做消融。"""

    company_one: float = 1.5     # 问题只提到 1 家公司，且该块属于它
    company_many: float = 1.2    # 提到 ≥2 家（跨公司对比），命中的任何一家
    section: float = 1.2         # 命中主章节
    section_secondary: float = 1.1   # 命中次章节
    table: float = 1.15           # 问数值 且 该块是表格


# 消融用：全 1.0 即「关掉所有软加权」
NO_WEIGHTS = Weights(1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    companies: list[Company] = field(default_factory=list)
    routes: tuple[str, ...] = ()
    recall_counts: dict[str, int] = field(default_factory=dict)
    merged_duplicates: int = 0          # 内容重复被合并掉的块数
    hard_filter: list[str] | None = None
    # 结构化表组路由的说明（None = 没走这条路）。条条件筛选类问题走它时，
    # 证据不再是 Top-K 相似度，而是「全部公司的同一张表」—— 页面上要显式说明，
    # 否则用户看到 17 条证据会以为检索失控了。
    structured: str | None = None
    structured_coverage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"<RetrievalResult {len(self.hits)} hits routes={self.routes} "
                f"companies={[c.stock_code for c in self.companies]}>")


def _preferred_sections(query: str) -> tuple[str, str | None]:
    """返回 (主章节前缀, 次章节前缀)；无偏好时返回 ("", None)。

    归因类线索先判定（见 EXPLAIN_HINTS），其余按 TOPIC_SECTIONS 顺序首个命中生效。
    """
    if any(h in query for h in EXPLAIN_HINTS):
        return "第三节", "第八节"
    for kws, secs in TOPIC_SECTIONS:
        if any(k in query for k in kws):
            return secs[0], (secs[1] if len(secs) > 1 else None)
    return "", None


def _looks_numeric(query: str) -> bool:
    return any(h in query for h in NUMERIC_HINTS)


def _score_row(row: int, info: dict, weights: Weights, codes: set[str],
               primary: str, secondary: str | None, numeric: bool) -> float:
    w = 1.0
    if codes:
        if info["stock_code"] in codes:
            w *= weights.company_one if len(codes) == 1 else weights.company_many
    sec = info["section"]
    if primary and sec.startswith(primary):
        w *= weights.section
    elif secondary and sec.startswith(secondary):
        w *= weights.section_secondary
    if numeric and info["content_type"] == "table":
        w *= weights.table
    return w


def resolve_company_arg(spec: str) -> list[str]:
    """把命令行里的公司写法（代码 / 证券简称 / 官方全称）解析成 stock_code 列表。

    认不出来就原样返回，让 store.allow_mask 抛错并列出可选代码 —— 报错信息
    比这里自己再写一遍「无法识别」更有用。
    """
    s = (spec or "").strip()
    if not s:
        return []
    out = []
    for token in s.replace("，", ",").split(","):
        t = token.strip()
        if not t:
            continue
        hit = [c for c in load_companies()
               if t in (c.stock_code, c.security_name, c.full_name)]
        out.append(hit[0].stock_code if hit else t)
    return out


def retrieve(query: str,
             *,
             top_k: int | None = None,
             k_bm25: int | None = None,
             k_vector: int | None = None,
             routes: tuple[str, ...] | list[str] | None = None,
             company: list[str] | str | None = None,
             weights: Weights | None = None,
             rrf_k: int = RRF_K,
             route_weights: dict[str, float] | None = None,
             structured: bool | None = None,
             st: Store | None = None) -> RetrievalResult:
    """召回 → 融合 → 加权 → Top-K。

    top_k/k_bm25/k_vector 默认取 config（TOP_K_FINAL / TOP_K_BM25 / TOP_K_VECTOR）。
    routes=None 表示「能用哪路就用哪路」：向量索引缺失时自动降级为 bm25，
    显式传了 "vector" 却不可用则记 warning 并跳过（不抛，否则评测全跑不了）。
    structured=None 取 config.structured_routing；传 False 即消融掉结构化表组路由。
    """
    st = st or store()
    weights = weights or Weights()
    use_structured = (config.structured_routing if structured is None
                      else bool(structured))
    rw = dict(ROUTE_WEIGHTS)
    if route_weights:
        rw.update(route_weights)
    top_k = int(top_k if top_k is not None else config.top_k_final)
    k_bm25 = int(k_bm25 if k_bm25 is not None else config.top_k_bm25)
    k_vector = int(k_vector if k_vector is not None else config.top_k_vector)

    warnings: list[str] = []
    if routes is None:
        # 必须**先尝试装载**再决定，不能看 st.has_vector —— FAISS 是惰性装载的，
        # 刚进这个函数时它必然是 False。之前按 has_vector 判断，结果是：
        # 向量索引建好之后，凡是不显式传 routes 的调用（ask.py、app、qa）
        # 依然静默只走 BM25，而日志里看不出任何异常。下面 ensure_vector()
        # 失败会记 warning 并降级，那才是正确的降级路径。
        routes = ("bm25", "vector")
    routes = tuple(routes)
    if "vector" in routes and not st.ensure_vector():
        warnings.extend(st.warnings[-1:])
        routes = tuple(r for r in routes if r != "vector")
        if not routes:
            routes = ("bm25",)
    if not routes:
        raise ValueError("routes 为空")

    # 显式指定公司 → 硬过滤（打分前生效）；否则只用识别结果做软加权
    if isinstance(company, str):
        company = resolve_company_arg(company)
    allow = st.allow_mask(company) if company else None
    companies = find_companies(query)
    codes = {c.stock_code for c in companies}
    if allow is not None:
        codes = set(company or [])          # 硬过滤时，加权集合就是被允许的集合

    primary, secondary = _preferred_sections(query)
    numeric = _looks_numeric(query)

    # ---- 逐路召回，记录 rank 与原始分 ----
    per_route: dict[str, list[Hit]] = {}
    for r in routes:
        k = {"bm25": k_bm25, "vector": k_vector}[r]
        per_route[r] = (st.search_bm25(query, k=k, allow=allow) if r == "bm25"
                        else st.search_vector(query, k=k, allow=allow))

    # ---- RRF 融合 ----
    rrf: dict[int, float] = {}
    raw: dict[int, dict[str, float]] = {}
    src: dict[int, str] = {}
    for r, hits in per_route.items():
        for h in hits:
            rrf[h.row] = rrf.get(h.row, 0.0) + rw.get(r, 1.0) / (rrf_k + h.rank)
            raw.setdefault(h.row, {})[r] = h.score
            src.setdefault(h.row, h.chunk_id)

    # ---- 软加权 ----
    scored: list[tuple[float, float, int]] = []      # (final, w, row)
    for row, base in rrf.items():
        w = _score_row(row, st.df.iloc[row], weights, codes, primary, secondary,
                       numeric)
        scored.append((base * w, w, row))
    scored.sort(key=lambda t: (-t[0], t[2]))

    # ---- 结构化表组：条件筛选类问题不再用相似度排序 ----
    # 判据与理由见 structured.py。三个前置条件缺一不可：
    #   * config.structured_routing 打开（可关，评测里做消融）
    #   * 问题里**没有**识别到具体公司（点了名就是单点/对比题）
    #   * 也没有硬过滤（既然已限定公司，就没有「跨公司枚举」可言）
    struct_rows: list[int] = []
    struct_note: str | None = None
    struct_cov: dict = {}
    if use_structured and allow is None and is_screening(query, companies):
        struct_rows, struct_note = structured_rows(st.df, query)
        if struct_rows:
            struct_cov = coverage(st.df, struct_rows)
        else:
            struct_note = None          # 没定位到表组 → 老老实实退回相似度路

    # ---- 去重 + 截断 ----
    # 39 条 chunk 内容两两重复（同一张表跨页被切了两块）。若两块都进 Top-5，
    # 等于白占一个名额，所以按正文去重、保留排名靠前的那块，并把被合并掉的
    # chunk_id 挂在命中上 —— 评测时 gold 若标的是另一块，不该算漏召回。
    hits: list[Hit] = []
    seen: dict[str, Hit] = {}
    merged = 0
    # 结构化表组排在最前（它是确定集，不是候选），相似度结果接在后面补充。
    for i, row in enumerate(struct_rows, 1):
        info = st.df.iloc[row]
        h = Hit(row=row, chunk_id=info["chunk_id"], score=round(1.0 - i * 1e-6, 6),
                source="structured", rank=i,
                scores={"structured": 1.0, "final": round(1.0 - i * 1e-6, 6)})
        seen[info["content"]] = h
        hits.append(h)
    # 走结构化路由时证据上限随之放宽：Top-6 装不下 17 家公司的同一张表，
    # 少一家就漏一个正确答案。补充条数仍是 config.top_k_final。
    limit = top_k + len(hits) if struct_rows else top_k
    for final, w, row in scored:
        info = st.df.iloc[row]
        content = info["content"]
        keep = seen.get(content)
        if keep is not None:
            keep.dup_chunk_ids.append(info["chunk_id"])
            merged += 1
            continue
        if len(hits) >= limit:
            continue
        h = Hit(row=row, chunk_id=info["chunk_id"], score=round(final, 6),
                source="+".join(sorted(per_route)), rank=len(hits) + 1,
                scores={**raw.get(row, {}), "rrf": round(rrf[row], 6),
                        "weight": round(w, 4), "final": round(final, 6)})
        seen[content] = h
        hits.append(h)

    return RetrievalResult(query=query, hits=hits, companies=companies,
                           routes=routes,
                           recall_counts={r: len(hs) for r, hs in per_route.items()},
                           merged_duplicates=merged, hard_filter=company,
                           structured=struct_note, structured_coverage=struct_cov,
                           warnings=warnings)


def main() -> int:
    st = store()
    print(st.describe())
    print(f"RRF k={RRF_K}，权重 {Weights()}")

    queries = [
        "联影医疗 2026 上半年营业收入是多少",
        "心脉医疗的研发投入占营业收入比例",
        "哪些公司归母净利润同比下滑？原因是什么",
    ]
    for q in queries:
        res = retrieve(q, top_k=5)
        prim, sec = _preferred_sections(q)
        print(f"\n『{q}』")
        print(f"  识别公司 {[c.security_name for c in res.companies] or '无'}"
              f" | 章节偏好 {prim or '无'}{'/' + sec if sec else ''}"
              f" | 各路召回 {res.recall_counts} | 合并重复 {res.merged_duplicates}")
        for h in res.hits:
            r = st.record(h.row)
            print(f"  {h.rank}. final={h.scores['final']:7.5f} w={h.scores['weight']:.2f} "
                  f"rrf={h.scores['rrf']:.5f} "
                  f"bm25={h.scores.get('bm25', float('nan')):7.3f} | "
                  f"{r['chunk_id']} {r['company'][:8]} | {r['section']} | "
                  f"p{r['page_start']} | {r['content_type']} | {r['char_count']}字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
