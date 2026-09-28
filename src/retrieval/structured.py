"""结构化表组路由：条件筛选类问题不走相似度排序，直接取**确定的那一组表**。

> ## ⚠️ 定位：**跨公司条件筛选增强**，不是通用召回能力
>
> 这条路是为「17 家公司里……的有哪几家」这**一类**问题加的确定性取表，触发条件
> 严格（问题要一个集合，且**没有**点名任何公司）。报告里出现的 **17/17** 是
> 「表组覆盖了几家公司的表」——一个**覆盖率**，**不是本系统的 RAG 准确率**，
> 也不能外推到其他题型（单点数值、同比原因、公司对比全部不触发它）。
>
> 它也**不保证答对**：表组只保证「该看的表都在桌上」，读表、比较、下结论是模型的
> 事。评测集 T02 专考这一点 —— 17 张表都摆上来，但表里没有「员工离职率」，模型
> 仍必须回答「未披露」；此时 17/17 的覆盖率一点忙都帮不上。
>
> 通用检索质量只看 §12 的三路 hit/recall/MRR，以及正式 10 题的逐题记录。

## 为什么必须单独有一条路

「17 家公司里，2026 上半年归母净利润同比下降的有哪几家？」这类问题的证据需求
与「联影医疗营收多少」**不是同一种东西**：后者要的是「最相似的几条」，前者要的是
**17 家公司各一张表**。Top-K 相似度检索在这里是结构性错配 —— 无论把 K 调到 6 还是
60，模型看到的都是「和问句最像的若干块」，而不是「所有公司的同一张表」。实测基线
也印证了这点：5 道条件筛选题在 BM25/向量/混合三路都是 hit@10 = 0（见
`evaluation/results/`），且返回的都是邻近的说明段而非表体。

这不是「调参能救」的问题，所以不再调 K、不再加关键词权重 —— 那些都只在池内重排。

## 确定性从哪来（本语料的一条硬结构）

`df.subsection` 含「主要会计数据」的块共 **37 条，恰好覆盖全部 17 家公司**，且结构
完全一致：

    <公司>-00NN  table  会计数据表（营业收入 / 归母净利润 / 现金流 / 总资产，450~600 字）
    <公司>-00NN  table  主要财务指标表（研发投入占营业收入的比例 / ROE / 每股收益，300~480 字）
    <公司>-00NN  text   变动原因说明（**只有 3 家有**：688108 / 688271 / 688289）

标题有两种写法（`六、` / `七、` / `六、 ` 带空格），所以匹配用「包含」不用等号。
六家上交所主板公司（600055/600587/603309/603387/603658/605369）用「七、」，科创板
用「六、」—— 这正是评测集 Q19 注释里那条「6 家不披露该指标」的由来。

## 与「加一路表头检索」的区别

那个方案（已实测推翻）是**再建一路召回**，然后让它跟正文路在 RRF 里抢名额 ——
表头判别力≈0，抢不过，gold 只从 #795 升到 #301。这里不是召回，是**按元数据直接
定位**：不参与打分、不参与融合，只回答「该看哪些块」这个集合问题。

## 诚实的边界

这一路能覆盖的是**结构化的筛选**（同一张表、跨全部公司）。它不解决「哪家公司的
XX 最高」这类需要读取表内数值再比较的问题 —— 那是模型读表的事，不是检索的事。
也**不能**用来回答表里没有的字段（如员工离职率）：结构化路由会把 17 张表都摆上来，
但表里没有那个指标，模型仍必须回答「未披露」（评测集 T02 就在考这一点）。

用法：
    from src.retrieval.structured import is_screening, structured_rows

    if is_screening(q, companies):          # companies 为空才会触发
        rows, note = structured_rows(df, q)

自检（打印 17 家的分组结果，确认两个家族各 17 条、无遗漏无重复）：
    python src/retrieval/structured.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

# 「要一个集合」的线索。刻意**不含**「哪家」——「联影和心脉哪家营收更高」也含
# 「哪家」，但那是两家对比题，答案在两张表里而不需要枚举集合。
SCREEN_HINTS: tuple[str, ...] = (
    "哪些", "哪几家", "哪几个", "几家", "有哪些", "所有公司", "全部公司", "家公司",
)

# 「要比例」的线索 → 取财务指标表；否则取会计数据表。
RATIO_HINTS: tuple[str, ...] = ("占比", "比例", "比重", "百分比", "收益率",
                                "每股收益", "率是", "率为")
# 「要原因」的线索 → 额外带上同小节的说明段（只有 3 家有）
REASON_HINTS: tuple[str, ...] = ("原因", "为什么", "为何", "说明", "解释")

FAMILY_DATA = "数据"      # 会计数据表（营收 / 归母净利润 / 现金流 / 总资产）
FAMILY_RATIO = "指标"     # 主要财务指标表（研发投入占比 / ROE / 每股收益）


def is_screening(query: str, companies=None) -> bool:
    """是不是「要一个集合」的问题。

    **点了名的公司直接返回 False**：「联影医疗在哪些小节披露了研发投入」也含
    「哪些」，但它有明确主体，应该走正常的相似度召回（并靠公司软加权/硬过滤收窄），
    而不是把 17 家公司的表全铺上来。识别到公司实体是这里唯一的否证条件。
    """
    if companies:
        return False
    return any(h in query for h in SCREEN_HINTS)


# 分子靠**行标签**投票，不靠「出现/不出现」。
# 为什么不能只判关键词：块是按页切的，数据表的块尾会捎带下一小节的标题
# 「(二) 主要财务指标」，所以「财务指标」这四个字在 34 张表里**全都有**
# （实测），单判它会 34/34 全错。行标签才是表体的判别特征。
MARK_DATA: tuple[str, ...] = (
    "归属于上市公司股东的净利润", "扣除非经常性损益", "经营活动产生的现金流量净额",
    "利润总额", "总资产", "营业收入",
)
MARK_RATIO: tuple[str, ...] = (
    "加权平均净资产收益率", "研发投入占营业收入的比例", "基本每股收益",
    "稀释每股收益", "每股净资产",
)


def family_of(content: str) -> str | None:
    """把一个小节内的块分成「会计数据表」还是「主要财务指标表」。

    实测这组标记在 34 张表上给出 34/34 的正确划分（数据表 A≥5/B=0，指标表 A≤3/B≥4），
    自检里断言了这一点 —— 语料一变就会在这里炸，而不是静默取错表。
    """
    a = sum(m in content for m in MARK_DATA)
    b = sum(m in content for m in MARK_RATIO)
    if a == 0 and b == 0:
        return None
    return FAMILY_RATIO if b > a else FAMILY_DATA


def structured_rows(df: pd.DataFrame, query: str,
                    subsection_kw: str | None = None) -> tuple[list[int], str]:
    """返回 (行号列表, 说明文字)。行号按 chunk_id 升序 = 按公司分组，便于模型逐家读。

    只认 `subsection` 含关键词、且 `content_type == "table"` 的块 —— 说明段是
    text，仅在问题问「原因」时才附加。
    """
    kw = subsection_kw or config.structured_subsection
    m = df["subsection"].fillna("").str.contains(kw, regex=False)
    if not m.any():
        return [], f"无块匹配 subsection 含 {kw!r}（语料结构变了？）"

    sub = df[m]
    want_ratio = any(h in query for h in RATIO_HINTS)
    want_reason = any(h in query for h in REASON_HINTS)
    target = FAMILY_RATIO if want_ratio else FAMILY_DATA

    rows: list[int] = []
    for row, r in sub.iterrows():
        if r["content_type"] != "table":
            continue                    # 说明段最后单独处理
        if family_of(r["content"]) == target:
            rows.append(int(row))
    # 说明段（text）只在问原因时加：它讲的是「为什么变动」，与筛选本身无关，
    # 无差别加进来会把 17 张表的信号稀释掉。
    if want_reason:
        rows += [int(row) for row, r in sub.iterrows() if r["content_type"] == "text"]

    rows.sort(key=lambda i: df["chunk_id"].iloc[i])
    note = (f"结构化表组：第二节「{kw}」下的"
            f"{'主要财务指标表' if want_ratio else '会计数据表'}"
            f"（每家公司 1 张，共 {len(rows)} 条"
            + ("，另附变动原因说明段" if want_reason else "") + "）")
    return rows, note


def coverage(df: pd.DataFrame, rows: list[int]) -> dict:
    """这组表覆盖了哪些公司 —— 给评测用，作为**不循环**的覆盖度量。

    评测集里条件筛选题的 gold 就是这些块，所以「hit@10 从 0 到 1」在定义上必然发生，
    不能当作效果证据。真正有信息量的是「17 家公司里覆盖了几家」：漏一家就会漏掉
    一个正确答案（比如漏了唯一亏损的那家，答案就少一家）。
    """
    codes = {df["stock_code"].iloc[i] for i in rows}
    return {"n_rows": len(rows), "n_companies": len(codes),
            "companies": sorted(codes)}


def main() -> int:
    from src.indexing.build_index import load_chunks

    df = load_chunks()
    n_all = df["stock_code"].nunique()
    print(f"语料 {len(df)} 条，{n_all} 家公司")
    print(f"匹配小节：subsection 含 {config.structured_subsection!r}")

    # 先断言表族划分本身是对的：这一步不依赖任何查询，纯粹校验 34 张表的
    # 分类是否恰好分成两个 17。分类错了两路选取都会静默取错表。
    sub = df[df["subsection"].fillna("").str.contains(
        config.structured_subsection, regex=False) & (df["content_type"] == "table")]
    fam = sub["content"].map(family_of)
    cnt = fam.value_counts().to_dict()
    print(f"表族划分：{cnt}（共 {len(sub)} 张）")
    assert cnt.get(FAMILY_DATA) == n_all and cnt.get(FAMILY_RATIO) == n_all \
        and fam.notna().all(), f"表族划分不是 17/17：{cnt} —— 标记集需要复核"

    ok = True
    for q in ("17 家公司里，2026 年上半年归母净利润同比下降的有哪几家？",
              "17 家公司里，研发投入占营业收入的比例超过 15% 的有哪几家？",
              "哪些公司归母净利润为负？原因是什么"):
        rows, note = structured_rows(df, q)
        cov = coverage(df, rows)
        print(f"\n『{q}』")
        print(f"  {note}")
        print(f"  覆盖 {cov['n_companies']}/{n_all} 家公司，{cov['n_rows']} 条")
        got = pd.Series([df["stock_code"].iloc[i] for i in rows]).value_counts().to_dict()
        print(f"  每公司条数：{got}")
        if cov["n_companies"] != n_all:
            ok = False
            print("  ✗ 覆盖不全！")

    print("\n--- 否证检查（这些不该触发结构化路由）---")
    for q in ("联影医疗和心脉医疗哪家营业收入更高？",
              "联影医疗在哪些小节披露了研发投入？",
              "哪些公司 2026 上半年归母净利润同比下滑？"):
        hit = is_screening(q, companies=["688271"] if "联影" in q else [])
        print(f"  {'触发' if hit else '不触发'}  {q}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
