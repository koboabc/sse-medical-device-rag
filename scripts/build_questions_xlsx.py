"""生成 evaluation/questions.xlsx —— 课程要求的 10 道评测题（题目 + 标准答案 + 出处页码）。

为什么标准答案必须由脚本「验」一遍，而不是由人「写」一遍：
    `expected_answer` 里的每一个数字都必须是半年报里**逐字存在**的。出题时最容易
    犯的错不是编造整段答案，而是把表格里的数字**记串**（17 家 × 6 个指标 ≈ 100 个数），
    或者把「推导值」（如两个毛利率相减）当成披露值写进去。所以本脚本第 1 步就是
    把 expected_answer 里的数字 token 抽出来，逐个去 gold 块的正文里查；查不到即
    `[FAIL]` 并**拒绝写文件**。这是「不允许凭模型知识生成」的机械保证。

三道校验：
    1. gold_chunk_ids 必须真实存在（否则后面评测恒为 0，是静默错误）；
    2. expected_answer 的数字 token 必须逐字出现在 gold 正文里（去掉千分位逗号后比对）；
    3. 检索自检（可选，`--no-retrieval` 跳过）：拿现在跑着的表头索引 + 混合召回
       跑一遍，看 gold 在不在 Top-10 里 —— 记下来，但不改题目。题目该由「半年报里
       有没有明确答案」决定，不能由「本系统现在能不能召回」决定，否则评测集就变成
       了给自己挑好做的题。

产物：
    evaluation/questions.xlsx        —— sheet1 六个规定字段；sheet2 答案依据（原文摘录）；
                                        sheet3 检索自检
    evaluation/datasets/questions.jsonl —— 同内容的机器可读版，字段与 qa.jsonl 一致，
                                        可直接喂给 scripts/eval_retrieval.py

用法：
    python scripts/build_questions_xlsx.py              # 含检索自检
    python scripts/build_questions_xlsx.py --no-retrieval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 编号用 R01–R10（R = required，课程要求的 10 题），**刻意不与 qa.jsonl /
# qa_formal10.jsonl 的 Q01–Q20 / T01–T05 共用编号空间**：那两套里 Q01 是
# 「联影营收」、Q04 是「拱东营收」，与这里的 R01「新华营收」含义不同，
# 同名不同义在对读两份结果时是会真出错的。
# ---------------------------------------------------------------------------
# 10 道题：8 道单公司题（覆盖 8 个主题，8 家不同公司）
#          + 2 道跨公司题（两家比较 1 道、多家综合 1 道，覆盖另外 9 家）
# 17 家公司恰好各出现一次 —— 这不是巧合，是排题时刻意做的：
#   单公司 8 家：600587 603387 688016 688351 603309 688271 688013 688617
#   跨公司 9 家：603658 600055 + 605369 688050 688289 688358 688085
# 公司名一律写「简称」，与 data/manifest.csv 的 security_name 一致，
# 这样 entities.find_companies() 能直接认出来（不会因为换了个说法而漏识别）。
# ---------------------------------------------------------------------------
QUESTIONS: list[dict] = [
    {
        "qid": "R01",
        "topic": "营业收入",
        "scope": "单公司",
        "question": "新华医疗 2026 年上半年营业收入是多少元？与上年同期相比增减多少？",
        "target_company": "新华医疗(600587)",
        "company": ["600587"],
        "expected_answer": (
            "5,160,672,255.84 元；上年同期 4,790,414,888.01 元，同比增加 7.73%。"
        ),
        "gold_chunk_ids": ["600587-0007"],
        "expected_pages": "600587: p5",
        "quote": "| 营业收入 | 5,160,672,255.84 | 4,790,414,888.01 | 7.73 |",
        "note": "第二节 > 七、公司主要会计数据和财务指标 > (一) 主要会计数据。"
                "新华医疗是 17 家里营业收入绝对值最大的一家（51.6 亿）。",
    },
    {
        "qid": "R02",
        "topic": "净利润",
        "scope": "单公司",
        "question": "基蛋生物 2026 年上半年归属于上市公司股东的净利润是多少元？同比变化多少？",
        "target_company": "基蛋生物(603387)",
        "company": ["603387"],
        "expected_answer": (
            "184,291,348.06 元；上年同期 108,808,763.23 元，同比增长 69.37%。"
        ),
        "gold_chunk_ids": ["603387-0009"],
        "expected_pages": "603387: p6-7",
        "quote": "| 归属于上市公司股东的净利润 | 184,291,348.06 | 108,808,763.23 | 69.37 |",
        "note": "第二节 > 七、公司主要会计数据和财务指标 > (一) 主要会计数据。"
                "17 家里归母净利润同比增幅最大的一家。",
    },
    {
        "qid": "R03",
        "topic": "毛利率/盈利能力",
        "scope": "单公司",
        "question": "心脉医疗 2026 年上半年（6 月末）主营业务毛利率是多少？"
                    "与 2025 年度相比是上升还是下降？",
        "target_company": "心脉医疗(688016)",
        "company": ["688016"],
        "expected_answer": (
            "2026 年 6 月末主营业务毛利率为 71.36%；报告同时披露 2025 年度为 71.09%、"
            "2024 年度为 72.98%，即较 2025 年度有所上升。"
        ),
        "gold_chunk_ids": ["688016-0031"],
        "expected_pages": "688016: p21-22",
        "quote": "2024 年度、2025 年度和2026 年6 月末，公司主营业务毛利率分别为72.98%、"
                 "71.09%和71.36%，毛利率存在一定波动。",
        "note": "第三节 > 四、风险因素 的「毛利率波动的风险」条目 —— 三年数字写在同一句里，"
                "顺序是「2024、2025、2026H1」，取错位就会答成 72.98% 或 71.09%。"
                "★上升幅度 0.27 个百分点（71.36−71.09）是**推导值**，报告没有直接给出，"
                "故未写进 expected_answer。",
    },
    {
        "qid": "R04",
        "topic": "研发投入",
        "scope": "单公司",
        "question": "微电生理 2026 年上半年研发投入合计是多少元？"
                    "研发投入占营业收入的比例是多少，同比有何变化？",
        "target_company": "微电生理(688351)",
        "company": ["688351"],
        "expected_answer": (
            "研发投入合计 46,801,080.41 元；上年同期 46,192,117.08 元，同比增加 1.32%。"
            "研发投入总额占营业收入比例为 17.00%，同比减少 3.64 个百分点"
            "（上年同期为 20.64%）。"
        ),
        "gold_chunk_ids": ["688351-0024", "688351-0008"],
        "expected_pages": "688351: p15-16; 688351: p7",
        "quote": "| 研发投入合计 | 46,801,080.41 | 46,192,117.08 | 1.32 |\n"
                 "| 研发投入总额占营业收入比例 （%） | 17.00 | 20.64 | 减少3.64个百分点 |",
        "note": "★两个数字在**两个章节**：金额在第三节 > 三、报告期内核心竞争力分析 > "
                "「3、研发投入情况表」（688351-0024）；比例在第二节 主要财务指标（688351-0008）。"
                "只召回到一处就答不全，是刻意保留的跨章节题。"
                "另注：公司全称「上海微创电生理医疗科技股份有限公司」，证券简称「微电生理」——"
                "题面用证券简称，因为别名表里没有「微创电生理」（用它会认不出公司，"
                "公司识别是软加权 company_one 的触发条件）。",
    },
    {
        "qid": "R05",
        "topic": "经营现金流",
        "scope": "单公司",
        "question": "维力医疗 2026 年上半年经营活动产生的现金流量净额是多少元？同比变化多少？",
        "target_company": "维力医疗(603309)",
        "company": ["603309"],
        "expected_answer": (
            "160,232,989.24 元；上年同期 33,854,199.97 元，同比增加 373.30%。"
        ),
        "gold_chunk_ids": ["603309-0008"],
        "expected_pages": "603309: p5-6",
        "quote": "| 经营活动产生的现金流量净额 | 160,232,989.24 | 33,854,199.97 | 373.30 |",
        "retrieval_note": (
            "实测机理＝**融合截断**（非深度不足）：gold 603309-0008 在向量路全库 **#10**、"
            "BM25 全库 #99 —— 向量路已经把它捞进了 k=20 候选池，是 RRF 融合后的 Top-10 "
            "排序把它挤出去的。与 Q01 / Q04 同一机理，调大深度救不回来（池里本来就有）。"
        ),
        "note": "第二节 > 七、公司主要会计数据和财务指标 > (一) 主要会计数据。"
                "注意与「归母净利润同比 −15.76%」方向相反（净利降、现金流大增），"
                "只记住一个数就会答反。",
    },
    {
        "qid": "R06",
        "topic": "主要业务变化",
        "scope": "单公司",
        "question": "联影医疗 2026 年上半年营业收入同比增长 17.22%，"
                    "公司说明收入增量主要来自哪里？",
        "target_company": "联影医疗(688271)",
        "company": ["688271"],
        "expected_answer": (
            "增量主要来自海外业务的持续放量与国内份额的逆势提升：公司说明，"
            "在国内医学影像与放射治疗设备市场招投标阶段性收缩的背景下，"
            "公司仍实现 17.22% 的收入增长。"
        ),
        "gold_chunk_ids": ["688271-0038"],
        "expected_pages": "688271: p39-40",
        "quote": "增量主要来自海外业务的持续放量与国内份额的逆势提升。",
        "note": "第三节 > 二、经营情况的讨论与分析（text 块，非表格）。"
                "同一块还给了「归母净利润同比下降 10.12%，主要受汇兑损益影响，以及"
                "全球化运营体系建设、前沿技术研发、市场能力拓展等投入增加」——"
                "可作追问，但本题只问增量来源。",
    },
    {
        "qid": "R07",
        "topic": "主要产品",
        "scope": "单公司",
        "question": "天臣医疗 2026 年半年报披露的核心业务与主要产品有哪些？",
        "target_company": "天臣医疗(688013)",
        "company": ["688013"],
        "expected_answer": (
            "核心业务持续聚焦外科手术关键环节（包括病变组织分离、切除及器官功能重建等）"
            "所需的多类吻合器产品，以及用于建立手术环境、实现术中止血的手术支持类系列产品，"
            "同时涵盖轻度痔病治疗与泌尿术后康复护理类产品；产品分「有源」与「无源」两大类："
            "有源产品主要包括微创腔镜类、手术机器人类及末端执行器类等，"
            "无源产品涵盖微创腔镜类、开放手术类及其它辅助类产品等。"
        ),
        "gold_chunk_ids": ["688013-0012"],
        "expected_pages": "688013: p9",
        "quote": "报告期内，公司核心业务持续聚焦外科手术关键环节（包括病变组织分离、切除及器官功能"
                 "重建等）所需的多类吻合器产品，以及用于建立手术环境、实现术中止血的手术支持类系列"
                 "产品，同时涵盖轻度痔病治疗与泌尿术后康复护理类产品。",
        "note": "第三节 > 一、报告期内公司所属行业及主营业务情况说明 > "
                "（一）主要业务、主要产品或服务情况。"
                "要点是「吻合器 + 手术支持类 + 痔病/泌尿康复」三块，加上有源/无源分类；"
                "只答「吻合器」是不全的。",
    },
    {
        "qid": "R08",
        "topic": "风险因素",
        "scope": "单公司",
        "question": "惠泰医疗 2026 年半年报提示的「毛利率波动风险」主要受哪些因素影响？",
        "target_company": "惠泰医疗(688617)",
        "company": ["688617"],
        "expected_answer": (
            "公司毛利率水平主要受集采政策、产品销售结构、市场竞争格局等多重因素综合影响；"
            "随着集采政策持续深化推进，相关产品终端价格存在下行压力；未来若出现高毛利产品"
            "销售占比降低、市场竞争进一步加剧等情况，公司可能面临毛利率下滑、整体盈利能力"
            "减弱的风险。"
        ),
        "gold_chunk_ids": ["688617-0033"],
        "expected_pages": "688617: p27",
        "quote": "4、毛利率波动风险\n公司毛利率水平主要受集采政策、产品销售结构、市场竞争格局等"
                 "多重因素综合影响。",
        "note": "第三节 > 四、风险因素（text 块）。同一章还有长期股权投资减值、应收账款坏账、"
                "行业监管、产品注册等条目，答成「集采」三条以外的内容即为串题。",
    },
    {
        "qid": "R09",
        "topic": "两家公司财务指标比较",
        "scope": "跨公司",
        "question": "安图生物和万东医疗 2026 年上半年归属于上市公司股东的净利润分别是多少？"
                    "哪家同比下降幅度更大？",
        "target_company": "安图生物(603658)、万东医疗(600055)",
        "company": ["603658", "600055"],
        "expected_answer": (
            "安图生物 497,124,027.29 元（上年同期 571,001,185.83 元，同比下降 12.94%）；"
            "万东医疗 -91,667,605.80 元（上年同期 51,298,644.68 元，同比下降 278.69%，"
            "由盈利转为亏损）。万东医疗下降幅度更大。"
        ),
        "gold_chunk_ids": ["603658-0008", "600055-0007"],
        "expected_pages": "603658: p6-7; 600055: p5",
        "quote": "| 归属于上市公司股东的净利润 | 497,124,027.29 | 571,001,185.83 | -12.94 |\n"
                 "| 归属于上市公司股东的净利润 | -91,667,605.80 | 51,298,644.68 | -278.69 |",
        "retrieval_note": (
            "实测机理＝**融合截断**：两家 gold 都在候选池内（BM25：603658-0008 #16、"
            "600055-0007 #12；向量：600055-0007 #5），但融合 Top-10 只保住了万东一家，"
            "缺的是安图那张表（向量 #21，境外一名）。两家同时进 Top-10 需要 2 个名额，"
            "本题的失败是「名额被更泛的财务报告块占掉」，不是「表没被找到」。"
        ),
        "note": "两家各自的第二节主要会计数据表。★万东医疗本报告期为**负数**（亏损），"
                "答案是「哪家降幅更大」而不是「哪家数字更小」——万东绝对值 9,166.76 万远小于"
                "安图的 4.97 亿，但降幅（-278.69%）远大于安图（-12.94%）。"
                "只比大小会答反。",
    },
    {
        "qid": "R10",
        "topic": "多家公司综合比较",
        "scope": "跨公司",
        "question": "请将拱东医疗、爱博医疗、圣湘生物、祥生医疗、三友医疗 5 家 2026 年上半年"
                    "营业收入从高到低排序，并给出最高与最低者的金额。",
        "target_company": "拱东医疗(605369)、爱博医疗(688050)、圣湘生物(688289)、"
                          "祥生医疗(688358)、三友医疗(688085)",
        "company": ["605369", "688050", "688289", "688358", "688085"],
        "expected_answer": (
            "从高到低：爱博医疗 932,766,774.39 元 > 圣湘生物 834,318,517.89 元 > "
            "拱东医疗 58,802.29 万元 > 三友医疗 274,020,799.16 元 > "
            "祥生医疗 241,448,492.64 元。最高为爱博医疗（932,766,774.39 元），"
            "最低为祥生医疗（241,448,492.64 元）。"
            "注意拱东医疗该表标注「单位：万元币种：人民币」，其余四家为元。"
        ),
        "gold_chunk_ids": ["688050-0007", "688289-0008", "605369-0009",
                           "688085-0008", "688358-0008"],
        "expected_pages": "688050: p6-7; 688289: p8; 605369: p5-6; 688085: p7-8; 688358: p8",
        "quote": "| 营业收入 | 932,766,774.39 | 786,691,059.79 | 18.57 |\n"
                 "| 营业收入 | 834,318,517.89 | 868,620,485.26 | -3.95 |\n"
                 "单位：万元币种：人民币\n| 营业收入 | 58,802.29 | 55,692.28 | 5.58 |\n"
                 "| 营业收入 | 274,020,799.16 | 249,760,157.28 | 9.71 |\n"
                 "| 营业收入 | 241,448,492.64 | 236,154,283.59 | 2.24 |",
        "retrieval_note": (
            "实测机理＝**候选池深度不够**（与上面两道不同）：5 张表里只有 2 张进了 k=20 池"
            "（605369-0009 向量 #14、688358-0008 向量 #2），另 3 张池外 —— 688050-0007 "
            "向量 #230 / BM25 #703、688289-0008 向量 #132 / BM25 全库 0 命中、688085-0008 "
            "向量 #292 / BM25 全库 0 命中。5 家 × 同一指标要**同时**挤进 20 条候选，"
            "这是相似度检索的结构性短板，也正是「跨公司条件筛选增强」存在的理由 —— "
            "但那条增强只对「要一个集合且没点名公司」的问题生效；本题点名了 5 家，走不到。"
        ),
        "note": "★5 张表分散在 4 个页码段，且拱东医疗那家**单位是万元**（全库唯一），"
                "58,802.29 万元 = 588,022,900 元 —— 把万元当元会把它排到最后一名，"
                "正确却应排第三。排序是**推导**（依据上面 5 个披露值），"
                "5 个数字本身均为披露值。"
                "另注：688050 公司全称「爱博诺德(北京)医疗科技股份有限公司」，"
                "证券简称「爱博医疗」，题面与答案统一用证券简称。",
    },
]


# ---------------------------------------------------------------------------
# 校验 1+2：gold 存在、数字逐字可查
# ---------------------------------------------------------------------------
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
YEARISH = {"2024", "2025", "2026"}


def norm(s: str) -> str:
    """去掉千分位逗号与空白，让 `1,234.5` 与 `1234.5` 可比。"""
    return re.sub(r"[\s,]", "", s)


def numeric_tokens(answer: str) -> list[str]:
    """抽出需要核验的数字 token。

    跳过年份（2024/2025/2026，它们出现在正文里是时间不是指标）和 1~3 位的纯整数
    （「6 家」「5 家」这类计数，出现在正文与否都说明不了什么）。
    """
    out = []
    for m in NUM.finditer(answer):
        t = m.group(0)
        if t in YEARISH:
            continue
        if "." not in t and len(t) < 4:
            continue
        out.append(t)
    return out


def verify(st, q: dict) -> list[str]:
    """返回问题列表；空列表 = 通过。"""
    bad: list[str] = []
    by_id = {cid: st.record(row) for row, cid in enumerate(st.df["chunk_id"])}
    missing = [c for c in q["gold_chunk_ids"] if c not in by_id]
    if missing:
        bad.append(f"gold_chunk_ids 不存在：{missing}")
        return bad

    body = norm("".join(by_id[c]["content"] for c in q["gold_chunk_ids"]))
    for tok in numeric_tokens(q["expected_answer"]):
        if norm(tok) not in body:
            bad.append(f"数字 {tok} 在 gold 原文里找不到（疑似凭记忆写的）")

    # quote 也必须逐字可查 —— 它是「答案依据」栏，写错了比答案错更误导
    for line in q["quote"].splitlines():
        if line.strip() and norm(line)[:16] not in body and line.strip() not in body:
            # 表格行可能被 PDF 抽取拆行，故退一步只查该行最长的数字 token
            toks = [t for t in numeric_tokens(line)]
            if toks and not all(norm(t) in body for t in toks):
                bad.append(f"quote 摘录核对失败：{line.strip()[:40]}…")
    return bad


# ---------------------------------------------------------------------------
# 校验 3：检索自检（不改题，只记录）
# ---------------------------------------------------------------------------
def retrieval_check(st, q: dict, top_k: int) -> dict:
    from src.retrieval.retriever import retrieve

    res = retrieve(q["question"], routes=("bm25", "vector"), top_k=top_k,
                   k_bm25=20, k_vector=20, st=st)
    gold = set(q["gold_chunk_ids"])
    got, first = [], None
    for rank, h in enumerate(res.hits, 1):
        ids = {h.chunk_id, *h.dup_chunk_ids}
        for g in gold & ids:
            if g not in got:
                got.append(g)
            if first is None:
                first = rank
    return {
        "n_gold": len(gold), "n_hit_gold": len(got),
        "all_covered": len(got) == len(gold),
        "first_rank": first,
        "top1": f"{res.hits[0].chunk_id} {st.record(res.hits[0].row)['company'][:12]}"
                if res.hits else "无命中",
        "companies_recognized": [c.security_name for c in res.companies],
    }


# ---------------------------------------------------------------------------
# 写文件
# ---------------------------------------------------------------------------
HEADERS = ["question_id", "question", "question_type", "target_company",
           "expected_answer", "expected_pages"]


def write_xlsx(path: Path, rows: list[dict], checks: dict[str, dict]) -> None:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    wrap = Alignment(wrap_text=True, vertical="top")
    head = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="4472C4")

    def sheet(ws, cols: list[str], data: list[list], widths: list[int]):
        ws.append(cols)
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
        for c in ws[1]:
            c.font, c.fill, c.alignment = head, fill, wrap
        for r in data:
            ws.append(r)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = wrap
        ws.freeze_panes = "A2"

    # sheet1：课程规定的六个字段，一列不多
    ws = wb.active
    ws.title = "questions"
    sheet(ws, HEADERS,
          [[q["qid"], q["question"], f"{q['scope']}-{q['topic']}",
            q["target_company"], q["expected_answer"], q["expected_pages"]]
           for q in rows],
          [10, 44, 20, 30, 52, 22])
    ws.auto_filter.ref = ws.dimensions

    # sheet2：答案依据 —— 每一行都能回到原文
    sheet(wb.create_sheet("标准答案依据"),
          ["question_id", "gold_chunk_ids", "报告原文摘录（逐字）", "出处页码", "备注 / 口径"],
          [[q["qid"], "、".join(q["gold_chunk_ids"]), q["quote"],
            q["expected_pages"], q["note"]] for q in rows],
          [10, 30, 62, 24, 62])

    # sheet3：检索自检 —— 现在这套索引能不能召回，与题目质量无关，仅供定位
    ws3 = wb.create_sheet("检索自检（附加）")
    sheet(ws3,
          ["question_id", "gold 块数", "Top-10 覆盖", "是否全中", "首个 gold 名次",
           "Top-1", "识别到的公司", "失败机理 / 备注（实测）"],
          [[q["qid"], checks[q["qid"]]["n_gold"],
            f"{checks[q['qid']]['n_hit_gold']}/{checks[q['qid']]['n_gold']}",
            "是" if checks[q["qid"]]["all_covered"] else "否",
            checks[q["qid"]]["first_rank"] or "未命中",
            checks[q["qid"]]["top1"],
            "、".join(checks[q["qid"]]["companies_recognized"]) or "无",
            q.get("retrieval_note", "")]
           for q in rows],
          [10, 10, 12, 10, 14, 34, 24, 70])
    note = ws3.cell(row=len(rows) + 3, column=1,
                    value="口径：「Top-10 覆盖 / 首个 gold 名次」＝表头增强索引（index/header）"
                          "· BM25+向量混合 · 默认软加权 · Top-10，与 README §12 同口径。"
                          "「失败机理」里的分路原始名次是另跑的（weights=NO_WEIGHTS，"
                          "k_bm25=k_vector=20），刻意关掉软加权以取**原始排名** —— 开了加权，"
                          "加权本身就会把名次改掉，分不清是融合截断还是加权截断。"
                          "实测日期 2026-09-28。"
                          "★这一栏只是「本系统现在捞不捞得到」，不是题目质量；"
                          "题目只由「半年报里有没有明确答案」决定。")
    note.alignment = wrap
    note.font = Font(italic=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    # 写临时文件 → 回读断言 → 再覆盖（项目里所有产物一律照此）。
    # 为什么这里非做不可：xlsx 被 Excel 打开时 `wb.save()` 会抛 PermissionError，
    # 而**上一步已经跑完了两分钟的检索自检**；更要命的是如果只捕获异常、
    # 不校验内容，磁盘上会静静留着上一版 —— 改过的编号没写进去却看不出来。
    # 临时名必须以 .xlsx 结尾：openpyxl 读文件时**只看后缀**，`questions.xlsx.tmp`
    # 会被 `load_workbook` 直接拒掉（InvalidFileException: does not support .tmp）。
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    wb.save(tmp)
    back = load_workbook(tmp)
    got = [r[0] for r in back["questions"].iter_rows(min_row=2, values_only=True)]
    assert back.sheetnames == ["questions", "标准答案依据", "检索自检（附加）"], back.sheetnames
    assert got == [q["qid"] for q in rows], f"回读的编号 {got} ≠ 预期"
    assert back["questions"].max_column == len(HEADERS), back["questions"].max_column
    back.close()
    try:
        tmp.replace(path)
    except PermissionError:
        print(f"\n[写不进去] {path} 正被占用（多半是 Excel 还开着）。\n"
              f"           内容已生成并校验通过，暂存在 {tmp}。\n"
              f"           关掉 Excel 后重跑本脚本即可（自检会重跑，约 2 分钟）。")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 evaluation/questions.xlsx")
    ap.add_argument("--no-retrieval", action="store_true", help="跳过检索自检")
    ap.add_argument("--index-dir", default="index/header",
                    help="自检用的索引（默认表头增强索引）")
    args = ap.parse_args()

    # 开跑前先探锁：目标被 Excel 打开时，`wb.save()` 要到两分钟检索自检**之后**
    # 才会失败（实测踩过一次，结果磁盘上留着旧版而日志里只有一行 traceback）。
    xlsx = ROOT / "evaluation" / "questions.xlsx"
    if xlsx.exists():
        try:
            with xlsx.open("r+b"):
                pass
        except PermissionError:
            print(f"[开跑前检查] {xlsx.name} 正被占用（多半是 Excel 开着）。\n"
                  f"             先关掉再重跑，否则检索自检白跑 —— 最后一步会写不进去。")
            return 1

    from src.retrieval.store import store
    st = store(args.index_dir)
    print(st.describe())

    # ---- 校验 ----
    fails = 0
    for q in QUESTIONS:
        bad = verify(st, q)
        mark = "[OK]  " if not bad else "[FAIL]"
        print(f"{mark} {q['qid']} {q['scope']}-{q['topic']}  "
              f"gold={len(q['gold_chunk_ids'])} "
              f"数字={len(numeric_tokens(q['expected_answer']))}")
        for b in bad:
            print(f"        ! {b}")
        fails += bool(bad)
    if fails:
        print(f"\n{fails} 道题未通过核对 —— 拒绝写文件。"
              f"（标准答案只允许来自半年报原文，见本文件 docstring）")
        return 1

    # ---- 检索自检 ----
    checks: dict[str, dict] = {}
    for q in QUESTIONS:
        if args.no_retrieval:
            checks[q["qid"]] = {"n_gold": len(q["gold_chunk_ids"]), "n_hit_gold": 0,
                                "all_covered": False, "first_rank": None, "top1": "未自检",
                                "companies_recognized": []}
            continue
        checks[q["qid"]] = retrieval_check(st, q, top_k=10)
        c = checks[q["qid"]]
        print(f"      {q['qid']} 检索自检：gold {c['n_hit_gold']}/{c['n_gold']}，"
              f"首个名次 {c['first_rank'] or '未命中'}")
    if not args.no_retrieval:
        n_ok = sum(c["all_covered"] for c in checks.values())
        print(f"      自检汇总：10 题里 {n_ok} 题的 gold 全部落在 Top-10。"
              f"（这条只说明当前索引的召回情况，不代表题目好坏 —— 题目只由"
              f"「半年报里有没有明确答案」决定。）")

    rows = sorted(QUESTIONS, key=lambda q: q["qid"])
    if not write_xlsx(xlsx, rows, checks):
        return 1

    # 机器可读的孪生文件：字段与 qa.jsonl 一致，eval_retrieval.py 可直接吃
    jl = ROOT / "evaluation" / "datasets" / "questions.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    tmp = jl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for q in rows:
            f.write(json.dumps({
                "qid": q["qid"], "type": q["topic"],
                "question": q["question"], "company": q["company"],
                "gold_chunk_ids": q["gold_chunk_ids"],
                "gold_answer": q["expected_answer"],
                "note": q["note"], "split": "questions_xlsx",
            }, ensure_ascii=False) + "\n")
    # 写临时文件 → 回读断言 → 再覆盖（项目里其它产物一律照此，避免半截文件）
    back = [json.loads(x) for x in tmp.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(back) == len(rows), f"回读 {len(back)} 行 ≠ {len(rows)} 行"
    assert all(b["gold_chunk_ids"] for b in back)
    tmp.replace(jl)

    print(f"\n已写出 {xlsx}（3 个 sheet）与 {jl}（{len(back)} 题）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
