"""把 R01–R10 的评测结果写成一份 PDF 结论（哪类题答得好 / 哪类题翻车 / 为什么）。

## 为什么单独写一份 PDF，而不复用 README §13.7

README 是**过程记录**（怎么出题、怎么核对、怎么复现），读者是维护者；
这份 PDF 是**结论**，读者是只想知道「这套系统在什么题上靠得住、什么题上靠不住」的人。
两边的取舍不同：PDF 里不放全部 10 篇回答原文（那在 `results.xlsx` 里），
但要把**机理**讲清楚 —— 尤其是「答错的 3 题没有一道是模型答错」这件事，
需要把逐路召回名次摆出来才算证据，而不是一句「召回不好」。

## 分析用的事实全部来自实测，不是推测

* 每题每条 gold 在 **BM25 / 向量各自 Top-20 里的名次**（`generation_questions_R01-R10.json`）
* gold 是否进了**最终 Top-6 证据**（同文件 `gold_covered_in_evidence`）
* 判定与错误类型（`review_questions_R01-R10.json`，人判）

三者交叉后得到本文最强的一条结论：**gold 进没进最终证据，与答对与否 10/10 对应**。

中文字体走 Windows 自带的 `msyh.ttc`（微软雅黑），reportlab 直接支持 TTC。
段落样式必须设 `wordWrap='CJK'`，否则中文不换行。

用法：
    python scripts/build_conclusion_pdf.py
    python scripts/build_conclusion_pdf.py --out evaluation/results/结论.pdf
"""

from __future__ import annotations

import argparse
import re
import json
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
RESULTS = EVAL / "results"
DEFAULT_OUT = EVAL / "R01-R10_评测结论.pdf"
GEN = RESULTS / "generation_questions_R01-R10.json"
REVIEW = RESULTS / "review_questions_R01-R10.json"
DS = EVAL / "datasets" / "questions.jsonl"

# 用户给定的错误类型表（errors.xlsx 用的同一份）
ERROR_TYPES = [
    ("E1", "错误公司"), ("E2", "错误页面"), ("E3", "文本切块错误"),
    ("E4", "表格解析错误"), ("E5", "跨页召回失败"), ("E6", "跨公司召回不足"),
    ("E7", "模型理解错误"), ("E8", "数字计算错误"), ("E9", "引用错误"),
    ("E10", "其他"),
]
ET_NAME = dict(ERROR_TYPES)


# ---------------------------------------------------------------- 字体 / 样式

def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MSYH", "C:/Windows/Fonts/msyh.ttc",
                                   subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MSYH-Bold", "C:/Windows/Fonts/msyhbd.ttc",
                                   subfontIndex=0))
    pdfmetrics.registerFontFamily("MSYH", normal="MSYH", bold="MSYH-Bold",
                                  italic="MSYH", boldItalic="MSYH-Bold")


def styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="MSYH", wordWrap="CJK", alignment=TA_LEFT)

    def st(name: str, **kw) -> ParagraphStyle:
        d = dict(base)
        d.update(kw)
        return ParagraphStyle(name, **d)

    return {
        "title": st("title", fontSize=20, leading=28, spaceAfter=4,
                    fontName="MSYH-Bold"),
        "sub": st("sub", fontSize=9.5, leading=14,
                  textColor=colors.HexColor("#555555"), spaceAfter=14),
        "h1": st("h1", fontSize=14, leading=20, fontName="MSYH-Bold",
                 keepWithNext=True, spaceBefore=16, spaceAfter=7,
                 textColor=colors.HexColor("#1a3f6b")),
        "h2": st("h2", fontSize=11.5, leading=17, fontName="MSYH-Bold",
                 keepWithNext=True, spaceBefore=10, spaceAfter=5),
        "body": st("body", fontSize=10, leading=16.5, spaceAfter=6),
        "bullet": st("bullet", fontSize=10, leading=16.5, leftIndent=14,
                     bulletIndent=3, spaceAfter=3),
        "quote": st("quote", fontSize=10, leading=16.5, leftIndent=12,
                    rightIndent=8, spaceBefore=4, spaceAfter=8,
                    textColor=colors.HexColor("#333333")),
        "cell": st("cell", fontSize=8.6, leading=12.6),
        "cellb": st("cellb", fontSize=8.6, leading=12.6, fontName="MSYH-Bold"),
        "note": st("note", fontSize=8.5, leading=13,
                   textColor=colors.HexColor("#666666")),
    }


S: dict[str, ParagraphStyle] = {}


TICK = re.compile(r"`([^`]+)`")


def _detick(m: re.Match) -> str:
    # 用函数式替换而不是 `\1` 反引用：拼脚本时 `\1` 被吞过一次，教训记在这里。
    return '<font color="#1a5a8a">' + m.group(1) + "</font>"


def P(text: str, st: str = "body") -> Paragraph:
    """反引号在 PDF 里没有 Markdown 语义，转成着色以保持「这是代码/路径」的提示。"""
    return Paragraph(TICK.sub(_detick, text), S[st])


def C(text: str, strong: bool = False) -> Paragraph:
    return Paragraph(str(text), S["cellb" if strong else "cell"])


# ---------------------------------------------------------------- 数据

def load() -> tuple[list[dict], dict, dict]:
    recs = json.loads(GEN.read_text(encoding="utf-8"))["records"]
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    ds = {}
    for line in DS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            ds[d["qid"]] = d
    return recs, review, ds


def rank_of(ids: list[str], g: str) -> str:
    return f"#{ids.index(g) + 1}" if g in ids else "池外"


def gold_rows(recs: list[dict]) -> list[dict]:
    out = []
    for r in recs:
        bm = r["retrieved_per_route"]["bm25"]
        ve = r["retrieved_per_route"]["vector"]
        ev = set(r["gold_covered_in_evidence"])
        out.append({
            "qid": r["qid"],
            "golds": [{"id": g, "bm25": rank_of(bm, g), "vec": rank_of(ve, g),
                       "in_ev": g in ev} for g in r["gold_chunk_ids"]],
            "n_in": len(ev), "n_all": len(r["gold_chunk_ids"]),
        })
    return out


# ---------------------------------------------------------------- 版面部件

TBL_STYLE = TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), "MSYH"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.6),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f3")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12335c")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c6d6")),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
])


def table(rows: list[list], widths: list[float], extra: list | None = None) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TBL_STYLE)
    if extra:
        t.setStyle(TableStyle(extra))
    return t


# ---------------------------------------------------------------- 正文

def build(recs: list[dict], review: dict, ds: dict) -> list:
    f: list = []
    wrong = [r["qid"] for r in recs if review[r["qid"]].get("correct") is False]
    right = [r["qid"] for r in recs if review[r["qid"]].get("correct") is True]
    et_count: dict[str, list[str]] = {}
    for qid in wrong:
        et = review[qid]["error_type"]
        et_count.setdefault(et, []).append(qid)
    golds = {g["qid"]: g for g in gold_rows(recs)}
    full = [g["qid"] for g in golds.values() if g["n_in"] == g["n_all"]]
    partial = [g["qid"] for g in golds.values() if g["n_in"] < g["n_all"]]

    # ---- 封面 ----
    f.append(P("上交所医疗器械财报 RAG：R01–R10 评测结论", "title"))
    f.append(P(f"17 份 2026 年半年度报告（4657 chunk）｜表头增强索引 <b>index/header</b>｜"
               f"混合召回（BM25 + FAISS）→ RRF 融合 → Top-6 证据 → <b>deepseek-chat</b>"
               f"（temperature 0.1）"
               f"<br/>出报告日期：{datetime.now():%Y-%m-%d}｜"
               f"判分为人判（`review_questions_R01-R10.json`），非脚本计算", "sub"))

    # ---- 1 一句话结论 ----
    f.append(P("一、一句话结论", "h1"))
    f.append(P(
        "10 道题答对 <b>7 道（70.0%）</b>：单公司题 7/8（87.5%）、跨公司题 <b>0/2（0%）</b>。"
        "把逐路召回名次与判定结果交叉后，得到一条没有例外的规律 —— "
        "<b>gold 块进了最终 Top-6 证据的 7 道题，7 道全对；gold 块掉出 Top-6 的 3 道题，"
        "3 道全错</b>。3 道错题里没有一道是模型答错：在证据够的题上它答得准，"
        "在证据缺的题上它明确说「证据里没有」而没有编造。", "quote"))
    f.append(P("因此这份结论的核心不是「模型行不行」，而是<b>召回在什么题型上会漏，以及为什么漏</b>。",
               "body"))

    # ---- 2 总览 ----
    f.append(P("二、总览", "h1"))
    f.append(table([
        [C("指标", True), C("结果", True), C("说明", True)],
        [C("总准确率"), C("<b>7/10 = 70.0%</b>"),
         C("已判定 10 题，无「未判」")],
        [C("单公司题（R01–R08）"), C("<b>7/8 = 87.5%</b>"),
         C("错的是 R05（经营现金流）")],
        [C("跨公司题（R09–R10）"), C("<b>0/2 = 0.0%</b>"),
         C("R09 两家比、R10 五家比，均未答全")],
        [C("引用越界编号"), C("0 次"), C("没有一处引用不存在的证据编号")],
        [C("凭空数字"), C("0 处"),
         C("回答里的数字除股票代码与 3 处自算差值外，全部逐字出自证据")],
    ], [4.0 * cm, 3.6 * cm, 9.4 * cm]))

    f.append(P("错误类型分布（用户给定的 E1–E10）", "h2"))
    rows = [[C("错误类型", True), C("名称", True), C("数量", True), C("涉及题目", True)]]
    for code, name in ERROR_TYPES:
        qs = et_count.get(code, [])
        rows.append([C(code, bool(qs)), C(name, bool(qs)),
                     C(str(len(qs)), bool(qs)), C("、".join(qs) or "—", bool(qs))])
    rows.append([C("合计", True), C("", True), C(str(len(wrong)), True),
                 C("、".join(wrong), True)])
    f.append(table(rows, [2.2 * cm, 4.2 * cm, 1.8 * cm, 5.8 * cm]))
    f.append(Spacer(1, 5))
    f.append(P(
        "E1–E5、E7–E9 <b>八类一个都没出现</b>：没有认错公司、没有引错页码、"
        "没有切块或表格解析缺陷、没有模型理解错、没有算错数、没有编造引用。"
        "错题全部集中在召回侧。", "note"))

    # ---- 3 答得好的 ----
    f.append(P("三、哪类题答得好", "h1"))

    f.append(P("3.1 定性叙述题：3/3，全对，且答得很完整", "h2"))
    f.append(P(
        "R06（收入增量来自哪里）、R07（核心业务与主要产品）、R08（毛利率波动风险受哪些因素影响）"
        "三题全对。这三题的共同点是<b>答案集中在一个正文块里</b>，而且那个块"
        "<b>本身含有公司名和密集的主题词</b>：", "body"))
    f.append(table([
        [C("题", True), C("gold 块", True), C("BM25 名次", True),
         C("向量名次", True), C("作答要点", True)],
        [C("R06"), C("688271-0038"), C("#1"), C("#2"),
         C("「海外业务持续放量 + 国内份额逆势提升」逐字答中，并补充国内收入 52.87 亿元/+8.49%、份额 +1.7 个百分点")],
        [C("R07"), C("688013-0012"), C("#6"), C("池外"),
         C("核心业务三句 + 有源/无源两大类及各自子类全中")],
        [C("R08"), C("688617-0033"), C("#1"), C("#1"),
         C("集采政策、产品销售结构、市场竞争格局三项全中，并给出后果表述")],
    ], [1.2 * cm, 2.4 * cm, 1.9 * cm, 1.9 * cm, 9.6 * cm]))
    f.append(Spacer(1, 4))
    f.append(P(
        "值得注意 R07：它的 gold 在<b>向量路完全没进 Top-20</b>，只靠 BM25 第 6 名进来 —— "
        "这说明两路召回在这里是互补的，任一路单独漏掉都不致命。", "note"))

    f.append(P("3.2 单点数值题：4/5，靠「同一数字在别处还有一份」救回来的", "h2"))
    f.append(P(
        "R01（营收）、R02（归母净利润）、R03（毛利率）、R04（研发投入）全对。"
        "其中 R01、R02、R04 的 gold 块<b>本身也没被融合保进 Top-6</b>（R01 的 gold 在 BM25 第 20 名、"
        "向量第 10 名；R02 在第 9/第 6 名），但它们仍然答对了 —— 因为<b>同一个数字在半年报里"
        "有冗余表述</b>：主要会计数据表里的营收/净利润，在第八节合并利润表、"
        "第三节管理层讨论与分析里还会再出现一次，模型从那些块里取到了同值。", "body"))
    f.append(P(
        "R04 是这一类的正面样板：两条 gold（研发投入情况表 688351-0024、主要财务指标表 688351-0008）"
        "<b>双双进了证据并都被引用</b>，金额、占比、同比三个数字全中，还主动区分了母公司口径。", "body"))

    f.append(P("3.3 一句话概括「好」的条件", "h2"))
    f.append(P(
        "答案落在<b>含公司名的正文块</b>里，或者答案数字在语料里有<b>冗余副本</b> —— "
        "满足其一就稳。反之（答案只存在于某一张不含公司名的表格里、且无副本）就会翻车，"
        "见下一节。", "quote"))

    # ---- 4 翻车的 ----
    f.append(P("四、哪类题翻车", "h1"))

    f.append(P("4.1 跨公司题：0/2 —— 全军覆没", "h2"))
    f.append(P(
        "R09（安图生物 vs 万东医疗）与 R10（拱东 / 爱博 / 圣湘 / 祥生 / 三友 五家排序）"
        "都要求<b>多家公司的同一指标同时出现在证据里</b>，而最终证据只有 Top-6 条。"
        "两题各答对一部分：", "body"))
    f.append(table([
        [C("题", True), C("答对了什么", True), C("缺了什么", True)],
        [C("R09"),
         C("万东医疗 −91,667,605.80 元、上年同期 51,298,644.68 元、同比 −278.69%，全对"),
         C("安图生物归母净利润（gold 603658-0008：BM25 第 16 名、向量池外，融合后被挤出 Top-6）")],
        [C("R10"),
         C("拱东医疗 588,022,932.18 元（<b>没踩万元陷阱</b>）、祥生医疗 241,448,492.64 元"),
         C("爱博医疗、圣湘生物、三友医疗三家<b>两路全在池外</b>（688289-0008 与 688085-0008 的 BM25 全库 0 命中）")],
    ], [1.2 * cm, 7.3 * cm, 8.5 * cm]))
    f.append(Spacer(1, 4))
    f.append(P(
        "R10 的模型行为值得单独肯定：证据只够 2/5，它<b>没有硬排一个顺序</b>，"
        "而是列出「缺少爱博医疗、圣湘生物、三友医疗三家公司的任何营业收入数据」，"
        "并给出已确认的两家数字后拒答。按「题目问到的都要答到」的严格口径这题记错，"
        "但它的拒答是对的。", "note"))

    f.append(P("4.2 单公司题里唯一翻车的：R05（经营现金流）", "h2"))
    f.append(P(
        "这题的特殊之处是<b>数对了一半</b>：经营活动现金流量净额 160,232,989.24 元、"
        "上年同期 33,854,199.97 元<b>完全正确</b> —— 模型从「现金流量表补充资料」"
        "（603309-0209）取到了与 gold 同值的数字。缺的是<b>公司披露的同比 +373.30%</b>："
        "模型说「证据中未直接给出同比变化」，改为自算增减额 126,378,789.27 元"
        "（减法正确）。", "body"))
    f.append(P(
        "为什么 R01/R02 靠冗余救回来了、R05 没有？因为<b>冗余救不了「只在一处存在」的信息</b>。"
        "净额本身在财务报告里有第二份，但那个 +373.30% 的同比比例<b>只写在 "
        "603309-0008（第二节 主要会计数据表）那一张表里</b>，而它没进证据。", "body"))
    f.append(P(
        "判定边界已如实记录：若按宽松口径把「自算的绝对增减额」也算作「同比变化」，"
        "本题可判对 —— 那样单公司就是 8/8、总准确率 80%。本报告与 `errors.xlsx` 采用与"
        "旧套一致的严格口径，并在原因栏写明。", "note"))

    # ---- 5 为什么 ----
    f.append(P("五、为什么：两种机理，一个根因", "h1"))

    f.append(P("5.1 铁律：gold 进没进最终证据，与对错 10/10 对应", "h2"))
    f.append(P("逐题把「每条 gold 在 BM25 / 向量各自 Top-20 里的名次」量出来，与判定对照：",
               "body"))
    rows = [[C("题", True), C("gold 块", True), C("BM25", True), C("向量", True),
             C("进最终证据", True), C("判定", True)]]
    for r in recs:
        qid = r["qid"]
        g = golds[qid]
        ok = review[qid].get("correct")
        v = "√ 答对" if ok else "× 答错"
        for i, item in enumerate(g["golds"]):
            rows.append([
                C(qid if i == 0 else ""), C(item["id"]),
                C(item["bm25"]), C(item["vec"]),
                C("是" if item["in_ev"] else "<b>否</b>"),
                C(v if i == 0 else ""),
            ])
    f.append(table(rows, [1.3 * cm, 2.7 * cm, 1.7 * cm, 1.7 * cm, 2.2 * cm, 2.0 * cm],
                   extra=[("BACKGROUND", (0, 1), (-1, -1), colors.white)]))
    f.append(Spacer(1, 5))
    f.append(P(
        f"把「gold 有缺口」的题挑出来：<b>{'、'.join(partial)}</b> —— 正好就是答错的 "
        f"3 题；「gold 全覆盖」的 <b>{'、'.join(full)}</b> 7 题则全部答对。"
        "这不是相关，是<b>一一对应</b>：召回决定对错，生成层在这一轮里没有失手。", "quote"))

    f.append(P("5.2 机理一：融合截断（R05、R09）—— 池子里有，但被挤出 Top-6", "h2"))
    f.append(P(
        "R05 的 gold 在<b>向量路第 10 名</b>、R09 安图那一条在 <b>BM25 第 16 名</b>，"
        "两条都<b>已经进了 k=20 的候选池</b>，是 RRF 融合之后的排序把它们挤出了 Top-6。"
        "这意味着<b>单纯调大召回深度救不回来</b> —— 池子再深，只要融合排序把它们排在 6 名之后就没有用。"
        "这个机理在旧评测集里已经对旗舰题 Q01 实测证伪过一次（「调大深度就能救」），"
        "这里是同一机理在新题面上的复现。", "body"))

    f.append(P("5.3 机理二：候选池深度不足（R10）—— 从头就没进池子", "h2"))
    f.append(P(
        "R10 的 5 张表里，只有祥生医疗那张进了池（向量第 2 名），拱东那张进到向量第 14 名但没保住；"
        "另外三家（爱博 688050-0007、圣湘 688289-0008、三友 688085-0008）"
        "<b>两路都在 Top-20 之外</b>，其中圣湘与三友那两张表的 <b>BM25 在全库 4657 条里 0 命中</b>。"
        "这一类和 R05/R09 不同：不是排序问题，是<b>根本没被检索到</b>。", "body"))

    f.append(P("5.4 共同根因：表格块里没有公司名", "h2"))
    f.append(P(
        "三种失败指向同一个事实：<b>17 家公司的「主要会计数据」表正文几乎同构，而且表内不含公司名</b>。"
        "问「维力医疗的现金流」，BM25 里分值最高的判别词恰恰是「维力」这个高 IDF 词，"
        "而它出现在正文的散文里、不出现在那张表里 —— 于是同名结构的表格输给了所有提到公司的散文块。"
        "向量侧同理：17 张几乎一样的表，模型无从分辨哪张属于哪家。", "body"))
    f.append(P(
        "R01/R02/R04 之所以没被这个根因击倒，是因为它们的答案数字在语料里有<b>冗余副本</b>"
        "（第八节合并利润表、第三节转述），绕开了那张表；R05 缺的同比比例、"
        "R09/R10 缺的对家数字，都<b>只在那张表里有</b>，绕不开。", "body"))

    f.append(P("5.5 已经有的对症手段，以及它为什么这次没生效", "h2"))
    f.append(P(
        "项目里本来就有一条针对「跨公司取同一张表」的<b>结构化表组路由</b>"
        "（`src/retrieval/structured.py`）：按语料的硬结构确定性地取出 17 家公司各自的"
        "「主要会计数据」表，曾经做到 17/17 家公司覆盖。但它有一个前置判据 —— "
        "<b>只在问题里没有点名公司时才触发</b>（判据见 `structured.py` 顶部）。"
        "R09 点名了 2 家、R10 点名了 5 家，所以这条路<b>两题都没有走</b>，"
        "退回了纯相似度检索。这是一个<b>接口缺口，不是能力缺口</b>。", "body"))

    # ---- 6 逐题一览 ----
    f.append(P("六、逐题结果一览", "h1"))
    rows = [[C("题", True), C("主题", True), C("题型", True),
             C("gold 名次（BM25/向量）", True), C("判定", True), C("错误类型", True)]]
    theme = {"R01": "营业收入", "R02": "净利润", "R03": "毛利率/盈利能力",
             "R04": "研发投入", "R05": "经营现金流", "R06": "主要业务变化",
             "R07": "主要产品", "R08": "风险因素",
             "R09": "两家公司财务指标比较", "R10": "多家公司综合比较"}
    for r in recs:
        qid = r["qid"]
        rv = review[qid]
        g = golds[qid]
        scope = "跨公司" if len(ds[qid]["company"]) > 1 else "单公司"
        rank = "；".join(f"{i['id']} {i['bm25']}/{i['vec']}" for i in g["golds"])
        v = "√ 答对" if rv.get("correct") else "× 答错"
        et = f"{rv['error_type']} {ET_NAME[rv['error_type']]}" if rv.get("error_type") else "—"
        rows.append([C(qid), C(theme[qid]), C(scope), C(rank), C(v), C(et)])
    f.append(table(rows, [1.2 * cm, 3.0 * cm, 1.6 * cm, 5.6 * cm, 1.7 * cm, 3.5 * cm],
                   extra=[("BACKGROUND", (0, 1), (-1, -1), colors.white)]))
    f.append(Spacer(1, 5))
    f.append(P("注：名次是每题 gold 块在 BM25 / 向量各自 Top-20 里的位置，「池外」= 未进 Top-20。"
               "完整回答原文、全部 Top-6 证据原文见 `evaluation/results.xlsx` 与 "
               "`generation_questions_R01-R10.json`。", "note"))

    # ---- 7 改进方向 ----
    f.append(P("七、改进方向（按证据，不按猜测）", "h1"))
    for i, (head, txt) in enumerate([
        ("跨公司比较题接入结构化表组",
         "把「点名了 N 家公司、要横向比同一指标」这一类也接到 `structured.py` 的表组路由上"
         "（按识别出的公司集合去取各自的「主要会计数据」表），而不是退回相似度检索。"
         "这是针对 R09/R10 的直接修法，也是这次失败暴露出的唯一接口缺口。"),
        ("给融合加「每家公司保底一条」",
         "R05、R09 的共同机理是池内 gold 被融合挤出 Top-6。修法不在加深池子，而在融合环节："
         "对识别到的每家公司，从候选池里保底保留其最高分的那条块再排序。"
         "这条对症融合截断，且不改变「不问公司名」的题（那类题识别不到具体公司）。"),
        ("不要换更大的模型",
         "3 道错题没有一道是模型答错：证据够时答得准，证据缺时明确说没有、不编数。"
         "R09 里它甚至主动拒绝用母公司净利润顶替归母净利润。换模型不解决这里的任何一题。"),
    ], 1):
        f.append(P(f"7.{i} {head}", "h2"))
        f.append(P(txt, "body"))

    # ---- 8 口径与产物 ----
    f.append(P("八、口径与产物", "h1"))
    f.append(P(
        "<b>口径</b>：表头增强索引 `index/header`；混合召回 = BM25（jieba）+ FAISS 余弦，"
        "各取 Top-20 候选 → RRF 融合（k=60）→ 公司/章节软加权 → 取 Top-6 证据；"
        "生成 = `deepseek-chat`，temperature 0.1，证据不足须明说。"
        "<b>这是一次采样的结果</b>，同一题换一轮表述会变，不能与旧评测集（另一套 10 题）"
        "的 7/10 相加或平均。", "body"))
    f.append(table([
        [C("文件", True), C("内容", True)],
        [C("evaluation/results.xlsx"),
         C("逐题 11 栏完整记录（含 BM25/向量逐路召回、最终 chunk、引用页码、回答、标准答案、判定、错误类型、错误原因）"
           "＋ 逐题证据明细 60 条 ＋ 口径与汇总")],
        [C("evaluation/errors.xlsx"),
         C("错题台账（含「缺哪块 gold」）＋ E1–E10 类型统计（含数量为 0 的类型）")],
        [C("evaluation/results/review_questions_R01-R10.json"),
         C("逐题判定（人判，两份表的输入）")],
        [C("evaluation/results/generation_questions_R01-R10.json"),
         C("原始记录：全部证据原文与回答原文、逐路召回、token 用量")],
        [C("evaluation/questions.xlsx"),
         C("题目与标准答案（R01–R10，8 单公司 + 2 跨公司）")],
    ], [6.4 * cm, 10.6 * cm]))
    f.append(Spacer(1, 6))
    f.append(P("复现：<font face='MSYH'>python scripts/eval_generation.py --dataset "
               "evaluation/datasets/questions.jsonl --index ./index/header<br/>"
               "python scripts/build_results_xlsx.py<br/>"
               "python scripts/build_conclusion_pdf.py</font>", "note"))
    return f


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("MSYH", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2 * cm, 1.1 * cm,
                      "上交所医疗器械财报 RAG — R01–R10 评测结论")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 R01–R10 评测结论 PDF")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    for p in (out,):
        if p.exists():
            try:
                with p.open("r+b"):
                    pass
            except PermissionError:
                print(f"[开跑前检查] {p.name} 正被占用（多半是 PDF 阅读器开着）。")
                return 1

    register_fonts()
    S.update(styles())
    recs, review, ds = load()
    doc = SimpleDocTemplate(str(tmp), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.7 * cm,
                            title="上交所医疗器械财报 RAG：R01–R10 评测结论",
                            author="RAG 项目评测")
    doc.build(build(recs, review, ds), onFirstPage=footer, onLaterPages=footer)

    # 回读：不靠「没报错」判断成败（教训见 memory: xlsx-save-readback…）。
    # 不能直接在字节里 grep —— TTF 子集化后正文是字形编号，中文和 "R01" 都不以
    # 原文出现；必须走 ToUnicode 映射把文本层抽出来验（reportlab 会写这张表）。
    from pypdf import PdfReader
    reader = PdfReader(str(tmp))
    text = "\n".join(pg.extract_text() or "" for pg in reader.pages)
    missing = [k for k in ("R01", "R10", "70.0", "跨公司", "融合截断",
                           "候选池深度不足", "E6", "R05") if k not in text]
    assert not missing, f"回读不到：{missing}"
    assert len(reader.pages) >= 5, f"页数异常：{len(reader.pages)}"
    print(f"回读通过：{len(reader.pages)} 页，文本层 {len(text):,} 字符")
    out.unlink(missing_ok=True)
    tmp.replace(out)
    print(f"已写出 → {out}（{out.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
