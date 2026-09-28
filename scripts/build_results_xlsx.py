"""把 R01–R10 的逐题记录 + 人判结论，落成两份课程要求的表。

    evaluation/results.xlsx   逐题 11 栏（题号 / 问题 / BM25 召回 / 向量召回 /
                              最终使用 chunk / 引用页码 / 系统回答 / 标准答案 /
                              是否答对 / 错误类型 / 错误原因）
    evaluation/errors.xlsx    错题明细 + E1–E10 各类型数量

## 为什么两份表要分文件

`results.xlsx` 是**逐题的完整留痕**（10 行，含答对的题），`errors.xlsx` 是
**错题台账**（只 3 行，但每行多出「缺哪块 gold」「机理属于哪一类」两栏）。
课程要求的是前者，但调优时看的是后者 —— 混在一张表里，答对的 7 行会把
3 行错题稀释掉。

## 为什么「是否答对」不在这里算

见 `scripts/eval_generation.py` 顶部：财报问答的错法（万元量级、本报告期 vs
上年同期）字符串比对判不出来。判定写在
`evaluation/results/review_questions_R01-R10.json`（人判），本脚本只负责排版。
判定缺失的题一律显示「未判」，**不计入任何比例** —— 宁缺不编。

## 检索结果逐题存三样

题面要求「必须逐题保存检索结果，不要只输出总体准确率」，所以：

* 第 3、4 栏存**每一路各自的 top-N**（BM25 20 条 / 向量 20 条，标出 gold 在第几名）
* 第 5 栏存**融合后最终进证据的块**（哪几条被引用、哪几条被冷落）
* 附加页 `逐题证据明细` 存每题每条证据的页码 / 章节 / 召回路径 / 是否 gold

「BM25 没召回」和「召回进了池子却被融合截断」是两种完全不同的病，只有逐路
才看得出来 —— 这正是旧套 Q01 / 本套 R05 定性的唯一依据。

用法：
    python scripts/build_results_xlsx.py
    python scripts/build_results_xlsx.py --gen evaluation/results/generation_questions_R01-R10.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
RESULTS = EVAL / "results"
DATASET = EVAL / "datasets" / "questions.jsonl"
DEFAULT_GEN = RESULTS / "generation_questions_R01-R10.json"
REVIEW = RESULTS / "review_questions_R01-R10.json"

# 用户给定的错误类型表。**顺序即编码顺序**，统计表要按它全列出来（含 0 的）。
ERROR_TYPES: list[tuple[str, str]] = [
    ("E1", "错误公司"),
    ("E2", "错误页面"),
    ("E3", "文本切块错误"),
    ("E4", "表格解析错误"),
    ("E5", "跨页召回失败"),
    ("E6", "跨公司召回不足"),
    ("E7", "模型理解错误"),
    ("E8", "数字计算错误"),
    ("E9", "引用错误"),
    ("E10", "其他"),
]
ET_NAME = dict(ERROR_TYPES)

HEADERS = ["question_id", "question", "BM25召回结果", "向量召回结果",
           "最终使用chunk", "引用页码", "系统回答", "标准答案",
           "是否答对", "错误类型", "错误原因"]

WRAP = Alignment(wrap_text=True, vertical="top")
TOP = Alignment(vertical="top")
HEAD_FILL = PatternFill("solid", fgColor="DDEBF7")
BAD_FILL = PatternFill("solid", fgColor="FCE4E4")
GOOD_FILL = PatternFill("solid", fgColor="E4F3E4")


# ---------------------------------------------------------------- 读数据

def load_dataset(p: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["qid"]] = d
    return out


def route_cell(ids: list[str], gold: set[str], top: int = 5) -> str:
    """逐路召回栏：top-N + 每条标 gold。没召回 gold 时明说 gold 在池中的名次。"""
    if not ids:
        return "（该路不可用）"
    head = []
    for i, cid in enumerate(ids[:top], 1):
        head.append(f"{i}.{cid}" + ("★gold" if cid in gold else ""))
    s = "、".join(head) + f"（共 {len(ids)} 条）"
    hit = [i for i, cid in enumerate(ids, 1) if cid in gold]
    miss = sorted(gold - set(ids))
    if hit:
        s += f"　→ gold 命中第 {'、'.join(str(i) for i in hit)} 名"
    if miss:
        s += f"　→ 未召回：{'、'.join(miss)}"
    return s


def build_rows(recs: list[dict], ds: dict[str, dict], review: dict) -> list[dict]:
    rows = []
    for r in recs:
        qid = r["qid"]
        item, rv = ds.get(qid, {}), review.get(qid, {})
        gold = set(r["gold_chunk_ids"])
        a = r.get("answer") or {}
        want = set(a.get("citations") or [])
        cited_ev = [e for e in r["evidence"] if e["idx"] in want]
        cited_ids = [e["chunk_id"] for e in cited_ev]
        pages = "；".join(f"{e['chunk_id']} {e['page_label']}" for e in cited_ev)
        et = rv.get("error_type")
        rows.append({
            "qid": qid,
            "scope": "跨公司" if len(item.get("company") or []) > 1 else "单公司",
            "type": r["type"],
            "question": r["question"],
            "bm25": route_cell(r["retrieved_per_route"].get("bm25", []), gold),
            "vector": route_cell(r["retrieved_per_route"].get("vector", []), gold),
            "used": ("、".join(cited_ids) or "（未引用任何证据）"),
            "used_all": "、".join(e["chunk_id"] for e in r["evidence"]),
            "pages": pages or "（无引用）",
            "answer": a.get("text") or (
                f"<生成失败：{a.get('error')}>" if a.get("error") else "（未调用生成）"),
            "gold_answer": item.get("gold_answer", r.get("gold_answer", "")),
            "correct": rv.get("correct"),
            "error_type": et,
            "reason": rv.get("reason", ""),
            "missing_gold": rv.get("missing_gold", []),
            "n_gold_in_evidence": len(r["gold_covered_in_evidence"]),
            "n_gold": len(r["gold_chunk_ids"]),
            "refused": a.get("insufficient"),
            "citations": a.get("citations") or [],
            "invalid_citations": a.get("invalid_citations") or [],
        })
    return rows


# ---------------------------------------------------------------- 排版

def style_header(ws, ncol: int, widths: list[int]) -> None:
    for c in range(1, ncol + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def write_results_xlsx(path: Path, rows: list[dict], meta: dict) -> None:
    from openpyxl import Workbook
    wb = Workbook()

    # ---- 页 1：逐题结果（课程要求的 11 栏，顺序不动）----
    ws = wb.active
    ws.title = "逐题结果"
    ws.append(HEADERS)
    for r in rows:
        verdict = {True: "✅ 答对", False: "❌ 答错", None: "— 未判"}[r["correct"]]
        et = f"{r['error_type']} {ET_NAME.get(r['error_type'], '')}".strip() if r["error_type"] else "—"
        ws.append([r["qid"], r["question"], r["bm25"], r["vector"], r["used"],
                   r["pages"], r["answer"], r["gold_answer"], verdict, et, r["reason"]])
    style_header(ws, len(HEADERS), [11, 46, 54, 54, 26, 30, 78, 62, 11, 16, 78])
    for i, r in enumerate(rows, start=2):
        for c in range(1, len(HEADERS) + 1):
            ws.cell(row=i, column=c).alignment = WRAP
        ws.cell(row=i, column=9).fill = GOOD_FILL if r["correct"] else (
            BAD_FILL if r["correct"] is False else PatternFill())
        ws.row_dimensions[i].height = 150
    ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{len(rows) + 1}"

    # ---- 页 2：逐题证据明细（每题每条证据一行，便于核页码）----
    ws2 = wb.create_sheet("逐题证据明细")
    h2 = ["question_id", "#", "chunk_id", "公司", "章节", "小节", "页码", "类型",
          "字数", "召回路径", "是否引用", "是否gold", "截断"]
    ws2.append(h2)
    for rec, r in zip(meta["_records"], rows):
        gold = set(rec["gold_chunk_ids"])
        cited = set(r["citations"])
        for e in rec["evidence"]:
            ws2.append([r["qid"], e["idx"], e["chunk_id"], e["company"], e["section"],
                        e["subsection"], e["page_label"], e["content_type"],
                        e["char_count"], "/".join(e["routes"]),
                        "✅" if e["idx"] in cited else "",
                        "★" if e["chunk_id"] in gold else "",
                        "是" if e["truncated"] else ""])
    style_header(ws2, len(h2), [11, 4, 14, 26, 22, 24, 11, 8, 7, 12, 9, 9, 6])
    for i in range(2, ws2.max_row + 1):
        for c in range(1, len(h2) + 1):
            ws2.cell(row=i, column=c).alignment = TOP
        if ws2.cell(row=i, column=12).value == "★":
            for c in range(1, len(h2) + 1):
                ws2.cell(row=i, column=c).fill = GOOD_FILL

    # ---- 页 3：口径与汇总 ----
    ws3 = wb.create_sheet("口径与汇总")
    stats = summarize(rows)
    ws3.append(["项", "值"])
    for k, v in meta["_meta"].items():
        ws3.append([k, v])
    ws3.append(["", ""])
    ws3.append(["—— 汇总 ——", ""])
    for k, v in stats["headline"]:
        ws3.append([k, v])
    ws3.append(["", ""])
    ws3.append(["错误类型", "数量", "涉及题号"])
    for code, name in ERROR_TYPES:
        qs = stats["by_type"].get(code, [])
        ws3.append([f"{code} {name}", len(qs), "、".join(qs) or "—"])
    ws3.append(["", ""])
    for line in stats["notes"]:
        ws3.append([line])
    style_header(ws3, 3, [34, 22, 40])
    for i in range(2, ws3.max_row + 1):
        for c in range(1, 4):
            ws3.cell(row=i, column=c).alignment = WRAP

    save_safely(wb, path, [ws, ws2, ws3])


def write_errors_xlsx(path: Path, rows: list[dict], meta: dict) -> None:
    from openpyxl import Workbook
    wb = Workbook()
    wrong = [r for r in rows if r["correct"] is False]

    ws = wb.active
    ws.title = "错题明细"
    h = ["question_id", "类型", "question", "错误类型", "错误类型名称",
         "错误原因", "证据缺口（gold 未进最终证据的块）", "gold 覆盖",
         "系统回答", "标准答案"]
    ws.append(h)
    for r in wrong:
        ws.append([r["qid"], r["scope"], r["question"],
                   r["error_type"], ET_NAME.get(r["error_type"], ""), r["reason"],
                   "、".join(r["missing_gold"]) or "—",
                   f"{r['n_gold_in_evidence']}/{r['n_gold']}",
                   r["answer"], r["gold_answer"]])
    style_header(ws, len(h), [11, 8, 40, 9, 16, 92, 22, 9, 70, 60])
    for i in range(2, ws.max_row + 1):
        for c in range(1, len(h) + 1):
            ws.cell(row=i, column=c).alignment = WRAP
        ws.row_dimensions[i].height = 190

    ws2 = wb.create_sheet("错误类型统计")
    ws2.append(["错误类型", "名称", "数量", "涉及题号"])
    by_type: dict[str, list[str]] = {}
    for r in wrong:
        by_type.setdefault(r["error_type"], []).append(r["qid"])
    for code, name in ERROR_TYPES:
        qs = by_type.get(code, [])
        ws2.append([code, name, len(qs), "、".join(qs) or "—"])
    ws2.append(["合计", "", len(wrong), "、".join(r["qid"] for r in wrong) or "—"])
    style_header(ws2, 4, [10, 20, 8, 26])
    ws2.append([])
    ws2.append(["注：本套错题全部集中在召回侧（E6 跨公司召回不足 2 题、E10 其他 1 题），"
                "没有出现 E1/E2/E3/E4/E5/E7/E8/E9 中的任何一类 —— 即没有认错公司、"
                "没有引错页码、没有切块/表格解析缺陷、没有模型理解错、没有算错数、"
                "没有引用越界编号。"])
    ws2.cell(row=ws2.max_row, column=1).alignment = WRAP
    ws2.append(["注：E10 那一题（R05）的净额与上年同期数字完全正确，缺的只是公司披露的"
                "同比比例 +373.30%；机理是 gold 块在候选池内被 RRF 融合截断，"
                "归不进 E1–E9 任何一类，故记 E10 并在原因栏写明机理。"])

    save_safely(wb, path, [ws, ws2])


def save_safely(wb, path: Path, sheets: list) -> bool:
    """写临时 .xlsx → 回读断言 → 再覆盖。

    临时名必须以 .xlsx 结尾：openpyxl 的 load_workbook **只看后缀**，
    `results.xlsx.tmp` 会被直接拒掉（InvalidFileException）。
    理由见 memory: xlsx-save-readback-and-excel-lock。
    """
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    wb.save(tmp)
    back = load_workbook(tmp)
    assert back.sheetnames == [ws.title for ws in sheets], (back.sheetnames,
                                                            [w.title for w in sheets])
    for ws in sheets:
        assert back[ws.title].max_row == ws.max_row, (ws.title, back[ws.title].max_row,
                                                      ws.max_row)
    back.close()
    try:
        tmp.replace(path)
    except PermissionError:
        print(f"\n[写不进去] {path.name} 正被占用（多半是 Excel 还开着）。"
              f"\n  临时文件留在 {tmp.name}，关掉 Excel 后改名即可。")
        return False
    return True


# ---------------------------------------------------------------- 统计

def summarize(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["correct"] is not None]
    right = [r for r in judged if r["correct"]]
    single = [r for r in judged if r["scope"] == "单公司"]
    cross = [r for r in judged if r["scope"] == "跨公司"]
    single_ok = [r for r in single if r["correct"]]
    cross_ok = [r for r in cross if r["correct"]]
    by_type: dict[str, list[str]] = {}
    for r in rows:
        if r["error_type"]:
            by_type.setdefault(r["error_type"], []).append(r["qid"])

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} = {a / b:.1%}" if b else "—"

    headline = [
        ("题目总数", len(rows)),
        ("已判定", len(judged)),
        ("答对数", len(right)),
        ("总准确率", pct(len(right), len(judged))),
        ("单公司题（R01–R08）", pct(len(single_ok), len(single))),
        ("跨公司题（R09–R10）", pct(len(cross_ok), len(cross))),
        ("答错数", len(judged) - len(right)),
    ]
    notes = [
        "「是否答对」由人判定（review_questions_R01-R10.json），不是脚本算的 —— "
        "「58,802.29 元」与「58,802.29 万元」字符全同而量级差一万倍，自动比对判不出来。",
        "准确率只对**已判定**的题计算；未判定的题不推算、不补齐。",
        f"每题全文（含全部 Top-6 证据原文、逐路召回、token 用量）见 "
        f"{DEFAULT_GEN.name}；本题逐条证据见本文件「逐题证据明细」页。",
        "错误类型分布见 errors.xlsx；本题集未出现的类型也以 0 列出。",
    ]
    return {"headline": headline, "by_type": by_type, "notes": notes,
            "n_correct": len(right), "n_judged": len(judged)}


def main() -> int:
    ap = argparse.ArgumentParser(description="由逐题记录 + 人判结论生成 results.xlsx / errors.xlsx")
    ap.add_argument("--gen", default=str(DEFAULT_GEN), help="逐题原始记录 JSON")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--review", default=str(REVIEW))
    ap.add_argument("--out-dir", default=str(EVAL))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    targets = [out_dir / "results.xlsx", out_dir / "errors.xlsx"]
    # 探锁放在最前：等跑完统计才发现 Excel 占着文件是白费
    for t in targets:
        if t.exists():
            try:
                with t.open("r+b"):
                    pass
            except PermissionError:
                print(f"[开跑前检查] {t.name} 正被占用（多半是 Excel 开着）。"
                      f"关掉后重跑。")
                return 1

    recs = json.loads(Path(args.gen).read_text(encoding="utf-8"))
    ds = load_dataset(Path(args.dataset))
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    rows = build_rows(recs["records"], ds, review)

    meta = {
        "_records": recs["records"],
        "_meta": {
            "数据集": Path(args.dataset).name + f"（{len(rows)} 题）",
            "逐题记录": Path(args.gen).name,
            "判定文件": Path(args.review).name,
            "索引": str(recs.get("index", "")),
            "生成模型": str(recs.get("llm", "")),
            "temperature": config.llm_temperature,
            "召回深度": f"BM25 {recs.get('k_bm25')} / 向量 {recs.get('k_vector')}",
            "最终证据": f"Top-{config.top_k_final}（条件筛选题走结构化表组时放宽）",
            "结构化路由": config.structured_routing,
            "记录时间": recs.get("generated", ""),
            "生成本表时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    write_results_xlsx(targets[0], rows, meta)
    write_errors_xlsx(targets[1], rows, meta)

    st = summarize(rows)
    print(f"\n{'=' * 64}")
    print(f"R01–R10 评测汇总（{meta['_meta']['索引']}）")
    print(f"{'=' * 64}")
    for k, v in st["headline"]:
        print(f"  {k:24s} {v}")
    print(f"\n  各错误类型数量：")
    for code, name in ERROR_TYPES:
        qs = st["by_type"].get(code, [])
        print(f"    {code:4s} {name:12s} {len(qs)}  {'、'.join(qs) if qs else ''}")
    print(f"\n  逐题：")
    for r in rows:
        v = {True: "✅", False: "❌", None: "—"}[r["correct"]]
        print(f"    {r['qid']} {r['scope']} {r['type'][:10]:12s} {v} "
              f"gold {r['n_gold_in_evidence']}/{r['n_gold']} "
              f"{('引用' + str(len(r['citations'])) + '条') if r['citations'] else '未引用'}"
              f"{'  拒答' if r['refused'] else ''}"
              f"{'  ' + r['error_type'] if r['error_type'] else ''}")
    print(f"\n已写出 → {targets[0]}")
    print(f"已写出 → {targets[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
