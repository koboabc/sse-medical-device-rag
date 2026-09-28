"""解析 data/pdf/ 下的 2026 半年度报告，切成 chunk 并输出。

对应要求：
  1. PyMuPDF 提取正文
  2. pdfplumber 提取表格
  3. 表格保持原始行列关系（Markdown 管道表）
  4. 表格转 Markdown
  5. 保留 PDF 页码（page_start / page_end）
  6. 识别章节标题（第N节 / 第N节+标题 / 一、 / （一））
  7. 按章节、页面、段落切块
  8. 普通 chunk 约 500~1000 中文字符
  9. 表格不拆碎（整表成块；跨页续表合并）

输出：
  data/chunks/{stock_code}_2026H1.jsonl   每行一个 chunk（含 content）
  data/metadata.parquet                 全部 chunk 的元数据+内容
  data/parse_failures.csv               表格/文件级失败记录（不中断整体）

用法：
    python src/indexing/parse_reports.py
    python src/indexing/parse_reports.py --limit 1 --workers 1 --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

REPORT_NAME = "2026年半年度报告"

# ---- chunk 尺寸（中文字符数）----
TARGET_MIN = 500            # 达到这个长度才考虑收尾
TARGET_MAX = 1000           # 超过就断开
HARD_MIN = 150              # 小于此值的碎块尽量并入相邻块
REPACK_CAP = 2000           # 兜底重打包时的硬上限
SENT_SPLIT = re.compile(r"(?<=[。！？；])")
LEADIN_MAX = 200            # 单个「表格引导标题」块的最大长度
LEADIN_CLUSTER_MAX = 300    # 引导标题簇的总长度上限
LEADIN_HEIGHT = 150.0       # 引导标题块距表格上边界的最大高度（pt）
ABSORB_MAX = 700            # 可被表格吸收的相邻小正文总量

# ---- 表格 ----
TABLE_CAPTION_GAP = 50.0    # 表题与本表上边界的最大垂直距离（pt）
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# ---- 章节标题 ----
SEC_RE = re.compile(r"^第\s*([一二三四五六七八九十百]+)\s*节\s*(.*)$")
SUB2_RE = re.compile(r"^([一二三四五六七八九十]+)\s*、\s*(\S.*)$")
DOTLEAD_RE = re.compile(r"\.{5,}\s*\d+\s*$")      # 目录点线 "释义....... 5"
PAGENO_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")   # 页脚 "12 / 272"


@dataclass
class Chunk:
    chunk_id: str
    stock_code: str
    company: str
    report: str
    section: str
    page_start: int
    page_end: int
    content: str
    content_type: str
    # ---- 附加 ----
    subsection: str = ""
    char_count: int = 0


@dataclass
class Unit:
    """切块前的中间单元：一段正文，或一张表。"""
    kind: str                       # text / table
    section: str = ""
    subsection: str = ""
    page_start: int = 0
    page_end: int = 0
    text: str = ""                  # 正文段；表格则为空
    md: str = ""                    # 仅表格：Markdown 管道表
    leadin: str = ""                # 仅表格：被本表消费掉的表题

    def render(self) -> str:
        if self.kind != "table":
            return self.text
        return "\n".join(p for p in (self.leadin, self.md) if p)


# ============================================================
# 页眉页脚
# ============================================================
def detect_running_lines(page_texts: list[str]) -> set[str]:
    """按出现频次找出页眉页脚行。

    半年报每页顶部有「XX公司2026 年半年度报告」、底部有「12 / 272」。
    跨页高频重复的短行几乎一定是页眉页脚，比写死正则通用。
    """
    from collections import Counter

    freq: Counter[str] = Counter()
    for t in page_texts:
        for ln in {l.strip() for l in t.splitlines() if l.strip()}:
            freq[ln] += 1
    n = max(len(page_texts), 1)
    threshold = max(3, int(n * 0.30))
    return {ln for ln, c in freq.items() if c >= threshold}


def clean_lines(text: str, running: set[str]) -> list[str]:
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s in running or PAGENO_RE.match(s) or DOTLEAD_RE.search(s):
            continue
        out.append(s)
    return out


def detect_toc_pages(page_texts: list[str], running: set[str]) -> set[int]:
    """找出目录页（1-based）。

    目录页会逐行列出「第一节」「第二节」…，每一行都长得像章节标题。
    若不排除，既会把 section 污染成最后一个目录项，又会产生一堆 3 字碎块。
    正文里每个「第X节」都独占一页开头，所以「一页出现 ≥3 个」必是目录。
    """
    toc: set[int] = set()
    for i, t in enumerate(page_texts[:15]):
        lines = clean_lines(t, running)
        if any(l == "目录" for l in lines):
            toc.add(i + 1)
            continue
        hits = sum(1 for l in lines
                   if (SEC_RE.match(l) and len(l) < 12) or DOTLEAD_RE.search(l))
        if hits >= 3:
            toc.add(i + 1)
    return toc


# ============================================================
# 表格 -> Markdown
# ============================================================
def _cell(v) -> str:
    if v is None:
        return ""
    return str(v).replace("\r", " ").replace("\n", " ").strip().replace("|", "\\|")


def table_to_markdown(rows: list[list]) -> str:
    """把 pdfplumber 的行列数据转成 Markdown 表，保持行列关系。"""
    cleaned = [[_cell(c) for c in row] for row in rows]
    cleaned = [r for r in cleaned if any(c for c in r)]
    if len(cleaned) < MIN_TABLE_ROWS:
        return ""
    ncols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (ncols - len(r)) for r in cleaned]

    keep = [j for j in range(ncols) if any(r[j] for r in cleaned)]
    if len(keep) < MIN_TABLE_COLS:
        return ""
    cleaned = [[r[j] for j in keep] for r in cleaned]

    head, body = cleaned[0], cleaned[1:]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _md_lines(md: str) -> list[str]:
    return [l for l in md.splitlines() if l.strip()]


def _md_rows(md: str) -> list[str]:
    """数据行（跳过表头与分隔行）。"""
    lines = _md_lines(md)
    return lines[2:] if len(lines) >= 2 else []


def _md_header(md: str) -> list[str]:
    lines = _md_lines(md)
    if not lines:
        return []
    return [c.strip() for c in lines[0].strip("|").split("|")]


def is_headerish(row: list[str]) -> bool:
    """这一行像不像表头（续表合并时判断要不要丢掉重复表头）。"""
    txt = "".join(row)
    if any(k in txt for k in ("项目", "附注", "期末余额", "年初余额", "本期发生额")):
        return True
    return sum(ch.isdigit() for ch in txt) == 0 and len(txt) < 40


# ============================================================
# 单文件解析
# ============================================================
def extract_units(pdf_path: Path, log) -> tuple[list[Unit], list[dict]]:
    """把一份 PDF 变成按阅读顺序排列的 Unit 序列。"""
    errors: list[dict] = []
    doc = pymupdf.open(pdf_path)
    n_pages = doc.page_count
    page_texts = [doc[i].get_text() for i in range(n_pages)]
    running = detect_running_lines(page_texts)
    toc_pages = detect_toc_pages(page_texts, running)

    units: list[Unit] = []
    cur_section = ""
    cur_subsection = ""

    with pdfplumber.open(pdf_path) as pl:
        for i in range(n_pages):
            page_no = i + 1
            if page_no in toc_pages:          # 目录页整页跳过
                continue
            t0 = time.monotonic()

            # ---------- 1. 表格（pdfplumber）----------
            boxes: list[tuple] = []
            mds: list[str] = []
            try:
                for t in pl.pages[i].find_tables():
                    try:
                        rows = t.extract()
                    except Exception as e:  # noqa: BLE001
                        errors.append({"file": pdf_path.name, "page": page_no,
                                       "stage": "extract_table",
                                       "error": f"{type(e).__name__}: {e}"})
                        continue
                    md = table_to_markdown(rows) if rows else ""
                    if md:
                        boxes.append(tuple(t.bbox))
                        mds.append(md)
            except Exception as e:  # noqa: BLE001
                errors.append({"file": pdf_path.name, "page": page_no,
                               "stage": "find_tables",
                               "error": f"{type(e).__name__}: {e}"})

            # ---------- 2. 正文块（pymupdf），剔除落在表格内的 ----------
            events: list[tuple[float, str, object]] = []
            for b in doc[i].get_text("blocks"):
                x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                if not btext.strip():
                    continue
                if any(_overlaps((x0, y0, x1, y1), tb) for tb in boxes):
                    continue
                events.append((float(y0), "text", btext))
            # 正文与表格按阅读顺序交错，否则章节归属会错
            events += [(boxes[k][1], "table", k) for k in range(len(boxes))]
            events.sort(key=lambda e: (e[0], 0 if e[1] == "text" else 1))

            # ---------- 3. 表题：表格上方紧邻的短正文块被本表「消费」 ----------
            #    附注里的排版是「40、应交税费」/「(1) 明细情况」/ 表格，
            #    得把整簇短块都收走，否则小标题会变成 7 字的孤儿块。
            leadin: dict[int, str] = {}
            consumed: set[int] = set()
            for ei, (_y, kind, payload) in enumerate(events):
                if kind != "table":
                    continue
                k = int(payload)
                top = boxes[k][1]
                parts: list[str] = []
                total = 0
                for back in range(ei - 1, -1, -1):
                    if events[back][1] == "table" or back in consumed:
                        break
                    lines = clean_lines(str(events[back][2]), running)
                    flat = " ".join(lines)
                    if (not flat or len(flat) > LEADIN_MAX
                            or total + len(flat) > LEADIN_CLUSTER_MAX
                            or top - events[back][0] > LEADIN_HEIGHT):
                        break
                    parts.append("\n".join(lines))
                    total += len(flat)
                    consumed.add(back)
                if parts:
                    leadin[k] = "\n".join(reversed(parts))

            # ---------- 4. 依序产出 Unit ----------
            for ei, (_y, kind, payload) in enumerate(events):
                if kind == "text":
                    lines = clean_lines(str(payload), running)
                    if not lines:
                        continue
                    # 块内若含章节标题，在标题处切开，前后分属不同章节。
                    # 注意：被表格消费掉的块（ei in consumed）也要扫标题——
                    # 否则「第一节 释义」一旦成为表题，整节的 section 就丢了。
                    seg: list[str] = []
                    for li, ln in enumerate(lines):
                        if is_section_line(ln):
                            if seg and ei not in consumed:
                                units.append(Unit("text", cur_section, cur_subsection,
                                                  page_no, page_no, "\n".join(seg)))
                            seg = []
                            cur_section = _fmt_section(SEC_RE.match(ln),
                                                       lines[li + 1:li + 2])
                            cur_subsection = ""
                            seg.append(ln)
                            continue
                        m2 = SUB2_RE.match(ln)
                        if m2 and len(ln) < 40:
                            cur_subsection = ln
                        seg.append(ln)
                    if seg and ei not in consumed:
                        units.append(Unit("text", cur_section, cur_subsection,
                                          page_no, page_no, "\n".join(seg)))
                else:
                    k = int(payload)
                    units.append(Unit("table", cur_section, cur_subsection,
                                      page_no, page_no, "", mds[k],
                                      leadin.get(k, "")))

            dt = time.monotonic() - t0
            if dt > 5.0:
                log(f"    p{page_no} 慢页 {dt:.1f}s（表格 {len(mds)}）")

    doc.close()
    return units, errors


TITLE_MAX = 16
TITLE_BAD_END = ("。", ".", "：", "；", ";", "，", ",")
TITLE_BAD_WORD = ("之", "详见", "参见", "本报告", "报告期", "本公司")


def is_section_line(ln: str) -> bool:
    """这一行是不是真的章节标题。

    正文里的交叉引用会被排到行首，例如「重要提示」页的
    「第三节“管理层讨论与分析”之“四、风险因素”。」——它会命中第X节正则。
    真标题短、不以句号收尾、不含「之/详见/参见」这类连接词。
    """
    m = SEC_RE.match(ln)
    if not m or len(ln) >= 40:
        return False
    title = (m.group(2) or "").strip()
    if not title:
        return True                      # 独占一行的「第X节」
    if len(title) > TITLE_MAX or title.endswith(TITLE_BAD_END):
        return False
    return not any(w in title for w in TITLE_BAD_WORD)


def _fmt_section(m: re.Match, nxt: list[str] | None = None) -> str:
    """章节名。标题常被排成独立一行（「第八节」换行「财务报告」），
    缺标题时向后看一行补上——检索「管理层讨论与分析」时靠它命中。"""
    num, title = m.group(1), (m.group(2) or "").strip()
    if not title and nxt:
        cand = nxt[0].strip()
        if cand and len(cand) <= 20 and not SEC_RE.match(cand) and not SUB2_RE.match(cand):
            title = cand
    return f"第{num}节 {title}".strip()


def _overlaps(a: tuple, b: tuple) -> bool:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if ix <= 0 or iy <= 0:
        return False
    return (ix * iy) / max(1e-6, (a[2] - a[0]) * (a[3] - a[1])) > 0.35


def merge_continued_tables(units: list[Unit]) -> list[Unit]:
    """合并跨页续表（如合并资产负债表 p127 35行 + p128 48行）。

    条件：紧邻上一页、列数相同、同章节；续表重复的表头丢掉。
    合并后 page_start/page_end 覆盖两页，满足要求 5 与 9。
    """
    out: list[Unit] = []
    for u in units:
        if (u.kind == "table" and out and out[-1].kind == "table"
                and u.page_start == out[-1].page_end + 1
                and u.section == out[-1].section):
            prev_hdr, cur_hdr = _md_header(out[-1].md), _md_header(u.md)
            if prev_hdr and len(prev_hdr) == len(cur_hdr):
                body = _md_rows(u.md)
                if body:
                    cells = [c.strip() for c in body[0].strip("|").split("|")]
                    if is_headerish(cells):
                        body = body[1:]
                if body:
                    out[-1].md += "\n" + "\n".join(body)
                    out[-1].page_end = u.page_end
                    continue
        out.append(u)
    return out


# ============================================================
# 切块
# ============================================================
def build_chunks(units: list[Unit], meta: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    b_ps = b_pe = 0
    b_sec = b_sub = ""

    def flush() -> None:
        nonlocal buf, buf_len, b_ps, b_pe
        content = "\n".join(buf).strip()
        if content:
            chunks.append(_mk(meta, b_sec, b_sub, b_ps, b_pe, content, "text"))
        buf, buf_len, b_ps, b_pe = [], 0, 0, 0

    def start(u: Unit) -> None:
        nonlocal b_ps, b_sec, b_sub
        b_ps, b_sec, b_sub = u.page_start, u.section, u.subsection

    for u in units:
        if u.kind == "table":
            flush()
            content = u.render().strip()
            if content:
                chunks.append(_mk(meta, u.section, u.subsection,
                                  u.page_start, u.page_end, content, "table"))
            continue

        if buf and u.section != b_sec:
            flush()
        if not buf:
            start(u)

        for pc in (_split_long(u.text) if len(u.text) > TARGET_MAX else [u.text]):
            if buf and u.page_start != b_pe and buf_len >= TARGET_MIN:
                flush(); start(u)                       # 尊重「按页面」
            if buf and buf_len + len(pc) > TARGET_MAX and buf_len >= TARGET_MIN:
                flush(); start(u)
            if not buf:
                start(u)
            buf.append(pc)
            buf_len += len(pc)
            b_pe = u.page_start
    flush()

    chunks = absorb_small_text(chunks)
    for i, c in enumerate(chunks, 1):
        c.chunk_id = f"{meta['stock_code']}-{i:04d}"
    return chunks


def absorb_small_text(chunks: list[Chunk]) -> list[Chunk]:
    """先重打包正文 run，再把剩下的孤儿小块塞进相邻表格块。

    表格会把缓冲区打断，导致「表前小标题」变成孤立碎块。
    这类文字本来就是表格的标题/备注，并进表格比单列更利于检索。
    """
    chunks = repack_text_runs(chunks)
    out: list[Chunk] = []
    pending: list[Chunk] = []          # 尚未落定的小正文块

    for c in chunks:
        if c.content_type == "text":
            pending.append(c)
            continue
        # 只取 pending 尾部「同章节且够不上目标区间」的连续小段当作表题。
        # 逐个回退而不是整批判断：前面若压着一个大块，不该带累后面这个小表题。
        cut = len(pending)
        while (cut > 0 and pending[cut - 1].section == c.section
               and pending[cut - 1].char_count < TARGET_MIN):
            cut -= 1
        out.extend(pending[:cut])
        tail = pending[cut:]
        if tail and sum(p.char_count for p in tail) <= ABSORB_MAX:
            c.content = "\n".join([p.content for p in tail] + [c.content])
            c.char_count = len(c.content)
            c.page_start = min(c.page_start, tail[0].page_start)
        else:
            out.extend(tail)
        pending = []
        out.append(c)
    out.extend(pending)
    return out


def _mk(meta: dict, sec: str, sub: str, ps: int, pe: int,
        content: str, ctype: str) -> Chunk:
    return Chunk("", meta["stock_code"], meta["company"], REPORT_NAME,
                 sec or "(报告前部)", ps, pe, content, ctype, sub, len(content))


def repack_text_runs(chunks: list[Chunk]) -> list[Chunk]:
    """把「连续同章节的正文块」重新打包到 500~1000 字。

    表格会把缓冲区打断，表格前后的引导文字容易变成碎块；
    这里以同章节连续正文为一个 run，贪心重打包，消掉过小的块。
    """
    out: list[Chunk] = []
    i, n = 0, len(chunks)
    while i < n:
        if chunks[i].content_type != "text":
            out.append(chunks[i]); i += 1
            continue
        j = i
        while (j + 1 < n and chunks[j + 1].content_type == "text"
               and chunks[j + 1].section == chunks[i].section):
            j += 1
        run = chunks[i:j + 1]
        cur = run[0]
        for nxt in run[1:]:
            merged_len = cur.char_count + nxt.char_count
            # 优先合并到目标区间内；实在合不下时，宁可略微超标也别留孤儿块
            if cur.char_count < TARGET_MIN and (
                    merged_len <= TARGET_MAX
                    or (cur.char_count < HARD_MIN and merged_len <= REPACK_CAP)):
                cur.content += "\n" + nxt.content
                cur.char_count = len(cur.content)
                cur.page_end = nxt.page_end
                if not cur.subsection:
                    cur.subsection = nxt.subsection
            else:
                out.append(cur); cur = nxt
        out.append(cur)
        i = j + 1
    return out


def _split_long(text: str) -> list[str]:
    out, cur = [], ""
    for s in (s for s in SENT_SPLIT.split(text) if s):
        if cur and len(cur) + len(s) > TARGET_MAX:
            out.append(cur); cur = s
        else:
            cur += s
    if cur:
        out.append(cur)
    return out


# ============================================================
def parse_one(pdf_path_str: str, meta: dict, verbose: bool) -> dict:
    pdf_path = Path(pdf_path_str)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t0 = time.monotonic()
    res = {"stock_code": meta["stock_code"], "company": meta["company"],
           "source_file": pdf_path.name, "chunks": 0, "table_chunks": 0,
           "errors": [], "status": "ok", "elapsed": 0.0}
    try:
        units, errors = extract_units(pdf_path, log)
        res["errors"] = errors
        chunks = build_chunks(merge_continued_tables(units), meta)

        out_dir = config.data_dir / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{meta['stock_code']}_2026H1.jsonl").open(
                "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

        res["chunks"] = len(chunks)
        res["table_chunks"] = sum(1 for c in chunks if c.content_type == "table")
    except Exception as e:  # noqa: BLE001
        res["status"] = "failed"
        res["errors"] = [{"file": pdf_path.name, "page": 0, "stage": "parse_file",
                          "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-800:]}]
    res["elapsed"] = time.monotonic() - t0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with (config.data_dir / "manifest.csv").open(
            "r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["download_status"] == "success"]
    if args.limit:
        rows = rows[: args.limit]
    print(f"待解析 {len(rows)} 份报告（并行 {args.workers} 进程）\n")

    jobs = [(str(config.root / r["local_path"]),
             {"stock_code": r["stock_code"], "company": r["company"]})
            for r in rows]

    t0 = time.monotonic()
    results: list[dict] = []
    if args.workers <= 1:
        for path, m in jobs:
            print(f"  解析 {m['stock_code']} {m['company']} ...", flush=True)
            results.append(parse_one(path, m, args.verbose))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(parse_one, p, m, False)
                                     for p, m in jobs]):
                r = fut.result()
                results.append(r)
                print(f"  [{r['stock_code']}] {'✓' if r['status'] == 'ok' else '✗'} "
                      f"{r['chunks']:>4} chunks (表格 {r['table_chunks']:>3})  "
                      f"{r['elapsed']:.1f}s", flush=True)
    elapsed = time.monotonic() - t0

    # ---- 失败记录（不中断整体）----
    all_errors = [e for r in results for e in r["errors"]]
    fail_path = config.data_dir / "parse_failures.csv"
    keys = ["file", "page", "stage", "error"]
    with fail_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for e in all_errors:
            w.writerow({k: e.get(k, "") for k in keys})
    print(f"\n解析失败/异常记录 {len(all_errors)} 条 -> {fail_path}")

    # ---- 汇总 parquet ----
    import pandas as pd

    records: list[dict] = []
    for r in results:
        jl = config.data_dir / "chunks" / f"{r['stock_code']}_2026H1.jsonl"
        if jl.exists():
            with jl.open("r", encoding="utf-8") as f:
                records += [json.loads(l) for l in f if l.strip()]
    df = pd.DataFrame(records)
    pq = config.data_dir / "metadata.parquet"
    df.to_parquet(pq, index=False)

    back = pd.read_parquet(pq)                        # 回读断言
    assert len(back) == len(df), f"parquet 回读不一致 {len(back)} != {len(df)}"
    assert back["chunk_id"].is_unique, "chunk_id 有重复"

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print("\n" + "=" * 64)
    print(f"  报告总数    : {len(results)}  (成功 {n_ok} / 失败 {len(results) - n_ok})")
    print(f"  chunk 总数  : {len(df)}")
    print(f"    其中表格  : {(df['content_type'] == 'table').sum()}")
    print(f"    其中正文  : {(df['content_type'] == 'text').sum()}")
    print(f"  字符总数    : {df['char_count'].sum():,}")
    print(f"  总耗时      : {elapsed:.1f} 秒")
    print(f"  chunks 目录 : {config.data_dir / 'chunks'}")
    print(f"  parquet     : {pq}")
    for r in results:
        if r["status"] != "ok":
            print(f"    ✗ {r['stock_code']} {r['company']}: {r['errors'][0]['error'][:100]}")
    print("=" * 64)

    tx = df[df["content_type"] == "text"]["char_count"]
    if len(tx):
        print(f"\n正文 chunk 字符数：min={tx.min()} p25={int(tx.quantile(.25))} "
              f"中位={int(tx.median())} p75={int(tx.quantile(.75))} max={tx.max()}")
        print(f"落在 {TARGET_MIN}~{TARGET_MAX} 区间："
              f"{((tx >= TARGET_MIN) & (tx <= TARGET_MAX)).mean():.0%}")
        print(f"小于 {HARD_MIN} 字的碎块：{(tx < HARD_MIN).sum()} 个")
    tb = df[df["content_type"] == "table"]["char_count"]
    if len(tb):
        print(f"表格 chunk 字符数：中位={int(tb.median())} max={tb.max()}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
