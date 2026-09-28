"""核验 data/manifest.csv：机械校验全部行 + 随机抽样回源比对。

核验三件事：
  1. 全部行的 URL 里嵌的股票代码 == 该行的 stock_code（防张冠李戴）
  2. 全部行确实是「全文」而非「摘要」——回源确认摘要另有一个不同的 PDF
  3. 随机抽 5 家重新打上交所接口，逐字段比对 manifest 是否与官方源一致

用法：
    python scripts/verify_manifest.py [--sample 5] [--seed 20260925]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config  # noqa: E402
from src.ingest.fetch_announcements import (  # noqa: E402
    SECURITY_TYPE, SSE_QUERY, SSE_STATIC, SSE_WWW,
    _headers, is_full_2026_h1_report, query_bulletins,
)

OK, BAD = "[OK]", "[!!]"
problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {OK if cond else BAD} {msg}")
    if not cond:
        problems.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    manifest_path = config.data_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"manifest: {len(rows)} 行 <- {manifest_path}\n")

    # ================= 1. 机械校验全部行 =================
    print("=" * 70)
    print("1) 机械校验（全部行）")
    print("=" * 70)
    for r in rows:
        code, title, url = r["stock_code"], r["title"], r["pdf_url"]
        fname = url.rsplit("/", 1)[-1]
        check(fname.startswith(f"{code}_"),
              f"{code} PDF 文件名前缀嵌码正确: {fname}")
        check(r["pdf_url"] == f"{SSE_STATIC}{r['sse_url_path']}",
              f"{code} pdf_url = static + 官方 URL 字段")
        check(r["announcement_url"] == f"{SSE_WWW}{r['sse_url_path']}",
              f"{code} announcement_url = www + 官方 URL 字段")
        check("2026-07-01" <= r["publish_date"] <= "2026-09-30",
              f"{code} 披露日 {r['publish_date']} 落在半年报窗口内")
        check(is_full_2026_h1_report(title), f"{code} 标题符合全文口径: {title}")
        check("摘要" not in title, f"{code} 标题不含「摘要」")
        check(r["bulletin_type"] == "半年报",
              f"{code} 官方 BULLETIN_TYPE = {r['bulletin_type']}")
        want = f"{code}_{r['company']}_2026H1.pdf"
        check(r["local_path"] == f"data/pdf/{want}",
              f"{code} local_path 命名合规: data/pdf/{want}")
        check(r["download_status"] == "pending",
              f"{code} download_status = {r['download_status']}")
        check(r["report_year"] == "2026" and r["report_type"] == "半年度报告",
              f"{code} report_year/type = {r['report_year']}/{r['report_type']}")
        print()

    # ================= 2. 随机抽样回源比对 =================
    random.seed(args.seed)
    sample = random.sample(rows, min(args.sample, len(rows)))
    print("=" * 70)
    print(f"2) 随机抽样 {len(sample)} 家回源比对（seed={args.seed}，可复现）")
    print("=" * 70)

    with requests.Session() as s:
        for r in sample:
            code = r["stock_code"]
            recs = query_bulletins(code, s)
            print(f"\n股票代码 {code}  →  上交所定期报告接口返回 {len(recs)} 条")
            fulls = [x for x in recs if is_full_2026_h1_report(x.get("TITLE") or "")]
            abstracts = [x for x in recs
                         if "2026年半年度报告" in (x.get("TITLE") or "")
                         and "摘要" in (x.get("TITLE") or "")]
            for x in recs:
                print(f"     {x.get('SSEDATE')} | {x.get('SECURITY_CODE')} | "
                      f"{x.get('BULLETIN_TYPE')} | {x.get('TITLE')}")
                print(f"        {x.get('URL')}")

            check(len(fulls) >= 1, f"{code} 官方源确有完整半年报")
            if not fulls:
                continue
            f0 = fulls[0]
            check(f0.get("SECURITY_CODE") == code,
                  f"{code} 官方记录 SECURITY_CODE 一致")
            check(f0.get("TITLE") == r["title"],
                  f"{code} 标题一致: {f0.get('TITLE')}")
            check(f0.get("SSEDATE") == r["publish_date"],
                  f"{code} 披露日一致: {f0.get('SSEDATE')}")
            check(f0.get("URL") == r["sse_url_path"],
                  f"{code} PDF 路径一致: {f0.get('URL')}")
            check(f0.get("BULLETIN_TYPE") == "半年报",
                  f"{code} 类型为半年报（非季报/年报）")
            # 全文 != 摘要
            if abstracts:
                check(all(a.get("URL") != f0.get("URL") for a in abstracts),
                      f"{code} 摘要 URL 与全文 URL 不同（未误取摘要）"
                      f" -> 全文 {f0.get('URL').rsplit('/',1)[-1]}"
                      f" vs 摘要 {abstracts[0].get('URL').rsplit('/',1)[-1]}")
            else:
                print(f"     (该家无摘要记录可比对)")
            time.sleep(config.download_delay)

    print("\n" + "=" * 70)
    if problems:
        print(f"核验失败：{len(problems)} 项")
        for p in problems:
            print("   -", p)
    else:
        print("核验通过：全部检查项一致")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
