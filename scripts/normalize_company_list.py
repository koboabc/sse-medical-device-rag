"""把 data/company_list.csv 重写为 utf-8-sig（带 BOM）并校验完整性。

为什么要这一步：中文 Windows 的 Excel 默认按本地代码页解读无 BOM 的 UTF-8，
公司名会变乱码。加 BOM 后 Excel 与 pandas 都能正确读。

同时做断言式校验 —— 校验不过就非零退出，不静默通过。

用法：
    python scripts/normalize_company_list.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "company_list.csv"

REQUIRED_FIELDS = ["股票代码", "公司名称", "交易所", "行业", "报告年份", "报告类型"]
MIN_COMPANIES = 10


def main() -> int:
    if not CSV_PATH.exists():
        print(f"[FAIL] 找不到 {CSV_PATH}")
        return 1

    # 读：容忍 BOM
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"读入 {len(rows)} 行 <- {CSV_PATH}")

    # ---- 校验 1：表头字段齐全 ----
    got_fields = list(rows[0].keys()) if rows else []
    if got_fields != REQUIRED_FIELDS:
        print(f"[FAIL] 表头不符\n  期望: {REQUIRED_FIELDS}\n  实际: {got_fields}")
        return 1
    print(f"[OK] 表头字段: {REQUIRED_FIELDS}")

    # ---- 校验 2：数量达标 ----
    if len(rows) < MIN_COMPANIES:
        print(f"[FAIL] 只有 {len(rows)} 家，少于要求的 {MIN_COMPANIES} 家")
        return 1
    print(f"[OK] 公司数量 {len(rows)} >= {MIN_COMPANIES}")

    # ---- 校验 3：代码唯一、6 位数字 ----
    codes = [r["股票代码"].strip() for r in rows]
    bad = [c for c in codes if not (len(c) == 6 and c.isdigit())]
    if bad:
        print(f"[FAIL] 非 6 位数字代码: {bad}")
        return 1
    dupes = {c for c in codes if codes.count(c) > 1}
    if dupes:
        print(f"[FAIL] 重复代码: {dupes}")
        return 1
    print("[OK] 股票代码 6 位数字且无重复")

    # ---- 校验 4：无空字段 ----
    for i, r in enumerate(rows, start=2):
        empty = [k for k, v in r.items() if not str(v).strip()]
        if empty:
            print(f"[FAIL] 第 {i} 行空字段: {empty}")
            return 1
    print("[OK] 无空字段")

    # ---- 校验 5：交易所与代码前缀自洽 ----
    for r in rows:
        code, exch = r["股票代码"], r["交易所"]
        is_star = code.startswith("688")
        if is_star and "科创板" not in exch:
            print(f"[FAIL] {code} 属科创板但交易所写作「{exch}」")
            return 1
        if not is_star and "科创板" in exch:
            print(f"[FAIL] {code} 非 688 开头却标为科创板")
            return 1
    print("[OK] 交易所与代码前缀自洽")

    # ---- 重写为 utf-8-sig（带 BOM）----
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ---- 回读断言 ----
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reread = list(csv.DictReader(f))
    assert len(reread) == len(rows), f"回读行数不一致: {len(reread)} != {len(rows)}"
    assert reread == rows, "回读内容与写入不一致"
    head = CSV_PATH.open("rb").read(3)
    assert head == b"\xef\xbb\xbf", f"未写入 BOM，前三字节 = {head!r}"
    print(f"[OK] 已重写为 utf-8-sig 并回读断言通过（{len(reread)} 行，BOM 就位）")

    # ---- 汇总 ----
    star = sum(1 for r in rows if r["股票代码"].startswith("688"))
    print("\n" + "-" * 60)
    print(f"合计 {len(rows)} 家：上证主板 {len(rows) - star} 家 / 科创板 {star} 家")
    for r in rows:
        print(f"  {r['股票代码']}  {r['公司名称']:<34} {r['行业']}")
    print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
