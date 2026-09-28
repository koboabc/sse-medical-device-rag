"""从上交所官方公告查询接口提取 2026 年半年度报告，生成 data/manifest.csv。

官方数据源（上交所自己的接口，非第三方）：
    https://query.sse.com.cn/security/stock/queryCompanyBulletin.do

接口要点（实测得出，2026-09-25）：
  * 必须带 `Referer: https://www.sse.com.cn/`，否则被拒
  * 必须给 `securityType`，只给 productId 会返回 0 条
  * `reportType2=DQBG` 圈定「定期报告」——这是拿到半年报的关键参数，
    `reportType` 是年报/季报的分类，与半年报无关，别用
  * 返回的是 JSONP，需剥掉 jsonCallBack(...) 外壳
  * 记录里的 `BULLETIN_TYPE` 字段直接标注「半年报」，比标题匹配可靠

筛选口径（按作业要求）：
  只要完整的「2026年半年度报告」，排除：摘要 / 英文版 / 修订 / 更正 / 补充 等

用法：
    python src/ingest/fetch_announcements.py            # 生成 manifest
    python src/ingest/fetch_announcements.py --dry-run  # 只看不写
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

# ---- 上交所接口 ----
SSE_QUERY = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
SSE_COMPANY = "https://query.sse.com.cn/commonQuery.do"   # 公司基本信息（含中文全称）
SSE_STATIC = "https://static.sse.com.cn"   # PDF 直链域名
SSE_WWW = "https://www.sse.com.cn"         # 公告页域名（301 跳 static）

# 公司基本信息查询的 sqlId（实测可用）
COMPANY_SQL_ID = "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L"

# 上交所要求 securityType 指定证券类别；0101=主板A股 120100/020100/020200/120200=科创板等
SECURITY_TYPE = "0101,120100,020100,020200,120200"

REPORT_YEAR = 2026
# 半年报披露窗口：报告期 2026-06-30 之后才开始披露
BEGIN_DATE = "2026-07-01"
END_DATE = "2026-09-30"

# ---- 筛选规则 ----
# 只认「以 2026年半年度报告[全文] 结尾」的标题：摘要（...报告摘要）、
# 英文版（...报告（英文版））、修订版（...报告（修订版））都会自然落选。
# 注意必须允许「全文」后缀 —— 圣湘生物的标题就是
# 「圣湘生物科技股份有限公司2026年半年度报告全文」，那正是完整报告本身，
# 早先只锚定「报告$」会把它误判为缺失。
FULL_REPORT_RE = re.compile(r"2026\s*年\s*半年度报告\s*(全文)?\s*$")

# 兜底排除词：万一个别公司标题形态特殊，这些词出现即排除
EXCLUDE_WORDS = [
    "摘要", "英文", "English", "ENGLISH",
    "修订", "更正", "补充", "取消", "撤回",
    "说明会", "业绩", "问询", "意见", "公告",
]

# 我们排除掉的、但值得记录下来的「近似项」，便于人工判断有无遗漏
NEAR_MISS_RE = re.compile(r"2026\s*年\s*半年度报告")


@dataclass
class Announcement:
    stock_code: str
    company: str
    report_type: str
    report_year: int
    publish_date: str
    announcement_url: str
    pdf_url: str
    local_path: str
    download_status: str
    # ---- 附加字段（作业要求「至少包括」，这些便于核查）----
    security_name: str = ""       # 证券简称（上交所口径）
    bulletin_type: str = ""       # 上交所 BULLETIN_TYPE，如「半年报」
    title: str = ""               # 公告原始标题
    sse_url_path: str = ""        # 接口返回的原始相对路径，可回溯
    company_full_name_official: str = ""  # 官方 FULL_NAME 字段
    name_mismatch: str = ""       # 与 company_list.csv 的 公司名称 不一致时记录

@dataclass
class MissingReport:
    stock_code: str
    company: str
    report_type: str
    report_year: int
    reason: str
    checked_at: str
    near_misses: str = ""         # 查到的近似公告标题，用 | 分隔


def _headers() -> dict[str, str]:
    return {
        "Referer": "https://www.sse.com.cn/",
        "User-Agent": config.user_agent,
        "Accept": "*/*",
    }


def query_company_info(stock_code: str, session: requests.Session) -> dict:
    """查公司基本信息，返回含 FULL_NAME（官方中文全称）的字典。

    用官方全称而不是人工填写的名字做文件名 —— 人工名字会过时
    （例：600055 万东医疗 2017 年已由「华润万东医疗装备股份有限公司」
    更名为「北京万东医疗科技股份有限公司」）。
    """
    params = {
        "sqlId": COMPANY_SQL_ID,
        "type": "inParams",
        "STOCK_CODE": stock_code,
        "isPagination": "false",
    }
    r = session.get(SSE_COMPANY, params=params, headers=_headers(), timeout=30)
    r.raise_for_status()
    result = (r.json() or {}).get("result") or []
    return result[0] if result else {}


def query_bulletins(stock_code: str, session: requests.Session) -> list[dict]:
    """查一家公司的定期报告公告，返回原始记录列表。"""
    params = {
        "jsonCallBack": "jsonpCallback",
        "isPagination": "true",
        "productId": stock_code,
        "securityType": SECURITY_TYPE,
        "reportType2": "DQBG",          # 定期报告
        "beginDate": BEGIN_DATE,
        "endDate": END_DATE,
        "pageHelp.pageSize": "100",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.endPage": "1",
    }
    r = session.get(SSE_QUERY, params=params, headers=_headers(), timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    # 剥 JSONP 外壳
    lp, rp = text.find("("), text.rfind(")")
    if lp == -1 or rp == -1:
        raise ValueError(f"{stock_code}: 返回非 JSONP: {text[:200]}")
    payload = json.loads(text[lp + 1: rp])
    return payload.get("pageHelp", {}).get("data") or []


def is_full_2026_h1_report(title: str) -> bool:
    """判断标题是否为「完整的 2026 年半年度报告」。"""
    t = title.strip()
    if not FULL_REPORT_RE.search(t):
        return False
    return not any(w in t for w in EXCLUDE_WORDS)


def pick_company_name(row: dict) -> str:
    return (row.get("公司名称") or "").strip()


def sanitize_filename(name: str) -> str:
    """清掉 Windows 文件名非法字符。"""
    return re.sub(r'[\\/:*?"<>|]', "", name).strip()


def build_local_path(stock_code: str, company: str) -> str:
    """目标文件名：股票代码_公司名称_2026H1.pdf（相对项目根目录）。"""
    fname = f"{stock_code}_{sanitize_filename(company)}_2026H1.pdf"
    return str((config.pdf_dir / fname).relative_to(config.root)).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只查询打印，不写文件")
    args = ap.parse_args()

    config.ensure_dirs()

    src_csv = config.company_list_path
    if not src_csv.exists():
        print(f"[FAIL] 找不到公司名单 {src_csv}")
        return 1

    with src_csv.open("r", encoding="utf-8-sig", newline="") as f:
        companies = list(csv.DictReader(f))
    print(f"公司名单：{len(companies)} 家 <- {src_csv.name}")
    print(f"查询区间：{BEGIN_DATE} ~ {END_DATE}   reportType2=DQBG（定期报告）\n")

    found: list[Announcement] = []
    missing: list[MissingReport] = []
    checked_at = time.strftime("%Y-%m-%d %H:%M:%S")

    with requests.Session() as session:
        for row in companies:
            code = row["股票代码"].strip()
            listed_name = pick_company_name(row)

            # 先用官方接口取中文全称，取不到才退回名单里的名字
            try:
                info = query_company_info(code, session)
            except Exception as e:  # noqa: BLE001
                info = {}
                print(f"  [{code}] 公司信息查询失败（退回名单名）: {e}")
            official = (info.get("FULL_NAME") or "").strip()
            company = official or listed_name
            mismatch = ""
            if official and listed_name and official != listed_name:
                mismatch = f"名单={listed_name} 官方={official}"
                print(f"  [{code}] !! 公司名不一致 -> {mismatch}")

            try:
                records = query_bulletins(code, session)
            except Exception as e:  # noqa: BLE001
                print(f"  [{code}] 查询失败: {type(e).__name__}: {e}")
                missing.append(MissingReport(
                    stock_code=code, company=company, report_type="半年度报告",
                    report_year=REPORT_YEAR, reason=f"接口查询失败: {e}",
                    checked_at=checked_at))
                time.sleep(config.download_delay)
                continue

            hits = [r for r in records
                    if is_full_2026_h1_report(r.get("TITLE") or "")]
            near = [r.get("TITLE", "") for r in records
                    if NEAR_MISS_RE.search(r.get("TITLE") or "")
                    and r not in hits]

            if hits:
                # 若有多条命中，取披露日期最新的（正常只会有一条）
                hits.sort(key=lambda r: r.get("SSEDATE") or "", reverse=True)
                h = hits[0]
                url_path = h.get("URL") or ""
                found.append(Announcement(
                    stock_code=code,
                    company=company,
                    report_type="半年度报告",
                    report_year=REPORT_YEAR,
                    publish_date=(h.get("SSEDATE") or "").strip(),
                    announcement_url=f"{SSE_WWW}{url_path}",
                    pdf_url=f"{SSE_STATIC}{url_path}",
                    local_path=build_local_path(code, company),
                    download_status="pending",
                    security_name=(h.get("SECURITY_NAME") or "").strip(),
                    bulletin_type=(h.get("BULLETIN_TYPE") or "").strip(),
                    title=(h.get("TITLE") or "").strip(),
                    sse_url_path=url_path,
                    company_full_name_official=official,
                    name_mismatch=mismatch,
                ))
                extra = f"  (另有 {len(hits) - 1} 条命中)" if len(hits) > 1 else ""
                nm = f"  [近似: {' | '.join(near)}]" if near else ""
                print(f"  [{code}] ✓ {h.get('SSEDATE')} {h.get('TITLE')}{extra}{nm}")
            else:
                reason = ("接口返回 0 条定期报告" if not records
                          else f"有 {len(records)} 条定期报告，但无完整 2026 半年报")
                missing.append(MissingReport(
                    stock_code=code, company=company, report_type="半年度报告",
                    report_year=REPORT_YEAR, reason=reason, checked_at=checked_at,
                    near_misses=" | ".join(t for t in near if t)))
                print(f"  [{code}] ✗ 未找到 —— {reason}"
                      + (f"  [近似: {' | '.join(near)}]" if near else ""))

            time.sleep(config.download_delay)   # 对官网友好

    # ---- 写 manifest ----
    manifest_path = config.data_dir / "manifest.csv"
    missing_path = config.data_dir / "missing_reports.csv"
    manifest_fields = list(Announcement.__dataclass_fields__.keys())
    missing_fields = list(MissingReport.__dataclass_fields__.keys())

    if args.dry_run:
        print("\n[dry-run] 未写文件")
    else:
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=manifest_fields)
            w.writeheader()
            for a in found:
                w.writerow(asdict(a))
        with missing_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=missing_fields)
            w.writeheader()
            for m in missing:
                w.writerow(asdict(m))

        # 回读断言：写多少读回多少
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            back = list(csv.DictReader(f))
        assert len(back) == len(found), f"回读不一致 {len(back)} != {len(found)}"

    print("\n" + "=" * 62)
    print(f"命中 {len(found)} 家 / 缺失 {len(missing)} 家（名单共 {len(companies)} 家）")
    print(f"  manifest        -> {manifest_path}")
    print(f"  missing_reports -> {missing_path}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
