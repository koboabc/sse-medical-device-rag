"""从问题里认出是哪家（哪几家）公司 —— 公司名 → stock_code。

为什么不用手写别名表：`data/manifest.csv` 里本来就有上交所给的
`security_name`（证券简称，如「联影医疗」）、`company_full_name_official`
（官方全称，如「北京万东医疗科技股份有限公司」）和 `stock_code`。
用数据而不是人工清单，才不会像 600055 更名那样悄悄过时。

匹配规则（都是为了「宁可少认出，不能认错」——认错会给错误公司加权，
直接污染召回，而少认一家只是退化成不加权）：
  1. **最长优先**：长别名先匹配，命中后把该段「吃掉」，短别名不能在其内部
     再匹配（否则「上海微创心脉医疗科技」被吃掉前，「微创」可能命中别家）
  2. **全局唯一性过滤**：一个别名如果在表里对应**不止一家**公司，整条丢弃。
     这条挡掉的是「医疗器械」「医疗科技」「生物科技」这类行业通名 —— 它们
     一旦进表，问「医疗器械板块哪家研发投入最高」就会给随机一家加权
  3. **派生别名只从证券简称剥行业词尾**（万东医疗 → 万东、基蛋生物 → 基蛋），
     绝不靠砍前缀生成（砍前缀只会产出「华医疗器械」这种碎片）
  4. **不做模糊匹配**：不做编辑距离/包含式猜测。认不出来就不加权。

用法：
    from src.retrieval.entities import find_companies

    find_companies("对比心脉医疗和惠泰医疗的研发投入强度")
    # → [Company(688016, '心脉医疗'), Company(688617, '惠泰医疗')]
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

MANIFEST = config.data_dir / "manifest.csv"

# 从官方全称剥掉的法人词尾（剥一次就停）
LEGAL_SUFFIXES = ("股份有限公司", "有限责任公司", "有限公司")
# 从证券简称剥掉的行业词尾，用于派生更短的叫法（万东医疗 → 万东）
INDUSTRY_SUFFIXES = ("医疗器械", "医疗科技", "医疗", "生物科技", "生物", "器械")
MIN_ALIAS_LEN = 2          # 派生别名下限（「万东」「联影」这类两字简称是口语常态）


@dataclass(frozen=True)
class Company:
    stock_code: str
    security_name: str
    full_name: str
    aliases: tuple[str, ...]

    def __repr__(self) -> str:
        return f"<Company {self.stock_code} {self.security_name}>"


def _derive_aliases(security_name: str, full_name: str) -> set[str]:
    """只做两种安全的派生：
       * 全称剥法人词尾（北京万东医疗科技股份有限公司 → 北京万东医疗科技）
       * 证券简称剥行业词尾（万东医疗 → 万东；基蛋生物 → 基蛋）
    故意**不**砍前缀 —— 砍前缀产出的是「华医疗器械」「医疗器械」这种碎片，
    既是行业通名又会跨公司撞车。
    """
    out: set[str] = set()
    for name in (security_name, full_name):
        s = name
        for suf in LEGAL_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        if s and len(s) >= MIN_ALIAS_LEN:
            out.add(s)
    for suf in INDUSTRY_SUFFIXES:
        if security_name.endswith(suf):
            base = security_name[: -len(suf)]
            if len(base) >= MIN_ALIAS_LEN:
                out.add(base)
            break
    return out


@lru_cache(maxsize=1)
def load_companies() -> tuple[Company, ...]:
    """读 manifest.csv 构建公司表（缓存）。

    最后一步做**全局唯一性过滤**：某个别名若对应多家公司，整条丢弃。
    这是防「医疗器械」这类通名误伤的关键闸门。
    """
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"缺少 {MANIFEST}，请先跑 src/ingest/fetch_announcements.py")
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    raw: dict[str, tuple[str, str, str, set[str]]] = {}   # code → (sec, full, listed, aliases)
    for r in rows:
        code = (r.get("stock_code") or "").strip()
        if not code or code in raw:
            continue
        sec = (r.get("security_name") or "").strip()
        full = (r.get("company_full_name_official") or "").strip()
        listed = (r.get("company") or "").strip()
        aliases = {sec, full, listed, code}
        aliases |= _derive_aliases(sec, full)
        raw[code] = (sec, full, listed, {a for a in aliases if len(a) >= MIN_ALIAS_LEN})

    # 别名 → 拥有它的公司数；>1 家的别名是通名，全部丢弃
    owner: dict[str, set[str]] = {}
    for code, (_, _, _, aliases) in raw.items():
        for a in aliases:
            owner.setdefault(a, set()).add(code)
    shared = {a for a, codes in owner.items() if len(codes) > 1}

    companies = []
    for code, (sec, full, _listed, aliases) in raw.items():
        keep = tuple(sorted((a for a in aliases if a not in shared),
                            key=len, reverse=True))
        companies.append(Company(code, sec, full, keep))
    companies.sort(key=lambda c: c.stock_code)
    if shared:
        print(f"[entities] 丢弃 {len(shared)} 个跨公司共用别名："
              f"{sorted(shared)[:8]}{' …' if len(shared) > 8 else ''}",
              file=sys.stderr)
    return tuple(companies)


def find_companies(question: str) -> list[Company]:
    """返回问题里提到的公司，按出现位置排序；没提到就返回空列表。

    用「吃掉已匹配片段」的办法实现最长优先：替换成等长占位符（长度不变，
    所以后续匹配的下标仍与原文一致）。只在真的命中时才重拼字符串。
    """
    q = question or ""
    masked = q
    found: list[tuple[int, Company]] = []
    hit_codes: set[str] = set()

    pairs = [(a, c) for c in load_companies() for a in c.aliases]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    for alias, comp in pairs:
        if comp.stock_code in hit_codes:
            continue
        pos = masked.find(alias)
        if pos < 0:
            continue
        masked = masked[:pos] + "\x00" * len(alias) + masked[pos + len(alias):]
        hit_codes.add(comp.stock_code)
        found.append((pos, comp))

    found.sort(key=lambda f: f[0])
    return [c for _, c in found]


def main() -> int:
    comps = load_companies()
    print(f"公司 {len(comps)} 家，别名总数 {sum(len(c.aliases) for c in comps)}")
    for c in comps:
        extra = [a for a in c.aliases if a not in (c.security_name, c.full_name)]
        print(f"  {c.stock_code}  {c.security_name:<6} "
              f"派生别名 {'/'.join(extra[:4]) or '无'}")

    cases = [
        ("联影医疗 2026 上半年营业收入是多少？", 1),
        ("对比心脉医疗和惠泰医疗的研发投入强度", 2),
        ("哪些公司上半年归母净利润同比下滑？下滑原因是什么？", 0),
        ("新华医疗的高值耗材业务板块毛利率是多少？", 1),
        ("联影和心脉哪家毛利率更高？", 2),                    # 两字简称
        ("上海微创心脉医疗科技(集团)股份有限公司的营收", 1),    # 官方全称
        ("医疗器械板块哪家研发投入最高？", 0),                 # 通名，必须认不出
        ("688271 的归母净利润是多少？", 1),                    # 股票代码
    ]
    bad = 0
    for q, want in cases:
        got = find_companies(q)
        ok = len(got) == want
        bad += 0 if ok else 1
        print(f"{'[OK]  ' if ok else '[FAIL]'} {q}\n        "
              f"→ {[f'{c.stock_code} {c.security_name}' for c in got] or '未识别到公司'}"
              f"（期望 {want} 家）")
    print(f"\n{len(cases) - bad}/{len(cases)} 例符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
