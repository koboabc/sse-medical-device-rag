"""在 chunk 里按「公司 / 关键词 / 章节」定位内容 —— 出题与核对 gold 用的镊子。

为什么需要它：评测集的 `gold_chunk_ids` 不能靠猜。标注的流程必须是
「读年报 → 找到那句话 → 记下 chunk_id 与页码 → 写进 qa.jsonl」，而这个流程
要一个能**把关键词在块内的上下文打出来**的工具：只给 chunk_id 看不出数字对不对，
只给全文又太长。所以这里默认打「关键词命中处 ±120 字」的窗口，`--full` 才打全文。

用法：
    # 某公司里含「营业收入」的块，只看财务报告那节
    python scripts/grep_chunks.py -c 688271 -k 营业收入 -s 第八节 -n 5
    # 多个关键词是 AND
    python scripts/grep_chunks.py -c 心脉医疗 -k 研发投入 -k 营业收入 -s 第三节
    # 直接读全文（核对表格里的具体数字时用）
    python scripts/grep_chunks.py -i 688016-0024 --full
    # 公司名/代码都能认；-c 也可给简称（走 entities 的别名表）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.entities import load_companies  # noqa: E402
from src.retrieval.store import store  # noqa: E402

WINDOW = 120


def snippet(content: str, keywords: list[str], width: int) -> str:
    """在第一个命中的关键词周围取窗口；没给关键词就取开头。"""
    low = content.lower()
    pos = -1
    for k in keywords:
        p = low.find(k.lower())
        if p >= 0:
            pos = p
            break
    if pos < 0:
        return content[:width * 2]
    half = width // 2
    a = max(0, pos - half)
    b = min(len(content), a + width * 2)
    return ("…" if a else "") + content[a:b].replace("\n", " / ") + ("…" if b < len(content) else "")


def resolve_codes(specs: list[str]) -> list[str] | None:
    """-c 接代码或名字（简称/全称/别名），统一解析成 stock_code。"""
    if not specs:
        return None
    comps = load_companies()
    out: list[str] = []
    for s in specs:
        t = s.strip()
        if not t:
            continue
        hit = [c for c in comps if t in c.aliases or t == c.stock_code]
        if not hit:
            raise SystemExit(f"认不出公司「{t}」。可用："
                             + "、".join(f"{c.stock_code} {c.security_name}" for c in comps))
        out.extend(c.stock_code for c in hit)
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser(description="在 chunk 里按公司/关键词/章节定位")
    ap.add_argument("-c", "--company", action="append", default=[],
                    help="公司代码或名字，可重复（多个是 OR）")
    ap.add_argument("-k", "--keyword", action="append", default=[],
                    help="内容关键词，可重复（多个是 AND，忽略大小写）")
    ap.add_argument("-s", "--section", default=None, help="章节前缀，如 第三节")
    ap.add_argument("-t", "--type", default=None, choices=["table", "text"],
                    help="内容类型")
    ap.add_argument("-i", "--id", default=None, help="按 chunk_id 精确定位")
    ap.add_argument("-n", "--limit", type=int, default=20, help="最多打印多少条")
    ap.add_argument("--full", action="store_true", help="打印整块正文（默认只打命中窗口）")
    ap.add_argument("--width", type=int, default=WINDOW, help="窗口半宽字数")
    ap.add_argument("--ids-only", action="store_true", help="只输出 chunk_id（便于喂给评测）")
    ap.add_argument("--json", default=None, help="命中的 chunk_id 写成 json 数组")
    args = ap.parse_args()

    st = store()
    df = st.df
    m = df.index.to_numpy() >= 0          # 全 True 的布尔掩码，方便逐条 AND
    codes = resolve_codes(args.company)
    if codes:
        m &= df["stock_code"].isin(codes).to_numpy()
    if args.section:
        m &= df["section"].str.startswith(args.section).to_numpy()
    if args.type:
        m &= (df["content_type"] == args.type).to_numpy()
    if args.id:
        m &= (df["chunk_id"] == args.id).to_numpy()
    for k in args.keyword:
        m &= df["content"].str.contains(k, case=False, regex=False).to_numpy()

    hit_rows = df.index[m].to_numpy()
    print(f"命中 {len(hit_rows)} 块"
          f"（公司 {codes or '全部'}；关键词 {args.keyword or '无'}；"
          f"章节 {args.section or '全部'}；类型 {args.type or '全部'}）")

    # 关键词多的块信息量大，按命中数降序排，先看最相关的
    def score(row: int) -> int:
        c = df.iloc[row]["content"]
        return -sum(c.count(k) for k in args.keyword)

    rows = sorted(hit_rows.tolist(), key=score)[: args.limit]
    for row in rows:
        r = st.record(row)
        print(f"\n[{r['chunk_id']}] {r['stock_code']} {r['company'][:16]} | "
              f"{r['section']} | {r['subsection'] or '—'} | "
              f"p{r['page_start']}-{r['page_end']} | {r['content_type']} | "
              f"{r['char_count']}字 | row={row}")
        if args.ids_only:
            continue
        body = r["content"] if args.full else snippet(r["content"], args.keyword, args.width)
        if args.full:
            print("    " + body.replace("\n", "\n    "))
        else:
            print("    " + body)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        ids = [st.record(row)["chunk_id"] for row in rows]
        p.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        print(f"\nchunk_id 已写出 → {p}")
    return 0 if hit_rows.any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
