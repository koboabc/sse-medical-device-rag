"""检索侧自检：行序契约、两路一致性、掩码与融合的不变量。

`verify_index.py` 校验的是**索引产物本身**（形状、归一化、写回没串位）；
本脚本校验的是**检索运行时**——即 `store.py / retriever.py` 用这些产物的方式。
两者不重叠：索引是对的，检索层仍可能拿错行（`.loc` 与 `.iloc` 之差）或用错
查询编码（漏掉 `prompt_name="query"`），而这两种错都**不报错**。

最要紧的一条是 `chunk_map.parquet` 当作独立参照去核对 `store.record()`：
chunk_map 是构建期写的，与运行期读的 metadata 是两条路径，二者对上才说明
「行号 → 记录」没串位。

用法：
    python scripts/verify_retrieval.py
    python scripts/verify_retrieval.py --resample 4   # 向量重编码抽查条数
                                             （向量索引不存在时该项跳过）
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import config  # noqa: E402
from src.indexing.build_index import META, doc_texts  # noqa: E402
from src.retrieval.entities import find_companies, load_companies  # noqa: E402
from src.retrieval.retriever import Weights, retrieve  # noqa: E402
from src.retrieval.store import store  # noqa: E402

OK, BAD, SKIP = "  [OK]  ", "  [FAIL]", "  [SKIP]"
fails: list[str] = []


def check(cond: bool, msg: str) -> bool:
    print((OK if cond else BAD) + " " + msg)
    if not cond:
        fails.append(msg)
    return bool(cond)


def skip(msg: str) -> None:
    print(SKIP + " " + msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", type=int, default=4,
                    help="向量重编码抽查条数（沿长度谱取，覆盖长块与短块）")
    ap.add_argument("--index", default=None,
                    help="索引目录（默认 config.index_dir）。指向 index/header 时"
                         "会按「表头+正文」重编码 —— 用正文去比表头索引的向量会"
                         "得到一堆极低 cos，那是比错了文本、不是索引坏了")
    args = ap.parse_args()

    st = store(args.index)
    header = st.index_dir.name == "header"
    df = st.df
    print(f"chunks {len(df)}  |  {st.describe()}\n")

    # ---------------- 1. 行序契约 ----------------
    print("[1/5] 行序契约（行号 → 记录）")
    cmap = pd.read_parquet(st.bm25_dir / "chunk_map.parquet")
    check(len(cmap) == len(df), f"chunk_map {len(cmap)} 行 == metadata {len(df)} 行")
    check(cmap["chunk_id"].tolist() == df["chunk_id"].tolist(),
          "chunk_map.chunk_id 与 load_chunks() 顺序完全一致")
    check((cmap["row"] == np.arange(len(cmap))).all(), "chunk_map.row == 0..n-1")

    # 独立核对 store.record()：拿 chunk_map 里记的 (row, chunk_id) 去反查，
    # 再用 metadata 的 content 校验正文前 40 字，确认没有取错行。
    pick = [0, len(df) // 3, len(df) // 2, len(df) - 1]
    ok = True
    for row in pick:
        want_id = cmap.iloc[row]["chunk_id"]
        got = st.record(row)
        if got["chunk_id"] != want_id or not got["content"].startswith(
                df.iloc[row]["content"][:40]):
            ok = False
            print(f"       行 {row}: 期望 {want_id}，实得 {got['chunk_id']}")
    check(ok, f"store.record() 抽查 {pick} 行与 chunk_map 一致（含正文首 40 字）")

    # 真正的坑不是 .loc/.iloc（load_chunks 里有 reset_index，两者今天等价），
    # 而是**绕开 load_chunks 直接 read_parquet**：parquet 的物理行序与排序后的
    # 行序不同，谁按位置取谁错位。这条断言把「物理序 ≠ 契约序」钉死。
    check((df.index.to_numpy() == np.arange(len(df))).all(),
          "load_chunks() 已 reset_index，行号 == 位置（.iloc 即 .loc）")
    raw = pd.read_parquet(META, columns=["chunk_id"])
    check(raw["chunk_id"].iloc[0] != df["chunk_id"].iloc[0],
          f"parquet 物理首行 {raw['chunk_id'].iloc[0]} ≠ 契约首行 "
          f"{df['chunk_id'].iloc[0]} —— 绕过 load_chunks() 直接按位置取必错位")
    check(set(raw["chunk_id"]) == set(df["chunk_id"]), "两种行序的内容是同一批 chunk")

    # ---------------- 2. BM25 ----------------
    print("\n[2/5] BM25 路")
    check(len(st.bm25.doc_freqs) == len(df), f"BM25 文档数 {len(st.bm25.doc_freqs)} == {len(df)}")
    check(st.bm25_ids == df["chunk_id"].tolist(), "BM25 chunk_ids 与 metadata 同序")
    check(all(len(t) > 0 for t in st.corpus_tokens), "无空分词文档")

    # ---------------- 3. 向量路 ----------------
    print("\n[3/5] 向量路")
    if not st.ensure_vector():
        skip(f"向量索引不可用，跳过向量相关检查（{st.faiss_dir / 'faiss.index'} 不存在或不一致）")
        skip("索引建好后重跑本脚本即可补上这些检查")
    else:
        vecs = np.load(st.faiss_dir / "embeddings.npy", mmap_mode="r")
        check(st._index.ntotal == len(df), f"faiss.ntotal {st._index.ntotal} == {len(df)}")
        check(vecs.shape == (len(df), config.embedding_dim),
              f"embeddings.npy 形状 {tuple(vecs.shape)}")
        check(st._vec_ids == st.bm25_ids,
              "FAISS 与 BM25 的 chunk_ids 完全一致（两路可对齐融合）")
        norms = np.linalg.norm(np.asarray(vecs[:2000], dtype=np.float32), axis=1)
        check(np.allclose(norms, 1.0, atol=1e-4),
              f"向量已 L2 归一化（前 2000 条 norm {norms.min():.4f}~{norms.max():.4f}）")

        from src.indexing.embedder import Embedder

        emb = st.embedder
        order = df["char_count"].sort_values().index.to_numpy()
        n = min(args.resample, len(df))
        rows = order[np.linspace(0, len(order) - 1, n).astype(int)]
        # 重编码必须用**建索引时的同一套文本**：表头索引的向量是在
        # 「表头+正文」上算的，拿正文去比会得到一堆很低的 cos —— 那是比错了
        # 输入，不是索引串位。用 doc_texts + 同一个 header 开关，否则这个
        # 检查在变体索引上会给出假故障。
        import types
        docs = doc_texts(df.loc[rows].reset_index(drop=True),
                         types.SimpleNamespace(header=header))
        fresh = emb.encode_documents(docs, batch_size=1, show_progress=False)
        worst = 1.0
        for k, row in enumerate(rows):
            a, b = fresh[k], np.asarray(vecs[row], dtype=np.float32)
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            worst = min(worst, cos)
            print(f"       行 {row:>5} {st.bm25_ids[row]:>12} "
                  f"{df.loc[row, 'char_count']:>6}字  cos={cos:.6f}")
        check(worst > 0.9999, f"向量行号未串位：{n} 条重编码最低 cos = {worst:.6f}")
        check("query" in emb.prompts, "模型自带 query prompt（查询侧非对称编码）")

    # ---------------- 4. 掩码（显式硬过滤） ----------------
    print("\n[4/5] 硬过滤掩码")
    allow = st.allow_mask(["688271"])
    check(allow is not None and allow.sum() > 0,
          f"allow_mask(['688271']) 命中 {int(allow.sum()) if allow is not None else 0} 块")
    hb = st.search_bm25("营业收入", k=10, allow=allow)
    check(bool(hb) and all(df.iloc[h.row]["stock_code"] == "688271" for h in hb),
          "BM25 掩码后只返回被允许公司的块")
    # 掩码只能**筛**，不能**改分**：同一条块在被掩码与不被掩码时分数必须一样
    full = {h.row: h.score for h in st.search_bm25("营业收入", k=50)}
    same = all(abs(full[h.row] - h.score) < 1e-9 for h in hb if h.row in full)
    check(same, "掩码不改变块本身的 BM25 分（只是把别的公司置 -inf）")
    if st.has_vector:
        hv = st.search_vector("营业收入", k=10, allow=allow)
        check(bool(hv) and all(df.iloc[h.row]["stock_code"] == "688271" for h in hv),
              "向量掩码后只返回被允许公司的块")
    else:
        skip("向量掩码检查（无向量索引）")

    # ---------------- 5. 融合不变量 ----------------
    print("\n[5/5] 融合与加权的不变量")
    q = "心脉医疗的研发投入占营业收入比例"
    res = retrieve(q, top_k=5)
    check(len(res.hits) <= 5, f"Top-K 截断生效（{len(res.hits)} ≤ 5）")
    check([h.rank for h in res.hits] == list(range(1, len(res.hits) + 1)),
          "命中 rank 连续从 1 开始")
    finals = [h.scores["final"] for h in res.hits]
    check(finals == sorted(finals, reverse=True), "命中按 final 降序排列")
    check(all(h.scores["weight"] >= 1.0 - 1e-9 for h in res.hits),
          f"软加权只放大不缩小（最小 weight {min(h.scores['weight'] for h in res.hits)}）")
    codes = {c.stock_code for c in res.companies}
    check(codes == {"688016"}, f"问题里识别出心脉医疗：{sorted(codes)}")

    # 同内容去重：被合并的块必须与保留的块正文完全相同
    dup_ok = True
    for h in res.hits:
        for d in h.dup_chunk_ids:
            same = st.df.loc[st.df["chunk_id"] == d, "content"].iloc[0] == \
                st.df.iloc[h.row]["content"]
            dup_ok &= bool(same)
    check(dup_ok, "被合并的同内容块正文与保留块一致（去重没有误伤）")

    # 硬过滤端到端
    res2 = retrieve("营业收入是多少", company=["688271"], top_k=5)
    check(bool(res2.hits) and all(df.iloc[h.row]["stock_code"] == "688271"
                                  for h in res2.hits),
          "--company 硬过滤端到端：结果全属指定公司")

    # 消融开关：关掉加权后，final 必须等于 rrf，且排序退化成纯 RRF 序
    b = retrieve(q, top_k=5, weights=Weights(1.0, 1.0, 1.0, 1.0, 1.0))
    check(all(h.scores["weight"] == 1.0 for h in b.hits), "全 1.0 权重下 weight 恒为 1")
    check(all(abs(h.scores["final"] - h.scores["rrf"]) < 1e-9 for h in b.hits),
          "关掉加权后 final == rrf（权重是唯一的分数修正项）")
    rrfs = [h.scores["rrf"] for h in b.hits]
    check(rrfs == sorted(rrfs, reverse=True), "无权重时结果严格按 RRF 降序")

    print(f"\n公司表 {len(load_companies())} 家，"
          f"「联影和心脉哪家毛利率更高」识别到 "
          f"{[c.security_name for c in find_companies('联影和心脉哪家毛利率更高')]}")
    print(f"\n{'全部通过' if not fails else f'{len(fails)} 项未通过'}")
    for f in fails:
        print(f"  ✗ {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
