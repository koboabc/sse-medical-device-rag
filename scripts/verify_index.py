"""校验两套索引：形状、映射、归一化，以及**写回行号没串位**。

最关键的一条是「写回没串位」：编码是按 token 长度排序的，算完要按行号
写回 memmap。一旦排列写错，向量和 chunk_id 就对不上，而这种错不会报错，
只会让检索结果整体错乱 —— 所以必须真的把某几行重新编码一遍比对。

用法：
    python scripts/verify_index.py                # 全量校验
    python scripts/verify_index.py --resample 16  # 重编码抽查条数
    python scripts/verify_index.py --index smoke  # 校验 --limit 冒烟留下的产物
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import config  # noqa: E402
from src.indexing.build_index import load_chunks  # noqa: E402

FAISS_DIR = config.index_dir / "faiss"
BM25_DIR = config.index_dir / "bm25"

OK, BAD = "  ✓", "  ✗"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print((OK if cond else BAD) + " " + msg)
    if not cond:
        fails.append(msg)


def verify_bm25(df: pd.DataFrame, ids: list[str]) -> None:
    print("\n[2/2] BM25 索引")
    bm_p, tk_p = BM25_DIR / "bm25.pkl", BM25_DIR / "corpus_tokens.pkl"
    b_ids_p = BM25_DIR / "chunk_ids.json"
    for p in (bm_p, tk_p, b_ids_p, BM25_DIR / "chunk_map.parquet"):
        check(p.exists(), f"存在 {p.name}")

    bm = pickle.loads(bm_p.read_bytes())
    corpus = pickle.loads(tk_p.read_bytes())
    b_ids = json.loads(b_ids_p.read_text(encoding="utf-8"))
    check(len(bm.doc_freqs) == len(df), f"BM25 文档数 {len(bm.doc_freqs)} == {len(df)}")
    check(len(corpus) == len(df), f"分词语料 {len(corpus)} == {len(df)}")
    check(b_ids == ids, "BM25 与 FAISS 的 chunk_ids 完全一致（两路结果可对齐融合）")
    check(all(len(t) > 0 for t in corpus), "没有空分词文档")
    check(not any("|" in t for t in corpus[:2000]), "分词已抹掉 Markdown 管道符")

    from src.indexing.build_index import tokenize

    qs = "联影医疗 营业收入"
    sc = bm.get_scores(tokenize(qs))
    top = np.argsort(-sc)[:5]
    print(f"  BM25 查询「{qs}」top5：")
    for i in top:
        r = df.iloc[int(i)]
        print(f"    {sc[i]:.3f}  {r['stock_code']} {r['company']} | "
              f"{r['section']} | p{r['page_start']} | {r['content_type']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", type=int, default=8,
                    help="重新编码多少条来验证写回行号（覆盖长/中/短）")
    ap.add_argument("--index", choices=["full", "smoke"], default="full",
                    help="smoke = 上次 --limit 冒烟测试留下的产物")
    args = ap.parse_args()

    df = load_chunks()
    if args.index == "smoke":
        # 冒烟产物在独立目录，且只覆盖按长度分层抽样的那几十条
        global FAISS_DIR
        FAISS_DIR = config.index_dir / "_smoke" / "faiss"
        ids_smoke = json.loads((FAISS_DIR / "chunk_ids.json").read_text("utf-8"))
        df = df[df["chunk_id"].isin(ids_smoke)].reset_index(drop=True)
    print(f"metadata.parquet: {len(df)} 条（{args.index}）\n")

    # ---------------- FAISS ----------------
    print("[1/2] 向量索引")
    import faiss

    idx_p = FAISS_DIR / "faiss.index"
    npy_p = FAISS_DIR / "embeddings.npy"
    ids_p = FAISS_DIR / "chunk_ids.json"
    map_p = FAISS_DIR / "chunk_map.parquet"
    for p in (idx_p, npy_p, ids_p, map_p):
        check(p.exists(), f"存在 {p.name}")

    from src.indexing.build_index import faiss_read

    index = faiss_read(idx_p)     # 不是 faiss.read_index：中文路径它打不开
    vecs = np.load(npy_p)
    ids = json.loads(ids_p.read_text(encoding="utf-8"))
    cmap = pd.read_parquet(map_p)
    full = args.index == "full"

    if full:
        check(index.ntotal == len(df), f"faiss.ntotal {index.ntotal} == chunk 数 {len(df)}")
        check(vecs.shape == (len(df), config.embedding_dim),
              f"embeddings.npy 形状 {vecs.shape} == ({len(df)}, {config.embedding_dim})")
    check(index.ntotal == len(ids), f"ntotal {index.ntotal} == chunk_ids.json {len(ids)}")
    check(len(ids) == len(cmap), f"chunk_ids {len(ids)} == chunk_map.parquet {len(cmap)}")
    check(ids == df["chunk_id"].tolist()[:len(ids)],
          "chunk_ids.json 与 metadata 行序一致（两套索引共用同一行序）")
    check(cmap["chunk_id"].tolist() == ids, "chunk_map.parquet 与 chunk_ids.json 同序")
    check((cmap["row"] == np.arange(len(cmap))).all(), "chunk_map.row == 0..n-1")

    if not full:
        check(index.ntotal == args.resample or index.ntotal <= len(df),
              f"冒烟产物 ntotal={index.ntotal}")

    # 归一化：IndexFlatIP 只有在向量 L2=1 时才等价余弦
    norms = np.linalg.norm(vecs, axis=1)
    check(np.allclose(norms, 1.0, atol=1e-4),
          f"向量已 L2 归一化（norm min {norms.min():.6f} max {norms.max():.6f}）")
    check(np.isfinite(vecs).all(), "向量无 NaN/Inf")
    check(not np.allclose(vecs[0], vecs[-1]), "首末向量不相同（不是占位零向量）")

    import faiss as _f
    ip = _f.IndexFlatIP(index.d)
    ip.add(vecs)
    d1, i1 = ip.search(vecs[:5], 1)
    check((i1.ravel() == np.arange(5)).all() and np.allclose(d1.ravel(), 1.0, atol=1e-3),
          "自检索：前 5 条各自最近邻是自己且相似度 ≈1（索引与 npy 一致）")

    # ---- 写回行号校验：重新编码，看是否落在正确的行 ----
    from src.indexing.embedder import Embedder

    emb = Embedder(device=config.embedding_device or None)
    n = min(args.resample, len(df))
    # 沿长度谱均匀取，长的短的都要覆盖 —— 排序编码最可能出错的就是两端
    order = df["char_count"].sort_values().index.to_numpy()
    pick = order[np.linspace(0, len(order) - 1, n).astype(int)]
    texts = df.loc[pick, "content"].tolist()
    fresh = emb.encode_documents(texts, batch_size=1, show_progress=False)

    print(f"\n  重编码 {n} 条（字符数 "
          f"{df.loc[pick, 'char_count'].min()}~{df.loc[pick, 'char_count'].max()}）")
    worst = 1.0
    for k, row in enumerate(pick):
        a, b = fresh[k], vecs[row]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = min(worst, cos)
        flag = OK if cos > 0.9999 else BAD
        if cos <= 0.9999:
            fails.append(f"行 {row} 写回串位")
        print(f"{flag} 行 {row:>5} {ids[row]:>12} {df.loc[row, 'char_count']:>6}字  "
              f"cos(重编码, 已存) = {cos:.6f}")
    check(worst > 0.9999,
          f"写回行号正确：{n} 条重编码与已存向量最低 cos = {worst:.6f}")

    # 查询侧：必须走 prompt_name="query"
    q = emb.encode_queries(["联影医疗的营业收入是多少"])
    check(q.shape == (1, config.embedding_dim), f"查询编码形状 {q.shape}")
    check(abs(float(np.linalg.norm(q[0])) - 1.0) < 1e-5, "查询向量已归一化")
    check(float(q[0] @ q[0]) > 0, "查询向量非零")
    check("query" in emb.prompts, "模型自带 query prompt（查询侧非对称编码）")

    D, I = index.search(q, 5)
    print("\n  查询「联影医疗的营业收入是多少」向量索引 top5：")
    for s, i in zip(D[0], I[0]):
        r = df.iloc[int(i)]
        print(f"    {s:.4f}  {r['stock_code']} {r['company']} | "
              f"{r['section']} | p{r['page_start']} | {r['content_type']}")

    # ---------------- BM25 ----------------
    if args.index == "smoke":
        print("\n[2/2] BM25 索引：冒烟模式跳过（冒烟只动向量，没碰 BM25）")
    else:
        verify_bm25(df, ids)

    print("\n" + "=" * 60)
    if fails:
        print(f"  {len(fails)} 项未通过：")
        for m in fails:
            print(f"    - {m}")
        return 1
    print("  全部通过")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
