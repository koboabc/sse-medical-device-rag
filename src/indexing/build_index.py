"""根据 data/metadata.parquet 建立两套检索索引。

  一、向量索引  index/faiss/   Qwen3-Embedding-0.6B（本地，不走在线 API）
  二、BM25 索引  index/bm25/    jieba 分词 + rank_bm25

设计要点：
  * 文档向量只算一次：向量以 npy（memmap）落盘并记录进度，
    重跑会从断点续算，绝不重复编码。CPU 上 0.6B 模型耗时以小时计，
    崩溃后从零开始是不可接受的。
  * 查询编码走 prompt_name="query"（见 embedder.py），与文档侧非对称
  * 两套索引共用同一行序（= metadata.parquet 的行序），
    各自目录下都保存 chunk_id ↔ 行号 的映射

用法：
    python src/indexing/build_index.py --probe      # 先测速，只编码 32 条
    python src/indexing/build_index.py              # 全量
    python src/indexing/build_index.py --only bm25  # 只建 BM25
    python src/indexing/build_index.py --force      # 丢弃已有向量，重算
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

FAISS_DIR = config.index_dir / "faiss"
BM25_DIR = config.index_dir / "bm25"
META = config.data_dir / "metadata.parquet"

EMB_NPY = FAISS_DIR / "embeddings.npy"
EMB_STATE = FAISS_DIR / "embeddings.state.json"
FAISS_INDEX = FAISS_DIR / "faiss.index"
BM25_PKL = BM25_DIR / "bm25.pkl"
BM25_TOKENS = BM25_DIR / "corpus_tokens.pkl"

# 表格 Markdown 的管道与分隔行对 BM25 是纯噪声，先抹掉
PIPE_RE = re.compile(r"[-:\s]*\|[-:\s|]*")


def faiss_write(index, path: Path) -> None:
    """落盘 FAISS 索引。

    不用 faiss.write_index()：它在 Windows 上走窄字符 fopen，路径含中文
    （本项目目录就是 D:\\复旦\\...）会直接报 "No such file or directory"，
    且目录其实存在。绕开办法是序列化成字节，用 Python 的文件 I/O 写，
    它走 UTF-16 路径，中文没问题。
    """
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(faiss.serialize_index(index).tobytes())


def faiss_read(path: Path):
    import faiss

    return faiss.deserialize_index(
        np.frombuffer(path.read_bytes(), dtype=np.uint8))


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def load_chunks() -> pd.DataFrame:
    if not META.exists():
        raise FileNotFoundError(f"缺少 {META}，请先跑 src/indexing/parse_reports.py")
    df = pd.read_parquet(META)
    df = df.sort_values("chunk_id").reset_index(drop=True)
    assert df["chunk_id"].is_unique, "chunk_id 有重复"
    assert (df["char_count"] > 0).all(), "存在空 chunk"
    return df


# ============================================================
# 文档文本：正文 与 正文+元数据表头
# ============================================================
# 表头用到的列。顺序即拼接顺序，改动会使已建索引的输入口径变化 → 必须重算。
HEADER_COLS = ("company", "stock_code", "report", "section", "subsection")


def make_headers(df: pd.DataFrame) -> list[str]:
    """每条 chunk 的元数据表头，形如：
    「上海联影医疗科技股份有限公司 688271 2026年半年度报告 第二节 ... 六、公司主要会计数据和财务指标。」

    为什么要把元数据拼进正文（`--header`）：
    这些信息在**表格块里根本不存在**。17 家公司的「主要会计数据」表正文几乎
    同构、都不含公司名，于是 BM25 与向量两路都无法分辨哪张表是谁的 —— 实测
    「联影医疗 2026 年上半年营业收入」的 gold（`688271-0009`）在 BM25 全库排
    #795、向量 #541，而两路的第 1 名都不含「营业收入」三个字。

    表头取的是**文档自带的真实元数据**，不是关键词规则、不是同义词扩展，所以
    它不引入任何猜测，也不需要维护词表 —— 这是它与「再加一路表头索引」那个
    已被实测推翻的方案（见 retriever.py 注释）的本质区别：那个是**另建一路**
    去跟正文路抢 RRF 名额，这个是把元数据还原进文档本身的向量/词频里。
    """
    s = df[list(HEADER_COLS)].fillna("").astype(str).agg(" ".join, axis=1)
    # subsection 有 274 条是空串，会产生连续空格；归一化掉，免得 jieba 切出空 token
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return (s + "。").tolist()


def doc_texts(df: pd.DataFrame, args) -> list[str]:
    """建索引用文本。`--header` 时 = 元数据表头 + 正文，否则就是正文。

    注意 `df["content"]` 本身**从不改动**：页面要显示原文、评测要核对原文、
    引用要给原文。表头只进索引，不进展示。
    """
    texts = df["content"].tolist()
    if not getattr(args, "header", False):
        return texts
    return [f"{h}\n{t}" for h, t in zip(make_headers(df), texts)]


# ============================================================
# 一、向量索引
# ============================================================
def make_batches(order: np.ndarray, lens: np.ndarray, batch_size: int,
                 token_budget: int) -> list[np.ndarray]:
    """把编码顺序切成批：批内条数 ≤ batch_size，且 最长条 × 条数 ≤ token_budget。

    order 已按长度降序，所以「批内最长」= 批内第一条，长度天然接近。
    token_budget 是防 OOM 的闸门：最后那几批每条都是 4000+ token，
    再按 batch_size 硬凑会撑爆内存。
    """
    batches: list[np.ndarray] = []
    cur: list[int] = []
    for i in order:
        n = int(lens[i])
        m = n if not cur else max(int(lens[cur[0]]), n)
        if cur and (len(cur) >= batch_size or m * (len(cur) + 1) > token_budget):
            batches.append(np.asarray(cur, dtype=np.int64))
            cur = []
            m = n
        cur.append(int(i))
    if cur:
        batches.append(np.asarray(cur, dtype=np.int64))
    return batches


def plan_order(emb, texts: list[str], max_seq_length: int | None,
               batch_size: int, token_budget: int):
    """按 token 长度降序排序 → 分桶成批。返回 (lens, order, batches, 指纹)。

    CPU 上 padding 是实打实的算力：顺序取批会把一条 8913 token 的表格块
    和 31 条几百 token 的短块凑成一批，整批 pad 到 8913，有效算力只剩 1/3.5。
    按长度排序后批内长度接近，padding 开销降到 ~1.1x。向量本身完全不变，
    只是编码次序变了，算完按行号写回。
    """
    lens = emb.token_lengths(texts, max_seq_length)
    order = np.argsort(-lens, kind="stable")
    batches = make_batches(order, lens, batch_size, token_budget)
    fp = hashlib.sha1(order.tobytes()).hexdigest()[:16]
    return lens, order, batches, fp


def open_embeddings(total: int, dim: int, force: bool, fp: str,
                    cap: int | None = None):
    """打开（或新建）向量 memmap，返回 (数组, 已完成的批数)。"""
    arr = None
    if not force and EMB_NPY.exists() and EMB_STATE.exists():
        try:
            st = json.loads(EMB_STATE.read_text(encoding="utf-8"))
            arr = np.load(EMB_NPY, mmap_mode="r+")
            if (arr.shape == (total, dim) and st.get("dim") == dim
                    and st.get("total") == total and st.get("order_fp") == fp):
                bdone = int(st.get("batches_done", 0))
                if 0 < bdone:
                    print(f"  ↩ 断点续算：已完成 {bdone} 批"
                          f"（{st.get('rows_written', 0)}/{total} 行）")
                return arr, bdone
            # 指纹不符最常见的原因就是改了截断上限：长度全变 → 长度降序的顺序变。
            # 这时旧向量**不是「重复计算」而是错的**（它们是按另一种输入口径算的），
            # 尤其已完成的那些正好是长块，受截断影响最大。所以必须作废，但要说清。
            old_cap = (st.get("model") or {}).get("max_seq_length")
            if cap and old_cap and int(old_cap) != int(cap):
                print(f"  ! 截断上限变了（{old_cap} → {cap}），批计划指纹随之改变；"
                      f"已有 {st.get('rows_written', 0)} 行是按旧口径算的，"
                      f"与新口径不在同一向量空间，作废重算")
            else:
                print("  ! 已有向量与本次不符（形状、维度或编码顺序变了），重建")
        except Exception as e:  # noqa: BLE001
            print(f"  ! 读取已有向量失败（{type(e).__name__}: {e}），重建")

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    # 把「写哪个目录」打出来。向量算一次以小时计，而写错目录的代价是**抹掉
    # 一份已有的、算了一下午的向量**（下面那两行 unlink 是无条件的）。
    # 一行日志换一次确认，值。
    print(f"  新建向量 memmap → {EMB_NPY}（{total} × {dim} float32 "
          f"= {human(total * dim * 4)}）")
    # Windows 不允许删掉仍被 mmap 打开的文件（WinError 32），所以重建前必须先
    # 释放上面那个 memmap。少这一句，凡是走到「已有向量作废重算」就会崩在
    # unlink 上 —— 而改截断上限恰好必然走这条路（实测踩到）。
    arr = None
    gc.collect()
    for p in (EMB_NPY, EMB_STATE):
        p.unlink(missing_ok=True)
    arr = np.lib.format.open_memmap(EMB_NPY, mode="w+", dtype="float32",
                                    shape=(total, dim))
    return arr, 0


def save_state(rows: int, dim: int, total: int, info: dict, **extra) -> None:
    EMB_STATE.write_text(json.dumps(
        {"rows_written": rows, "dim": dim, "total": total, "model": info, **extra},
        ensure_ascii=False, indent=2), encoding="utf-8")


def build_faiss(df: pd.DataFrame, args) -> dict:
    import faiss

    from src.indexing.embedder import Embedder

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(df)
    texts = doc_texts(df, args)
    if getattr(args, "header", False):
        print(f"  文本 = 元数据表头 + 正文（{len(HEADER_COLS)} 列："
              f"{' '.join(HEADER_COLS)}）；展示用的 df['content'] 不变")
        print(f"  例：{texts[0].splitlines()[0][:100]}")

    # ---- 已建好则直接复用，不重新编码 ----
    if not args.force and FAISS_INDEX.exists() and EMB_STATE.exists():
        st = json.loads(EMB_STATE.read_text(encoding="utf-8"))
        if st.get("rows_written") == total and st.get("total") == total:
            idx = faiss_read(FAISS_INDEX)
            if idx.ntotal == total:
                print(f"  ✓ 向量索引已存在（{idx.ntotal} 条 × {idx.d} 维），跳过编码")
                return {"skipped": True, "n": int(idx.ntotal), "dim": int(idx.d),
                        "elapsed": 0.0, "model": st.get("model", {})}

    t0 = time.monotonic()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    emb = Embedder(device=args.device, batch_size=args.batch_size,
                   max_seq_length=args.max_seq_length)
    info = emb.info()
    print(f"  设备 {info['device']} | torch {info['torch']} | "
          f"CUDA={info['cuda_available']} | dim={info['dim']} | "
          f"max_seq_length={info['max_seq_length']}")
    print(f"  batch_size ≤ {args.batch_size}（实际取 min(批内条数, 该值)，"
          f"且受 token 预算 {args.token_budget} 约束）| "
          f"padding=批内动态 | normalize_embeddings=True")
    if info["device"] == "cpu":
        import torch
        print(f"  CPU 线程 {torch.get_num_threads()} / 物理核 "
              f"{os.cpu_count()}")
    # GPU 状态要打清楚：cuda_available=False 有两种截然不同的原因 ——
    # （a）装了 CPU 版轮子（torch.version.cuda is None，有卡也用不上），
    # （b）轮子带 CUDA 但机器无卡。含糊其辞会让人以为「换台机器就能用 GPU」。
    import torch as _t
    if info["cuda_device"]:
        p = _t.cuda.get_device_properties(0)
        print(f"  GPU: {info['cuda_device']} | 显存 {p.total_memory / 2**30:.1f} GB"
              f" | 算力 {p.major}.{p.minor}")
    elif _t.version.cuda is None:
        print("  GPU: 不可用 —— torch 是纯 CPU 轮子（torch.version.cuda=None），"
              "机器有没有卡都用不上")
    else:
        print(f"  GPU: 无可用 CUDA 设备（torch 编译时 CUDA={_t.version.cuda}）")

    lens, order, batches, fp = plan_order(
        emb, texts, args.max_seq_length, args.batch_size, args.token_budget)
    padded = sum(int(lens[b].max()) * len(b) for b in batches)
    print(f"  token 总数 {int(lens.sum()):,}"
          f"（中位 {int(np.median(lens))} p95 {int(np.percentile(lens, 95))} "
          f"max {int(lens.max()):,}）")
    # 这里的 padded 容易被误读成「浪费掉的 token」，其实它是**真实计算量**：
    # 批内动态 padding 到该批最长条，所以 = 批内条数 × 批内最长。
    # 真正浪费的只有 padded - lens.sum()，下面直接把它打出来。
    print(f"  按长度排序分 {len(batches)} 批（批内最长 × 条数 ≤ "
          f"{args.token_budget} token）")
    print(f"  计算量 {padded:,} token = 文本 {int(lens.sum()):,} + "
          f"padding {int(padded - lens.sum()):,}"
          f"（浪费 {1 - lens.sum() / padded:.1%}；padding 策略为"
          f"**批内动态**，未传 padding=\"max_length\"、未传 max_length）")
    # 截断上限的**真实代价**必须打出来。cap 只影响长度 >cap 的块，本语料这些
    # 块几乎全是第八节的财务表 —— 也就是数值题要考的那些表。只打印
    # "max_seq_length=512" 会让人以为只是省算力，看不到丢的是正文。
    cap = args.max_seq_length
    if cap:
        full = emb.token_lengths(texts, max_seq_length=10 ** 9)   # 不截断
        clipped = int((full > cap).sum())
        lost = int(full.sum() - lens.sum())
        print(f"  max_seq_length={cap} 只作截断上限，不参与 padding")
        if clipped:
            kinds = df.loc[full > cap, "content_type"].value_counts().to_dict()
            secs = df.loc[full > cap, "section"].value_counts().head(3).to_dict()
            print(f"  ⚠ 截断代价：{clipped}/{total} 条（{clipped/total:.1%}）被切，"
                  f"丢掉 {lost:,} token（占全文 {lost/full.sum():.1%}）")
            print(f"    被切块类型 {kinds}；最集中的章节 {secs}")
            print(f"    —— 文本只保留到 {cap} token，较长表格的**后半截正文"
                  f"不进入向量**。向量检索对这些块的召回会变差。")
        else:
            print(f"    （本语料最长 {int(lens.max())}，无一条被截断，cap 不起作用）")
    else:
        print(f"  max_seq_length={info['max_seq_length']} 只作截断上限，不参与 padding；"
              f"本语料最长 {int(lens.max())}，无一条被截断")
    # 每批与累计的真实计算量，进度条与 ETA 都按 token 算（见循环里的注释）
    per_batch_tok = [int(lens[b].max()) * len(b) for b in batches]
    cum_tok = np.concatenate([[0], np.cumsum(per_batch_tok)]).astype(np.int64)

    if args.probe:
        k = min(args.probe_batches, len(batches))
        pick = np.unique(np.linspace(0, len(batches) - 1, k).astype(int))
        s = time.monotonic()
        p_tok = 0
        for j in pick:
            b = batches[j]
            emb.encode_documents([texts[i] for i in b], batch_size=len(b),
                                 show_progress=False)
            p_tok += int(lens[b].max()) * len(b)
        dt = time.monotonic() - s
        est = dt / p_tok * padded
        print(f"\n  [探针] 抽测 {len(pick)} 批 = padding 后 {p_tok:,} token，"
              f"耗时 {dt:.1f}s = {p_tok / dt:,.0f} tok/s")
        print(f"  [探针] 全量 {padded:,} token 预计 {est / 60:.1f} 分钟"
              f"（{est / 3600:.2f} 小时）")
        print(f"  [探针] 抽到的批：条数 "
              f"{[len(batches[j]) for j in pick]}，批内最长 token "
              f"{[int(lens[batches[j]].max()) for j in pick]}")
        return {"probe": True, "tok_per_sec": p_tok / dt, "est_sec": est}

    arr, bi = open_embeddings(total, emb.dim, args.force, fp,
                             cap=args.max_seq_length)
    rows_done = int(sum(len(b) for b in batches[:bi]))
    save_state(rows_done, emb.dim, total, info, batches_done=bi,
               n_batches=len(batches), order_fp=fp, real_tokens=int(lens.sum()),
               computed_tokens=padded,
               max_seq_length=int(emb.max_seq_length))

    # 本次真正编码了几批 / 花了多久，单独记。
    # 不能拿「总耗时 - 建 FAISS 耗时」当 encode_sec：断点续算时那个数
    # 只剩模型载入时间，会写成一个漂亮的假值（本项目就踩过：冒烟那一跑
    # 明明编码了 8 分钟，落盘的却是 11.8 秒）。
    bi_at_start = bi
    t_start = time.monotonic()
    t_mark = t_start
    while bi < len(batches):
        b = batches[bi]
        vecs = emb.encode_documents([texts[i] for i in b], batch_size=len(b),
                                    show_progress=False)
        arr[b] = vecs            # 按行号写回：编码次序 ≠ 行序
        arr.flush()
        bi += 1
        rows_done += len(b)
        save_state(rows_done, emb.dim, total, info, batches_done=bi,
                   n_batches=len(batches), order_fp=fp, real_tokens=int(lens.sum()),
                   computed_tokens=padded,
                   max_seq_length=int(emb.max_seq_length))
        t = time.monotonic()
        if t - t_mark >= 10 or bi == len(batches):
            el = t - t_start
            frac = bi / len(batches)
            # ETA 按 **token** 外推，不按批数：批是按长度降序切的，前几批的
            # token 数可达尾批的百倍，按批数线性外推会虚高约一倍（本项目实测：
            # 前 5 批时按批数显示「剩 711 分钟」，按 token 算真实剩余约 5.3 小时）。
            # 用实测速率外推还会偏保守 —— 剩余批更短、每 token 更快（自注意力
            # 随长度二次增长，长块段实测 79 tok/s，短块段回到 110）。
            tok_this_run = int(cum_tok[bi] - cum_tok[bi_at_start])
            tok_done = int(cum_tok[bi])
            rate = tok_this_run / max(el, 1e-9)          # token/秒
            left = (padded - tok_done) / rate if rate > 0 else 0.0
            print(f"  批 {bi:>4}/{len(batches)}  行 {rows_done:>5}/{total}"
                  f"  已用 {el / 60:>5.1f} 分钟  剩约 {left / 60:>5.1f} 分钟"
                  f" | {rate:>6.1f} tok/s  {rows_done / el:>5.2f} chunk/s"
                  f" | token {tok_done:,}/{padded:,}"
                  f"（{tok_done / padded * 100:>4.1f}%）", flush=True)
            t_mark = t

    enc_sec = time.monotonic() - t_start       # 只含本次的编码循环
    n_encoded = len(batches) - bi_at_start

    # ---- 建 FAISS ----
    tb = time.monotonic()
    vecs = np.asarray(arr, dtype=np.float32)
    index = faiss.IndexFlatIP(emb.dim)          # 向量已归一化 → 内积即余弦
    index.add(vecs)
    faiss_write(index, FAISS_INDEX)
    faiss_sec = time.monotonic() - tb

    write_mapping(FAISS_DIR, df)
    (FAISS_DIR / "build_info.json").write_text(json.dumps({
        **info,
        "n_vectors": total,
        "index_type": "IndexFlatIP",
        "metric": "cosine（向量已 L2 归一化）",
        "faiss_build_sec": round(faiss_sec, 1),
        "chunk_count": total,
        "token_total": int(lens.sum()),
        # 字段名要说清是什么：computed_tokens 是真实计算量（批内最长×条数），
        # real_tokens 是文本真实 token，二者之差才是 padding 浪费。
        # 旧名 "padding_tokens" 被误读成「浪费了 220 万 token」，已改。
        "computed_tokens": padded,
        "real_tokens": int(lens.sum()),
        "padding_waste": int(padded - lens.sum()),
        "padding_waste_ratio": round(1 - lens.sum() / padded, 4),
        "padding_strategy": "批内动态（sentence-transformers 默认；未传 max_length）",
        "encoder_order": "按 token 长度降序分桶（编码次序 ≠ 行序，按行号写回）",
        "text": ("元数据表头 + 正文" if getattr(args, "header", False) else "正文"),
        "header_cols": list(HEADER_COLS) if getattr(args, "header", False) else [],
        "encode_sec": round(enc_sec, 1),
        "encoded_batches_this_run": n_encoded,
        "resumed_from_batch": bi_at_start,
        "wall_sec": round(time.monotonic() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"skipped": False, "n": total, "dim": emb.dim,
            "elapsed": time.monotonic() - t0, "model": info,
            "tokens": int(lens.sum()), "computed_tokens": padded,
            "efficiency": lens.sum() / padded}


# ============================================================
# 二、BM25
# ============================================================
def tokenize(text: str) -> list[str]:
    import jieba

    cleaned = PIPE_RE.sub(" ", text).replace("|", " ").replace("**", " ")
    return [t for t in (w.strip().lower() for w in jieba.lcut(cleaned)) if t]


def build_bm25(df: pd.DataFrame, args) -> dict:
    from rank_bm25 import BM25Okapi

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    total = len(df)

    if not args.force and BM25_PKL.exists() and BM25_TOKENS.exists():
        try:
            bm = pickle.loads(BM25_PKL.read_bytes())
            if len(bm.doc_freqs) == total:
                print(f"  ✓ BM25 索引已存在（{len(bm.doc_freqs)} 篇），跳过")
                return {"skipped": True, "n": total, "elapsed": 0.0,
                        "tokens": int(sum(len(d) for d in bm.doc_freqs))}
        except Exception as e:  # noqa: BLE001
            print(f"  ! 已有 BM25 读取失败（{type(e).__name__}: {e}），重建")

    print(f"  jieba 分词中（{total} 篇）…")
    tk0 = time.monotonic()
    corpus = [tokenize(c) for c in doc_texts(df, args)]
    tk_sec = time.monotonic() - tk0
    n_tok = sum(len(t) for t in corpus)
    n_empty = sum(1 for t in corpus if not t)
    assert n_empty == 0, f"{n_empty} 篇分词后为空"
    print(f"  分词完成 {tk_sec:.1f}s，共 {n_tok:,} 个 token，"
          f"平均 {n_tok / total:.0f}/篇")

    bm25 = BM25Okapi(corpus)
    BM25_PKL.write_bytes(pickle.dumps(bm25, protocol=4))
    BM25_TOKENS.write_bytes(pickle.dumps(corpus, protocol=4))

    write_mapping(BM25_DIR, df)
    (BM25_DIR / "build_info.json").write_text(json.dumps({
        "tokenizer": "jieba.lcut + 小写 + 抹掉 Markdown 管道",
        "text": ("元数据表头 + 正文" if getattr(args, "header", False) else "正文"),
        "header_cols": list(HEADER_COLS) if getattr(args, "header", False) else [],
        "bm25": "rank_bm25.BM25Okapi（k1=1.5, b=0.75 为库默认）",
        "n_docs": total,
        "n_tokens": n_tok,
        "avg_tokens_per_doc": round(n_tok / total, 1),
        "tokenize_sec": round(tk_sec, 1),
        "chunk_count": total,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"skipped": False, "n": total, "elapsed": time.monotonic() - t0,
            "tokens": n_tok, "tokenize_sec": tk_sec}


# ============================================================
def write_mapping(out_dir: Path, df: pd.DataFrame) -> None:
    """保存 chunk ↔ 索引位置 的映射。两份都在，索引目录自包含。"""
    chunk_ids = df["chunk_id"].tolist()
    (out_dir / "chunk_ids.json").write_text(
        json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
    slim = df[["chunk_id", "stock_code", "company", "section", "subsection",
               "page_start", "page_end", "content_type", "char_count"]].copy()
    slim.insert(0, "row", range(len(slim)))
    slim.to_parquet(out_dir / "chunk_map.parquet", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["faiss", "bm25"], default=None)
    ap.add_argument("--device", default=config.embedding_device or None,
                    help="默认自动检测：cuda→mps→cpu")
    ap.add_argument("--batch-size", type=int, default=config.embedding_batch_size,
                    help="每批最多几条（批内按长度排序，padding 浪费很小）")
    ap.add_argument("--token-budget", type=int, default=config.embedding_token_budget,
                    help="每批的 token 上限（批内最长 × 条数），防超长表格块 OOM")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU 线程数，默认用 torch 自己的默认值")
    ap.add_argument("--max-seq-length", type=int,
                    default=config.embedding_max_seq_length or None,
                    help="截断上限（0/空=不截断）。注意这不是加速开关：调小会按 token 丢弃正文，" "被切的主要是第八节财务表")
    ap.add_argument("--probe", action="store_true", help="只测速，不建索引")
    ap.add_argument("--probe-batches", type=int, default=12,
                    help="探针抽测多少批（沿长度谱均匀取）")
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 条，用于冒烟测试（产物不完整，下次跑会重建）")
    ap.add_argument("--force", action="store_true", help="丢弃已有产物重算")
    ap.add_argument("--header", action="store_true",
                    help="索引文本改为「元数据表头 + 正文」（表头见 make_headers），"
                         "产物写到 index/header/ 下 —— **不覆盖**现有索引")
    args = ap.parse_args()

    # 线程数要在 import torch 之前设进环境变量才生效（torch 在 Embedder 里才导入）
    if args.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_NUM_THREADS"):
            os.environ[k] = str(args.threads)

    if args.limit and args.only != "faiss":
        ap.error("--limit 只能配 --only faiss：它会以少量文档重建 BM25 索引，"
                 "把已有的全量 BM25 覆盖掉")

    # ---- 索引目录解析：--header 与 --limit 各走独立子目录 ----
    # 为什么 --header 必须写另一个目录（而不是就地重建）：
    # 加了表头 → 文本变 → token 长度变 → 批计划指纹 order_fp 变 →
    # `open_embeddings()` 判定「已有向量与本次不符」，**unlink 掉 embeddings.npy
    # 从头再算**。而那一份是 3.4 小时算出来的、且正是当前线上索引。分开写，
    # 新旧两套才能并存，也才能在同一套 23 题上做 before/after。
    global FAISS_DIR, BM25_DIR, EMB_NPY, EMB_STATE, FAISS_INDEX, BM25_PKL, BM25_TOKENS
    _root = config.index_dir
    if args.limit:
        # 冒烟测试的产物写到独立目录，绝不碰真索引
        _root = _root / "_smoke"
    if args.header:
        _root = _root / "header"
    FAISS_DIR = _root / "faiss"
    BM25_DIR = _root / "bm25"
    EMB_NPY = FAISS_DIR / "embeddings.npy"
    EMB_STATE = FAISS_DIR / "embeddings.state.json"
    FAISS_INDEX = FAISS_DIR / "faiss.index"
    BM25_PKL = BM25_DIR / "bm25.pkl"
    BM25_TOKENS = BM25_DIR / "corpus_tokens.pkl"
    if args.header:
        print(f"! --header：索引文本 = 元数据表头 + 正文，"
              f"产物写到 {FAISS_DIR.parent}（现有 {config.index_dir / 'faiss'} 不受影响）")

    config.index_dir.mkdir(parents=True, exist_ok=True)
    df = load_chunks()
    if args.limit:
        # 按长度分层抽样：冒烟测试要覆盖到 8913 token 的长表格块那一端
        s = df.sort_values("char_count", kind="stable").index.to_numpy()
        pick = s[np.linspace(0, len(s) - 1, args.limit).astype(int)]
        df = df.loc[np.sort(pick)].reset_index(drop=True)
        print(f"! --limit {args.limit}：只处理前 {len(df)} 条，"
              f"产物写到 {FAISS_DIR}（冒烟测试用）")
    print(f"chunk 总数 {len(df)}，来自 {META}")
    print(f"  表格 {(df['content_type'] == 'table').sum()} / "
          f"正文 {(df['content_type'] == 'text').sum()}")
    print(f"  字符总数 {df['char_count'].sum():,}\n")

    result: dict = {}
    t_all = time.monotonic()
    try:
        if args.only in (None, "faiss"):
            print("[1/2] 向量索引 (Qwen3-Embedding-0.6B → FAISS)")
            result["faiss"] = build_faiss(df, args)
            print()
        if args.only in (None, "bm25"):
            print("[2/2] BM25 索引 (jieba + rank_bm25)")
            result["bm25"] = build_bm25(df, args)
            print()
    except KeyboardInterrupt:
        print("\n中断。向量进度已存盘，重跑本脚本会从断点续算。")
        return 130

    if args.probe:
        return 0

    total_sec = time.monotonic() - t_all

    # ---------------- 汇总 ----------------
    f_size = FAISS_INDEX.stat().st_size if FAISS_INDEX.exists() else 0
    emb_size = EMB_NPY.stat().st_size if EMB_NPY.exists() else 0
    b_size = BM25_PKL.stat().st_size if BM25_PKL.exists() else 0
    tok_size = BM25_TOKENS.stat().st_size if BM25_TOKENS.exists() else 0
    f = result.get("faiss", {})
    b = result.get("bm25", {})

    print("=" * 66)
    print("  建索引完成")
    print("=" * 66)
    print(f"  chunk 数量        : {len(df)}")
    print(f"  embedding 数量    : {f.get('n', 0)}")
    print(f"  embedding 维度    : {f.get('dim', 0)}")
    if f.get("tokens"):
        print(f"  编码 token 总数   : {f['tokens']:,}"
              f"（批内补齐后计算量 {f['computed_tokens']:,}，"
              f"padding 浪费 {1 - f['efficiency']:.1%}）")
    print(f"  设备              : {f.get('model', {}).get('device', '-')}"
          f"（CUDA={f.get('model', {}).get('cuda_available')}）")
    print(f"  FAISS 大小        : {human(f_size)}  "
          f"（{FAISS_DIR / 'faiss.index'}）")
    print(f"    向量 npy        : {human(emb_size)}  (embeddings.npy)")
    print(f"    IndexFlatIP     : {f.get('n', 0)} × {f.get('dim', 0)} float32，精确检索")
    print(f"  BM25 大小         : {human(b_size + tok_size)}  "
          f"（bm25.pkl {human(b_size)} + corpus_tokens.pkl {human(tok_size)}）")
    print(f"    BM25 token 总数 : {b.get('tokens', 0):,}")
    print(f"  建索引耗时        : {total_sec:.1f} 秒"
          f"（向量 {f.get('elapsed', 0):.1f}s / BM25 {b.get('elapsed', 0):.1f}s）")
    if f.get("skipped") or b.get("skipped"):
        print(f"  注：{'向量' if f.get('skipped') else ''}"
              f"{'、' if f.get('skipped') and b.get('skipped') else ''}"
              f"{'BM25' if b.get('skipped') else ''}复用了已有产物，未重算")
    print("=" * 66)
    print(f"\n  映射文件：{FAISS_DIR / 'chunk_ids.json'}"
          f" 与 {BM25_DIR / 'chunk_ids.json'}（行号 → chunk_id）")
    print(f"           {FAISS_DIR / 'chunk_map.parquet'}（含公司/章节/页码，供过滤）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
