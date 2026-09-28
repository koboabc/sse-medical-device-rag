"""检索侧的唯一数据入口：元数据 / BM25 / 向量索引 的装载与降级。

为什么要有这一层：
  * **行号是全局契约**。`build_index.load_chunks()` 把 metadata.parquet 按
    chunk_id 排序后，那个行序 == `chunk_map.parquet` 的 row == FAISS 的 I
    == BM25 argsort 的下标。而 metadata.parquet 的**物理行序不是**这个行序
    （物理首行是 605369-0001，排序后首行是 600055-0001）。任何一处忘记
    走 load_chunks()，检索结果就会整体错位且**不会报错**。所以这里装载时
    就断言 chunk_map 与 load_chunks 完全同序，把错位变成启动时的硬失败。
  * **向量索引可能不存在**。索引构建以小时计，且中途可中断；BM25 早就建好了。
    所以 FAISS 走惰性装载 + 缺失即降级，只记 warning，绝不抛异常 —— 否则
    「只有 BM25」时连召回评测都跑不了。
  * **Embedder 很贵**（载入 0.6B 要 2.6 GB 内存 + 约 30 秒）。只在真正走向量
    路时才构造，且全进程单例。

用法：
    from src.retrieval.store import store

    st = store()                      # 惰性：不碰 FAISS、不载模型
    hits = st.search_bm25("联影医疗 营业收入", k=20)
    st.has_vector                    # 向量索引是否可用
    rec = st.record(hits[0].row)     # 行号 → 完整记录（含正文）

命令行自检：
    python src/retrieval/store.py
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402
from src.indexing.build_index import load_chunks, tokenize  # noqa: E402

# 默认索引目录（= 正文索引）。**其余变体走 `Store(index_dir=...)`**：
#   index/           正文索引（建索引未加 --header）
#   index/header/    「元数据表头 + 正文」索引（build_index.py --header）
# 两套的 chunk_id 行序完全相同（都来自同一个 load_chunks()），所以同一套评测题
# 可以直接对比，只是召回所依据的向量/词频不同。
#
# 这里保留模块级常量是为了向后兼容（scripts/verify_retrieval.py 直接 import 它们）；
# Store 实例自己持有 faiss_dir/bm25_dir，不再读这两个全局。
BM25_DIR = config.index_dir / "bm25"
FAISS_DIR = config.index_dir / "faiss"


def index_variant(index_dir: str | Path | None = None) -> str:
    """索引变体的中文名：**按目录实际名字判断**，不看它是不是配置默认值。

    早期版本用「与 `config.index_dir` 相等 ⇒ 正文」来推。那在默认布局下是对的，
    但只要把 `INDEX_DIR` 指到 `index/header`（`.env` 里一行的事），表头索引就会
    被显示成「正文」—— 页面和报告里几个数字放一起就分不清谁是谁，正好犯
    「结果自带口径」这条规矩。改成看目录名，配成什么就报什么。
    """
    d = Path(index_dir) if index_dir else config.index_dir
    return "表头增强" if d.name == "header" else "正文"
# 冒烟产物在 index/_smoke/faiss/，只覆盖 48 条分层抽样，行序与全量不一致，
# 绝不能当生产索引用（能用 FAISS_DIR 的检查兜住这点：两者不是同一目录）
SMOKE_DIR = config.index_dir / "_smoke" / "faiss"


@lru_cache(maxsize=2)
def _shared_embedder(device: str | None, max_seq_length: int | None):
    """Embedder 进程级单例。载入 0.6B 要 2.6 GB 内存，不能被索引变体各持一份。

    max_seq_length 必须与建索引时**一致**（config.embedding_max_seq_length）。
    查询很短、截断本来就不生效，所以不传也不会立刻出错 —— 但那就留下了
    一个陷阱：任何经这里重编码文档的路径（增量更新、抽样核对）都会算出
    与索引不同口径的向量，且是静默的。口径只该有一个来源：config。
    """
    from src.indexing.embedder import Embedder

    return Embedder(device=device, max_seq_length=max_seq_length)


@dataclass
class Hit:
    """一条召回结果。row 是全局行号，可反查任何一路的产物。"""

    row: int
    chunk_id: str
    score: float
    source: str                 # "bm25" | "vector"
    rank: int                   # 该路内的排名，从 1 开始
    scores: dict[str, float] = field(default_factory=dict)   # 融合后带上各路原始分
    dup_chunk_ids: list[str] = field(default_factory=list)   # 内容重复被合并掉的块

    def __repr__(self) -> str:  # 调试时一行能看完
        return (f"<Hit {self.chunk_id} {self.source} rank={self.rank} "
                f"score={self.score:.4f}>")


class Store:
    """元数据 + 两路索引的持有者。进程内应只建一个（见模块末尾的 store()）。"""

    def __init__(self, index_dir: str | Path | None = None,
                 load_vector: bool = False) -> None:
        # 索引变体：默认目录放正文索引，index/header 放「表头+正文」索引。
        # 行序契约两套一致（同一个 load_chunks()），所以可以自由切换做 A/B。
        self.index_dir: Path = Path(index_dir) if index_dir else config.index_dir
        self.faiss_dir: Path = self.index_dir / "faiss"
        self.bm25_dir: Path = self.index_dir / "bm25"

        self.df: pd.DataFrame = load_chunks()
        self.n: int = len(self.df)
        self.warnings: list[str] = []

        self._load_bm25()
        self._check_alignment()

        self._index = None            # FAISS 索引对象
        self._vec_ids: list[str] = []
        self._embedder = None         # 惰性：不走向量路就永不构造
        self.has_vector: bool = False

        if load_vector:
            self.ensure_vector()

    # ---------------- 装载 ----------------
    def _load_bm25(self) -> None:
        bm_p, tk_p = self.bm25_dir / "bm25.pkl", self.bm25_dir / "corpus_tokens.pkl"
        for p in (bm_p, tk_p, self.bm25_dir / "chunk_ids.json"):
            if not p.exists():
                raise FileNotFoundError(
                    f"缺少 {p}，请先跑 python src/indexing/build_index.py --only bm25"
                    + ("--header" if self.index_dir.name == "header" else ""))
        # rank_bm25 的对象用 pickle 存（协议 4），读法与写入侧一致
        self.bm25 = pickle.loads(bm_p.read_bytes())
        self.corpus_tokens = pickle.loads(tk_p.read_bytes())
        self.bm25_ids: list[str] = json.loads(
            (self.bm25_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        if len(self.bm25.doc_freqs) != self.n:
            raise ValueError(
                f"BM25 文档数 {len(self.bm25.doc_freqs)} != metadata 行数 {self.n}，"
                f"索引与数据不同步，请重建")

    def _check_alignment(self) -> None:
        """行序契约的硬校验：索引行号 ↔ chunk_id ↔ metadata 行。

        `verify_index.py` 里也断言了同一件事，但那是**离线校验脚本**；
        这里是**运行时**入口，每次检索都受它保护。错位是静默错误，
        必须在装载阶段就炸掉。
        """
        cmap_p = self.bm25_dir / "chunk_map.parquet"
        if not cmap_p.exists():
            raise FileNotFoundError(f"缺少 {cmap_p}，请先重建 BM25 索引")
        cmap = pd.read_parquet(cmap_p)
        got = cmap["chunk_id"].tolist()
        want = self.df["chunk_id"].tolist()
        if len(cmap) != self.n or got != want:
            raise ValueError(
                "chunk_map.parquet 与 load_chunks() 行序不一致 —— 检索结果会整体错位。"
                f"（chunk_map {len(cmap)} 行 / metadata {self.n} 行；"
                f"首个 chunk_id {got[0] if got else None} vs {want[0] if want else None}）"
                "请重建索引：python src/indexing/build_index.py")
        if self.bm25_ids != want:
            raise ValueError("chunk_ids.json 与 load_chunks() 行序不一致，请重建索引")

    def ensure_vector(self) -> bool:
        """按需装载 FAISS。返回是否可用；不可用只记 warning，不抛。"""
        if self.has_vector:
            return True
        idx_p = self.faiss_dir / "faiss.index"
        ids_p = self.faiss_dir / "chunk_ids.json"
        if not idx_p.exists() or not ids_p.exists():
            self.warnings.append(
                f"向量索引不存在（{idx_p}），本次只走 BM25。"
                f"建索引：python src/indexing/build_index.py --only faiss"
                + (" --header" if self.index_dir.name == "header" else ""))
            return False

        from src.indexing.build_index import faiss_read

        idx = faiss_read(idx_p)      # 不是 faiss.read_index：中文路径它打不开
        ids = json.loads(ids_p.read_text(encoding="utf-8"))
        if idx.ntotal != self.n or ids != self.df["chunk_id"].tolist():
            self.warnings.append(
                f"向量索引与 metadata 不一致（ntotal={idx.ntotal} / "
                f"chunk_ids={len(ids)} / metadata={self.n}），已忽略向量路")
            return False
        self._index, self._vec_ids, self.has_vector = idx, ids, True
        return True

    @property
    def embedder(self):
        """Embedder 单例。载入 0.6B 要 2.6 GB 内存，只在真正编码时才构造。"""
        if self._embedder is None:
            self._embedder = _shared_embedder(config.embedding_device or None,
                                              config.embedding_max_seq_length or None)
        return self._embedder

    # ---------------- 召回 ----------------
    def allow_mask(self, stock_codes: list[str] | None) -> np.ndarray | None:
        """把「只允许这些公司」变成一个布尔掩码。

        硬过滤必须在**打分前**生效：若先召回 20 条再筛掉不相关的公司，
        目标公司的块可能压根没进前 20（尤其是它只占 4657 条里的 1/17）。
        """
        if not stock_codes:
            return None
        want = {str(c).strip() for c in stock_codes}
        mask = self.df["stock_code"].isin(want).to_numpy()
        if not mask.any():
            raise ValueError(f"指定的公司 {sorted(want)} 不在语料里，"
                             f"可选：{sorted(self.df['stock_code'].unique())}")
        return mask

    def search_bm25(self, query: str, k: int = 20,
                    allow: np.ndarray | None = None) -> list[Hit]:
        """BM25 召回。查询侧必须复用 build_index.tokenize（同款 jieba 规则）。"""
        q = tokenize(query)
        if not q:
            return []
        scores = np.asarray(self.bm25.get_scores(q), dtype=np.float64)
        if scores.size != self.n:
            raise ValueError(f"BM25 打分长度 {scores.size} != {self.n}")
        if allow is not None:
            scores = np.where(allow, scores, -np.inf)
        # rank_bm25 对无重叠词给出 0 分；取 top-k 前先剔掉 0，避免
        # 查询词全不在词表时凑出一堆无意义的「第 1 名」。
        # `> 0` 同时挡掉 allow 掩码留下的 -inf。
        order = np.argsort(-scores)[:max(k, 1)]
        hits = [Hit(int(i), self.bm25_ids[int(i)], float(scores[i]), "bm25", r + 1)
                for r, i in enumerate(order) if scores[i] > 0]
        return hits

    def search_vector(self, query: str, k: int = 20,
                      allow: np.ndarray | None = None) -> list[Hit]:
        """向量召回。查询侧走 prompt_name="query"（非对称编码，见 embedder）。

        FAISS 的 search 不能带掩码，所以限定公司时要多搜一些再筛
        （k*20 上限，够覆盖 1/17 的语料占比；FlatIP 是精确检索，多搜只是多算点内积）。
        """
        if not self.ensure_vector():
            return []
        # (1, dim) 已 L2 归一化。ascontiguousarray 是防御性的：faiss 的 python
        # 包装直接拿底层指针，非连续数组会读出乱数（不报错）。
        qv = np.ascontiguousarray(self.embedder.encode_queries([query]),
                                  dtype=np.float32)
        want = max(k, 1) * (20 if allow is not None else 1)
        d, i = self._index.search(qv, min(int(want), self.n))
        hits: list[Hit] = []
        for s, row in zip(d[0], i[0]):
            if row < 0:
                continue
            if allow is not None and not allow[int(row)]:
                continue
            hits.append(Hit(int(row), self._vec_ids[int(row)], float(s),
                            "vector", len(hits) + 1))
            if len(hits) >= max(k, 1):
                break
        return hits

    # ---------------- 反查 ----------------
    def record(self, row: int) -> dict:
        """行号 → 完整记录（含正文）。这是拿到 content 的唯一正确途径。"""
        r = self.df.iloc[int(row)]
        return {
            "row": int(row),
            "chunk_id": r["chunk_id"],
            "stock_code": r["stock_code"],
            "company": r["company"],
            "report": r["report"],          # 报告名（如「2026年半年度报告」）
            "section": r["section"],
            "subsection": r["subsection"],
            "page_start": int(r["page_start"]),
            "page_end": int(r["page_end"]),
            "content_type": r["content_type"],
            "char_count": int(r["char_count"]),
            "content": r["content"],
        }

    def describe(self) -> str:
        """一行状态摘要，给 CLI 与页面状态栏用。

        「未装载」和「未建」必须分开说：FAISS 是惰性的，刚启动时 has_vector
        永远是 False。把这两种情况混成一句「无向量」，会让人以为索引没建好
        —— 本项目就这么被误导过一次，查了半天才发现索引是好的、只是没去载。
        """
        if self.has_vector:
            v = f"{self.n} 条向量（已装载）"
        elif (self.faiss_dir / "faiss.index").exists():
            v = "向量索引已建（未装载，首次走向量路时载入 0.6B 模型，约 30 秒）"
        else:
            v = "无向量索引（未建，将只走 BM25）"
        variant = index_variant(self.index_dir)
        return (f"chunks {self.n} | BM25 {len(self.bm25.doc_freqs)} 篇 | {v} | "
                f"索引变体 {variant}（{self.index_dir}）")


@lru_cache(maxsize=4)
def store(index_dir: str | None = None) -> Store:
    """进程内单例（按索引变体各一个）。惰性：不主动装载 FAISS、不载入模型。

    index_dir=None → 正文索引（config.index_dir）；"index/header" → 表头索引。
    lru_cache 的键含 index_dir，所以两套索引可以同时驻留在一个进程里做对比
    —— 但**注意向量路**：两套索引的向量空间相同（同一个 embedding 模型），
    只是文档侧文本不同，所以共用同一个 Embedder 是安全的。
    """
    return Store(index_dir=index_dir, load_vector=False)


def main() -> int:
    st = Store(load_vector=True)
    print(st.describe())
    for w in st.warnings:
        print(f"  ! {w}")
    for q in ("联影医疗 营业收入", "心脉医疗 研发投入"):
        print(f"\nBM25 「{q}」top3：")
        for h in st.search_bm25(q, k=3):
            r = st.record(h.row)
            print(f"  {h.score:8.3f}  {r['chunk_id']}  {r['company'][:12]} | "
                  f"{r['section']} | p{r['page_start']} | {r['content_type']} | "
                  f"{r['char_count']}字")
    if st.has_vector:
        q = "联影医疗的营业收入是多少"
        print(f"\n向量 「{q}」top3：")
        for h in st.search_vector(q, k=3):
            r = st.record(h.row)
            print(f"  {h.score:8.4f}  {r['chunk_id']}  {r['section']} | "
                  f"p{r['page_start']} | {r['content_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
