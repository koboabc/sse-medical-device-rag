"""Qwen3-Embedding-0.6B 封装。

要点：
  * 设备自动检测：CUDA → MPS → CPU，有 GPU 就用 GPU
  * 文档与查询走**非对称**编码路径
      - 文档：不加任何指令（模型的 prompts.document 为空串）
      - 查询：prompt_name="query"（模型的 prompts.query 是检索指令）
    这是 Qwen3-Embedding 的官方用法；对文档也套指令会拉低召回。
  * normalize_embeddings=True，向量已 L2 归一化，FAISS 用内积即等价余弦
  * 只用本地模型目录，不碰任何在线 Embedding API

用法：
    from src.indexing.embedder import Embedder
    emb = Embedder()
    V = emb.encode_documents(texts)     # (n, 1024) float32
    q = emb.encode_queries(["联影医疗营收多少"])   # (1, 1024) float32
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config  # noqa: E402

MODEL_DIR = config.embedding_model_dir
QUERY_PROMPT = "query"


def pick_device() -> str:
    """自动检测：CUDA → MPS → CPU。"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_dir: Path | str = MODEL_DIR,
                 device: str | None = None,
                 batch_size: int = 32,
                 max_seq_length: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在：{self.model_dir}")

        self.device = device or pick_device()
        self.batch_size = batch_size

        # CPU 上用 float32：配置里是 bfloat16，CPU 上 bf16 反而更慢
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = SentenceTransformer(
            str(self.model_dir), device=self.device,
            model_kwargs={"torch_dtype": dtype},
        )
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.max_seq_length = self.model.max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()

        # 记录模型自带的 prompt 配置，便于把"查询用了指令"这件事留痕
        self.prompts = dict(getattr(self.model, "prompts", {}) or {})

    # ---------------- 文档 ----------------
    def encode_documents(self, texts: list[str], batch_size: int | None = None,
                         show_progress: bool = True) -> np.ndarray:
        """文档编码：不加指令。"""
        return self._encode(texts, prompt_name=None,
                            batch_size=batch_size, show_progress=show_progress)

    # ---------------- 查询 ----------------
    def encode_queries(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """查询编码：必须带 prompt_name="query"。"""
        if QUERY_PROMPT not in self.prompts:
            raise RuntimeError(
                f"模型没有名为 {QUERY_PROMPT!r} 的 prompt，实际有 {list(self.prompts)}；"
                f"查询侧不能裸编码，否则与文档侧不在同一指令空间"
            )
        return self._encode(texts, prompt_name=QUERY_PROMPT,
                            batch_size=batch_size, show_progress=False)

    # ---------------- 分桶 ----------------
    def token_lengths(self, texts: list[str],
                      max_seq_length: int | None = None) -> np.ndarray:
        """逐条 token 数（截断后）。用于按长度分桶批处理，压掉 padding 浪费。"""
        tok = self.model.tokenizer
        cap = max_seq_length or self.max_seq_length
        return np.array(
            [len(tok(t, truncation=True, max_length=cap)["input_ids"]) for t in texts],
            dtype=np.int64)

    # ---------------- 内部 ----------------
    def _encode(self, texts: list[str], prompt_name: str | None,
                batch_size: int | None, show_progress: bool) -> np.ndarray:
        kw = {}
        if prompt_name:
            kw["prompt_name"] = prompt_name
        # 注意：这里**故意不传** padding / max_length。
        # sentence-transformers 的默认是 padding=True —— 即**批内动态 padding**
        # 到该批最长条，而不是 padding 到模型的 max_seq_length。
        # 传 padding="max_length" 才是灾难：每批都会补到 32768，算力浪费几十倍。
        # 本项目实测（4657 条）动态 padding 的浪费只有 1.6%。
        v = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,      # 归一化，FAISS 内积 == 余弦
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            **kw,
        )
        return np.ascontiguousarray(v, dtype=np.float32)

    def info(self) -> dict:
        import torch

        return {
            "model_dir": str(self.model_dir),
            "device": self.device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            "dim": int(self.dim),
            "max_seq_length": int(self.max_seq_length),
            "batch_size": int(self.batch_size),
            "normalize_embeddings": True,
            "query_prompt_name": QUERY_PROMPT,
            "query_prompt": self.prompts.get(QUERY_PROMPT, ""),
            "document_prompt": self.prompts.get("document", ""),
        }


if __name__ == "__main__":
    import json

    e = Embedder()
    print(json.dumps(e.info(), ensure_ascii=False, indent=2))
