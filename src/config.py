"""集中配置：从 .env 读取，带默认值。

用法：
    from src.config import config
    print(config.embedding_model, config.top_k_final)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录 = 本文件的上上级
ROOT = Path(__file__).resolve().parents[1]

# 先加载 .env（不存在也不报错，用默认值）
load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None or v.strip() == "" else v.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).lower() in {"1", "true", "yes", "on"}


def _path(key: str, default: str) -> Path:
    """相对路径一律相对项目根目录解析。"""
    p = Path(_env(key, default))
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass(frozen=True)
class Config:
    # ---- LLM ----
    llm_provider: str = _env("LLM_PROVIDER", "deepseek")
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_model: str = _env("LLM_MODEL", "deepseek-chat")
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 2048)
    llm_timeout: int = _env_int("LLM_TIMEOUT", 60)
    # 单条证据进 prompt 的正文上限。表格块最长 13758 字，6 条全量会让 prompt
    # 被一张表挤满。截断按行做且**保留表头**（见 generator.clip_for_prompt），
    # 并在 prompt 里标注已截断 —— 不让模型以为看到的就是完整表格。
    llm_evidence_max_chars: int = _env_int("LLM_EVIDENCE_MAX_CHARS", 1800)

    # ---- Embedding ----
    # 本地模型目录，不走在线 API。Qwen3-Embedding-0.6B 原生 1024 维。
    embedding_model: str = _env("EMBEDDING_MODEL", "models/Qwen3-Embedding-0.6B")
    # 空串 = 自动检测（cuda → mps → cpu），见 embedder.pick_device()
    embedding_device: str = _env("EMBEDDING_DEVICE", "")
    # 实测（cap=1024，L≈300-500 的同质批，CPU 12 线程，单次测量）：
    # bs=4 → 165.0 tok/s，bs=8 → 159.1，bs=16 → 128.6，bs=32 → 110.9。
    # **越小越快**，且 bf16 比 fp32 慢 4.6 倍（Alder Lake 无 AVX512-BF16），
    # 所以 CPU 上固定 fp32 + 小批。这就是历史"110 tok/s"的来源（那是 bs=32）。
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 4)
    embedding_dim: int = _env_int("EMBEDDING_DIM", 1024)
    # 截断上限。**这不是加速开关**：0.6B 的原生窗口是 32768，但本语料全长只用
    # 2,174,989 token（中位 381，最长 8913），从未接近过 32K。调小它等于按 token
    # 丢弃正文：cap=512 切掉 1,631 块 / 516,142 token（24%），其中**散文 744/1268
    # = 58.7%**（第三节管理层讨论 219 块，"变动原因"就在那儿）—— 只省 0.5 小时。
    # 定 1024：只切最长的 286 个表格（8.6% token），散文一条不切。
    # 改这个值会让批计划指纹变化 → 已算好的向量作废重算（必须作废：新旧向量
    # 来自不同输入口径，混在一个索引里是错的）。
    embedding_max_seq_length: int = _env_int("EMBEDDING_MAX_SEQ_LENGTH", 1024)
    # 每批的 token 上限（批内最长条 × 批内条数）。CPU 上 padding 是实打实的
    # 算力，所以编码按 token 长度排序分桶，批内长度接近；这个上限是防超长
    # 表格块（最长 8913 token）把批撑爆内存的闸门。
    embedding_token_budget: int = _env_int("EMBEDDING_TOKEN_BUDGET", 24576)

    @property
    def embedding_model_dir(self) -> Path:
        """模型目录：相对路径按项目根解析。"""
        p = Path(self.embedding_model)
        return p if p.is_absolute() else (ROOT / p).resolve()

    # ---- 检索 ----
    chunk_size: int = _env_int("CHUNK_SIZE", 512)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 64)
    # 每路召回的深度与最终证据数，由需求定为 20 / 20 / 6。
    #
    # ⚠ 实测记要（2026-09-27，全库 4657，问题「联影医疗 2026 年上半年营业收入是多少」）：
    # **深度不是这个问题的症结，调深救不回来**。答案在 `688271-0009`（第二节 > 六、
    # 公司主要会计数据和财务指标，营收 7,051,579,098.09）与 `688271-0010`，而这两块
    # 的 BM25 全库排名是 **#795 / #593**、向量全库排名是 **#541 / #596** —— 深度取
    # 20、50、100 都进不了池子，软加权更是一条也不生效（加权只能在池内重排）。
    # 曾经的「第八节排名第 24、取 50 就够」是**误记**：第 24 名是 688271-0063（第三节
    # 正文），联影第八节最好的块是 688271-0201（三、公司基本情况，#18），也不是报表本体。
    #
    # 真正的成因有两条，都已实测：
    #   1. 那张会计数据表里**没有「联影」二字**，而第三节/第四节的散文里反复写全称
    #      「上海联影医疗科技股份有限公司」——「联影」「医疗」在 17 份报告里都很稀有
    #      （高 IDF），于是 BM25 第 1 名（688271-0135，第四节 ESG 正文）**一次「营业收入」
    #      都没出现**，纯靠实体词刷分压过内容词。
    #   2. 向量侧同理：17 家公司的「主要会计数据」表正文几乎同构、都不含公司名，
    #      模型无从分辨哪张是联影的。
    #
    # 两条已验证的修法（**都已实现**，见 README 与 evaluation/results/）：
    #   A. 把 (公司/代码/报告/章节/小节) 拼到正文前重算（≈4 h，写到 index/header/，
    #      不动现有索引）：对抗集实测 gold `688271-0009` 余弦 0.3433 → **0.6659**
    #      （越过现第 6 名 0.6605），别家公司的同名表只涨 0.06~0.11 —— 公司名在真做
    #      跨公司消歧。BM25 单路实测 hit@10 0.150→0.300（救回 3 题、弄坏 0 题），
    #      旗舰题 gold 从全库 #795 升到 #132（仍进不了 Top-20，改善≠解决）。
    #   B. 条件筛选题改走结构化表组（`structured.py`，免重算、默认开）：
    #      hit@10 0.000→**1.000**，证据覆盖 17/17 家公司，其余 15 题一条未变。
    # TOP_K_* 保留 20 是需求所定 —— 深度不是这两类问题的症结，调大救不回来。
    top_k_vector: int = _env_int("TOP_K_VECTOR", 20)
    top_k_bm25: int = _env_int("TOP_K_BM25", 20)
    top_k_final: int = _env_int("TOP_K_FINAL", 6)
    rerank_enabled: bool = _env_bool("RERANK_ENABLED", False)

    # ---- 跨公司条件筛选增强：结构化表组路由 ----
    # ⚠ 定位：这是**一个题型**的增强，不是通用召回能力。报告里的 17/17 是
    # 「表组覆盖了几家公司的表」（覆盖率），不是本系统的 RAG 准确率，也不要外推。
    # 「17 家公司里……的有哪几家」这类问题要的是**全部公司的同一张表**，不是
    # 「最相似的 6 条」，Top-K 在这里是结构性错配（实测三路 hit@10 全为 0）。
    # 所以由元数据直接定位表组，不参与打分（详见 src/retrieval/structured.py）。
    #
    # ⚠ 代价要说清：这条路给出的证据数**不再等于 TOP_K_FINAL** —— 17 家公司
    # 就是 17 条证据（+ 相似度补充 STRUCTURED_SUPPLEMENT 条）。需求里的「Top-6」
    # 描述的是相似度路径；筛选题把它放宽是刻意的，因为 6 条装不下 17 家公司，
    # 少一家就会漏掉一个正确答案。页面上会显式标出「本次走结构化路由」。
    structured_routing: bool = _env_bool("STRUCTURED_ROUTING", True)
    # 定位表组的小节关键词（用「包含」匹配，容忍 `六、`/`七、`/`六、 ` 三种写法）
    structured_subsection: str = _env("STRUCTURED_SUBSECTION", "主要会计数据")
    # 结构化表组之外，再补几条 RRF 召回。「……的原因是什么」这类复合问题里，
    # 解释往往在第三节正文，而那是相似度路确实擅长的部分。
    structured_supplement: int = _env_int("STRUCTURED_SUPPLEMENT", 6)

    # ---- 路径 ----
    root: Path = ROOT
    data_dir: Path = _path("DATA_DIR", "./data")
    raw_dir: Path = _path("RAW_DIR", "./data/raw")
    pdf_dir: Path = _path("PDF_DIR", "./data/pdf")
    processed_dir: Path = _path("PROCESSED_DIR", "./data/processed")
    index_dir: Path = _path("INDEX_DIR", "./index")
    eval_dir: Path = _path("EVAL_DIR", "./evaluation")

    # ---- 采集 ----
    download_delay: float = _env_float("DOWNLOAD_DELAY", 1.0)
    download_max_retry: int = _env_int("DOWNLOAD_MAX_RETRY", 3)
    user_agent: str = _env(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Financial-RAG-Coursework/0.1",
    )

    @property
    def company_list_path(self) -> Path:
        return self.data_dir / "company_list.csv"

    def assert_llm_ready(self) -> None:
        """调 LLM 前先调它，把「忘了填 key」变成明确报错而不是 401。"""
        if not self.llm_api_key or self.llm_api_key.startswith("sk-xxx"):
            raise RuntimeError(
                f"LLM_API_KEY 未配置。请复制 .env.example 为 .env 并填入真实 key"
                f"（当前 provider={self.llm_provider}, model={self.llm_model}）"
            )

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.pdf_dir, self.processed_dir,
                  self.index_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)


config = Config()


if __name__ == "__main__":
    import platform
    import sys

    print(f"Python      : {platform.python_version()} ({sys.executable})")
    print(f"项目根目录   : {config.root}")
    print(f"公司名单     : {config.company_list_path}"
          f"  {'存在' if config.company_list_path.exists() else '缺失'}")
    print(f"LLM         : {config.llm_provider} / {config.llm_model}")
    key_ok = bool(config.llm_api_key) and not config.llm_api_key.startswith("sk-xxx")
    print(f"  API key   : {'已配置' if key_ok else '未配置'}")
    print(f"Embedding   : {config.embedding_model} on {config.embedding_device}")
    print(f"检索         : 向量 {config.top_k_vector} + BM25 {config.top_k_bm25}"
          f" → Top {config.top_k_final}")
    try:
        import torch

        print(f"PyTorch     : {torch.__version__}, CUDA={torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch     : 未安装（装 CPU 版：pip install torch "
              "--index-url https://download.pytorch.org/whl/cpu）")
