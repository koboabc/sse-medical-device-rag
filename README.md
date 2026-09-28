# 上交所医疗器械财报 RAG

面向**上海证券交易所医疗器械上市公司 2026 年半年度报告**的可问答知识库。
金融研究生课程作业。

> 当前状态：**全流程已跑通**。17 份 2026 半年报（3362 页 / 305 万字）已采集解析为
> **4657 个片段**，BM25 + Qwen3-Embedding-0.6B 双路索引已建好（正文索引 `index/` 与
> 表头增强索引 `index/header/` 两套），混合召回 + 跨公司条件筛选增强已实测，
> 检索评测已完成（内部 20 题 + 正式 10 题逐题落盘，含索引 A/B），
> Streamlit 页面可交互（监听 `0.0.0.0`）。**生成层（DeepSeek）已跑通**：
> 正式 10 题拿到回答 10/10，逐题人工判定**答对 7 题**（含 2 道拒答陷阱
> **2/2 正确拒答**），10 题里没有一处凭空数字。
> 进度见 §8，实测数字见 §12，结论与课程要求对照见 §13。

> **刚 clone 下来请先读 §0**：仓库里只有代码与结论 —— 模型权重、财报 PDF、
> 检索索引这三样需要你自己准备，`.env` 也要自己建。

---

## 0. 从零跑起来（GitHub 上传版）

> 这个仓库是**可复现的代码与结论**，不是可复现的数据本身。1.2 GB 模型权重、
> 25 MB 财报 PDF、105 MB 索引都不在仓库里 —— 它们都能由「公开财报 + 本仓库代码 +
> 你自己的 `.env`」重新生成。下面按「装环境 → 补三样料 → 配密钥 → 起页面」排列。

### 0.1 仓库里有什么、没有什么

| | 内容 | 说明 |
|---|---|---|
| ✅ 上传 | `src/` `scripts/` `app.py` | 全流程代码：采集 / 解析 / 切分 / 建索引 / 混合召回 / 生成 / 评测 |
| ✅ 上传 | `evaluation/` | 内部 20 题评测集 + 正式 10 题逐题记录 + `results.xlsx` / `errors.xlsx` / 结论 PDF |
| ✅ 上传 | `data/company_list.csv`、`data/manifest.csv` | 17 家样本名单；公告标题、披露页直链、PDF sha256（均为公开信息，**无密钥**） |
| ✅ 上传 | `README.md` `requirements.txt` `.env.example` `.streamlit/config.toml` | 文档与配置模板 |
| ❌ 排除 | `.venv/`、`.env` | 虚拟环境；**密钥文件，唯一绝对不能入库的东西** |
| ❌ 排除 | `Models/Qwen3-Embedding-0.6B/`（1.2 GB） | 见 §0.3 第 1 条 |
| ❌ 排除 | `data/pdf/`、`data/chunks/`、`data/metadata.parquet` | 见 §0.3 第 2、3 条 |
| ❌ 排除 | `index/`（含表头增强变体 `index/header/`，共 105 MB） | 见 §0.3 第 4 条 |
| ❌ 排除 | `__pycache__/` `*.pyc` `*.log` `*.tmp` `*.bak` `node_modules/` `~$*.xlsx` | 缓存、日志与 Office 锁文件 |

排除规则写在根目录 `.gitignore`，原则一句话：**凡是能由「公开数据 + 代码 + `.env`」
重新生成的，一律不入库**。想确认某个文件到底会不会被上传，用退出码判断：

```bash
git check-ignore -q data/manifest.csv; echo $?   # 1 = 未被忽略（会上传）
git check-ignore -q .env;               echo $?   # 0 = 被忽略（不会上传）
```

### 0.2 环境安装

```bash
cd 上交所医疗器械财报RAG
python -m venv .venv

# Windows
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Linux / macOS
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

- 本机实测 **Python 3.11 / Windows 11**，依赖里的 `torch==2.6.0` 与 `faiss-cpu`
  装的就是 **CPU 版**。本项目**全程 CPU 推理**（开发机无 NVIDIA GPU，只有
  Intel Iris Xe 核显），**不要**改装 `faiss-gpu` 或 CUDA 版 torch。
- **别用 anaconda base 装** —— base 里的 `transformers 2.1.1` 会盖掉依赖（§3 记了坑）。
- 自检：`python scripts/check_env.py`（打印 Python / torch / 各项依赖是否齐全）。

### 0.3 三样必须自己准备的东西

**① 向量模型 Qwen3-Embedding-0.6B（约 1.2 GB）—— 必须单独下载，仓库不含权重**

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir Models/Qwen3-Embedding-0.6B

# 国内网络可用镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    Qwen/Qwen3-Embedding-0.6B --local-dir Models/Qwen3-Embedding-0.6B
```

下完后目录里应有 `config.json` / `model.safetensors` / `tokenizer.json` 等。

> ⚠️ **大小写坑（已知，未改代码）**：`src/config.py` 里 `EMBEDDING_MODEL` 的默认值是
> **小写** `models/Qwen3-Embedding-0.6B`。Windows 路径不区分大小写，本机直接跑得通；
> **Linux / macOS clone 下来必须二选一**：把权重下到小写 `models/`，或在 `.env` 里写
> `EMBEDDING_MODEL=./Models/Qwen3-Embedding-0.6B`。这个默认值属于核心配置，
> 按「不修改 RAG 核心代码」的要求原样保留。

**② 财报 PDF（17 份，约 25 MB）—— 必须单独下载，仓库不含原文**

`data/manifest.csv` 已存好每份公告的**披露页直链**，直接跑下载脚本即可：

```bash
python src/ingest/download_reports.py             # 按 manifest 批量下到 data/pdf/
python src/ingest/download_reports.py --limit 1   # 先试跑 1 份
```

- 下载要过 `static.sse.com.cn` 的阿里云 WAF（`acw_sc__v2` JS 挑战），脚本用
  `scripts/waf/sse_cookie_solver.js` 解一次 cookie 再复用。**首次需装该目录依赖**：
  `cd scripts/waf && npm install && cd ../..`（需要本机有 Node）。
- 公告清单本身已入库，所以**不需要**重跑 `src/ingest/fetch_announcements.py`
  （那是当初生成 `manifest.csv` 用的脚本，同样依赖 WAF 求解器）。
- 想跳过采集只复现检索与生成：把自备 PDF 放进 `data/pdf/`，文件名与 `manifest.csv`
  的 `file_name` 列对齐即可。

**③ 解析切分 → chunk（约 7.7 MB，全量约 133 秒）**

```bash
python src/indexing/parse_reports.py                     # 17 份 → 4657 个片段
python src/indexing/parse_reports.py --limit 1 --verbose # 单份调试
```

产物：`data/chunks/*.jsonl` + `data/metadata.parquet`（行序有契约，见 §11）。

**④ FAISS + BM25 索引（105 MB）—— 必须本地构建**

```bash
# 正文索引 → index/faiss + index/bm25
python src/indexing/build_index.py

# 表头增强索引 → index/header/（§12.3 的 A/B 实测：三路都更好，推荐演示用它）
INDEX_DIR=./index/header python src/indexing/build_index.py --header
```

- 4657 条 chunk 全量编码在 CPU 上约 **6 小时**（实测吞吐与分批曲线见 §12.6）。
  中断了直接重跑，**已算过的 batch 不会重算**（断点状态写在 `index/*/emb_state*.json`）。
  跑完前别关机；睡眠只是暂停，进度不丢。
- 建完自检：
  ```bash
  python scripts/verify_index.py
  python scripts/verify_retrieval.py --index index/header
  ```

### 0.4 配置 DeepSeek API Key

```bash
cp .env.example .env
# 然后用编辑器填 .env：
#   LLM_PROVIDER=deepseek
#   LLM_BASE_URL=https://api.deepseek.com
#   LLM_MODEL=deepseek-chat
#   LLM_API_KEY=sk-你自己的key
# 演示与评测建议再加一行（用表头增强索引）：
#   INDEX_DIR=./index/header
```

- `.env.example` 里的 key 是 32 个 `x` 的**占位符**，直接跑会被拦住并提示未配置。
- `.env` 已被 `.gitignore` 排除 —— **任何时候都不要 `git add .env`**。
- 只验证「是否就绪」、不回显 key：
  ```bash
  python -c "import sys; sys.path.insert(0,'.'); from src import config; config.assert_llm_ready(); print('LLM 就绪:', True)"
  ```
- 全流程只有生成层（DeepSeek）用到这一个外部服务；密钥若泄露请立刻在控制台吊销。

### 0.5 启动 Streamlit 问答页

```bash
# 默认正文索引
.venv/Scripts/python.exe -m streamlit run app.py

# 推荐：表头增强索引
INDEX_DIR=./index/header .venv/Scripts/python.exe -m streamlit run app.py
```

浏览器打开 <http://localhost:8501>。**第一次问一个走向量路的问题会加载 0.6B 模型，
约 30 秒**，之后是秒级；页面会展开显示两路召回、RRF 分数、软加权是否触发、
以及最终引用的「公司 + 章节 + 页码 + chunk_id」。
`.streamlit/config.toml` 监听 `0.0.0.0:8501`（同网段设备可访问，注意会消耗 API 额度；
只想本机用可加 `--server.address 127.0.0.1`）。

### 0.6 数据来源与合规

- 全部数据来自**上海证券交易所官网**公开披露的 2026 年半年度报告
  （`www.sse.com.cn` 披露页 → `static.sse.com.cn` PDF 直链）。`data/manifest.csv`
  逐份记录 `stock_code / security_name / title / url / sha256`，可逐条回溯核对。
- 17 家样本的筛选口径见 §2；公告清单字段说明见 §9。
- 仓库内**不含** PDF 原文、模型权重、检索索引或任何密钥。`evaluation/` 里保留的
  少量回答摘录与引用，出处均已标到「公司 + 章节 + 页码 + chunk_id」，仅用于课程评测。

---

## 1. 项目目标

把 10 家以上沪市医疗器械公司的 2026 半年报做成一个能问答的 RAG 系统，并给出
可复现的检索/生成评测。典型问题形态：

- 「联影医疗 2026 上半年营收多少？同比增长多少？」
- 「哪些公司上半年归母净利润同比下滑？下滑原因是什么？」
- 「对比心脉医疗和惠泰医疗的研发投入强度。」
- 「新华医疗的高值耗材业务板块毛利率是多少？」

## 2. 样本口径

| 项 | 取值 |
|---|---|
| 交易所 | 上海证券交易所（主板 + 科创板） |
| 行业 | 医疗器械 |
| 报告期 | 2026 年半年度 |
| 报告类型 | 半年度报告（全文，非摘要） |
| 公司数量 | **17 家**（要求 ≥10） |

名单见 [`data/company_list.csv`](data/company_list.csv)，
逐家的**核实等级与证据**见 [`docs/company_list_notes.md`](docs/company_list_notes.md)。

**17 家构成**：上证主板 6 家 / 科创板 11 家。

| 代码 | 简称 | 板块 | 细分 |
|---|---|---|---|
| 600055 | 万东医疗 | 主板 | 医学影像设备 |
| 600587 | 新华医疗 | 主板 | 医疗设备 |
| 603309 | 维力医疗 | 主板 | 医用耗材 |
| 603387 | 基蛋生物 | 主板 | 体外诊断 |
| 603658 | 安图生物 | 主板 | 体外诊断 |
| 605369 | 拱东医疗 | 主板 | 医用耗材 |
| 688013 | 天臣医疗 | 科创板 | 外科器械 |
| 688016 | 心脉医疗 | 科创板 | 高值耗材 |
| 688050 | 爱博医疗 | 科创板 | 眼科器械 |
| 688085 | 三友医疗 | 科创板 | 骨科植入 |
| 688108 | 赛诺医疗 | 科创板 | 高值耗材 |
| 688271 | 联影医疗 | 科创板 | 医学影像设备 |
| 688277 | 天智航 | 科创板 | 手术机器人 |
| 688289 | 圣湘生物 | 科创板 | 体外诊断 |
| 688351 | 微电生理 | 科创板 | 电生理介入 |
| 688358 | 祥生医疗 | 科创板 | 超声诊断设备 |
| 688617 | 惠泰医疗 | 科创板 | 电生理介入 |

> ✅ 上表此前有 4 家（600055 万东医疗、603658 安图生物、688016 心脉医疗、
> 688289 圣湘生物）只核实到「沪市 + 医疗器械」，半年报未确认。现已通过上交所
> 官方公告接口逐一查证，**4 家全部已披露 2026 年半年度报告**，无需启用替换预案。
> 公司名以官方 `FULL_NAME` 字段为准（600055 现名「北京万东医疗科技股份有限公司」）。

## 3. 环境实况（2026-09-25 实测）

| 项 | 结果 |
|---|---|
| Python | **3.11.4**，虚拟环境 `.venv`（`python -m venv .venv` 建的，见 §4） |
| PyTorch | **2.6.0+cpu 已安装**（`torch.cuda.is_available() == False`） |
| CUDA | **不可用** —— 本机无 NVIDIA 独立显卡（已确认，**不再尝试 CUDA**） |
| GPU 硬件 | 仅 Intel(R) Iris(R) Xe 核显（`DriverVersion 32.0.101.7085`） |
| CPU | Intel Core i5-1240P（4P + 8E = 12 核 / 16 逻辑核）；编码实测用满 ≈6 核 |
| 磁盘余量 | 172.7 GB |

**这决定了技术选型**：没有 CUDA，本地跑不动 LLM，也享受不到 GPU 加速。所以：

- **Embedding**：本地模型 **`Qwen3-Embedding-0.6B`**（1024 维，CPU 可跑，≈2.6 GB）。
  *初版选的是 `BAAI/bge-small-zh-v1.5`（512 维），因其上窗口只有 512 token、
  会切掉 113 个超长表格块（最长 13758 字）而更换，理由见 §7 第 4 条。*
- **向量检索**：`faiss-cpu`（不是 `faiss-gpu`）
- **答案生成**：一律走 API（DeepSeek / 通义千问 / OpenAI，OpenAI 兼容接口）
- **torch**：必须装 CPU 版，**不要**加 `--index-url .../cu121`，否则装了也白装

### ⚠️ anaconda base 环境有坑

base 里存在一个 **`transformers==2.1.1`**（2019 年的版本），且它**已损坏**：

```
ImportError: cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_'
```

原因是该版本引用了现代 urllib3 已移除的内部符号。而 `sentence-transformers`
需要 `transformers>=4.34`。**因此不要在 base 环境直接 `pip install`**，
请新建虚拟环境：

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
pip install -r requirements.txt
```

## 4. 快速开始

```bash
# 1. 建环境
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 2. 配密钥（无 GPU，LLM 必须用 API）
cp .env.example .env
#    编辑 .env，填入 LLM_API_KEY

# 3. 自检环境
python scripts/check_env.py

# 4. 校验公司名单（会重写为 utf-8-sig 并回读断言）
python scripts/normalize_company_list.py
```

> CSV 写成 **utf-8-sig（带 BOM）** 是为了中文 Windows 的 Excel 直接双击不乱码；
> pandas 用 `encoding="utf-8-sig"` 读即可。

## 5. 目录结构

```
上交所医疗器械财报RAG/
├── data/
│   ├── company_list.csv      # ★ 公司名单（17 家）
│   ├── manifest.csv          # 公告清单（PDF 直链 + 披露日期）
│   ├── raw/                  # 上交所原始公告元数据
│   ├── pdf/                  # 半年报 PDF（17 份，24.35 MB）
│   ├── chunks/               # 按份切分后的 jsonl（17 份）
│   └── metadata.parquet      # ★ 全部 4657 个 chunk 的元数据 + 正文
├── src/
│   ├── config.py             # 读 .env 的配置（集中所有默认值）
│   ├── ingest/               # fetch_announcements.py / download_reports.py
│   ├── indexing/             # parse_reports.py / build_index.py / embedder.py
│   ├── retrieval/            # store / entities / retriever / structured / generator / qa
│   └── utils/
├── index/                    # 正文索引：bm25/ + faiss/（不入库）
│   └── header/               # 表头增强索引（公司+代码+报告+章节 前置，不入库）
├── evaluation/
│   ├── datasets/qa.jsonl     # 内部评测集 20 题 + 3 拒答陷阱
│   ├── datasets/qa_formal10.jsonl  # ★ 正式评测集 10 题（见 §13）
│   └── results/              # ★ 评测结果与逐题记录（不入库）
├── screenshots/              # 演示截图
├── scripts/
│   ├── check_env.py          # 环境自检（只打印 key 是否就绪，不回显）
│   ├── verify_retrieval.py   # 行序契约 / 两路一致性 / 归一化 自检
│   ├── ask.py                # 召回 CLI（--no-llm 只看召回）
│   ├── eval_retrieval.py     # 检索评测（三路对照 + 消融）
│   ├── eval_generation.py    # 生成评测（逐题记录 召回/证据/回答/对错）
│   ├── compare_eval.py       # 两份结果逐题对比（救回/弄坏/名次变化）
│   ├── grep_chunks.py        # 出题与核对 gold 用
│   └── normalize_company_list.py / verify_*.py
├── docs/company_list_notes.md / sse_waf_access.md
├── app.py                    # ★ Streamlit 问答页
├── requirements.txt
└── .env.example
```

## 6. 采集策略（已执行完毕，记录当时的取舍）

17 份 PDF 已全部下载完成。当时的取舍：

- 单次请求间隔 ≥1s（`.env` 的 `DOWNLOAD_DELAY`），不给上交所网站压力
- 先小规模试：只拉 1 家的半年报，跑通「下载 → 解析 → 切分 → 索引」全链路再放量
- 半年报 PDF 有文本层，`pymupdf` 可直接抽取；**表格**用 `pdfplumber` 兜底
- 财务报表附注里的表格是问答的主要信息来源，切分时**表格不可跨页切断**

## 7. 技术路线（规划 → 实际）

1. **采集**：上交所公告页 → PDF → `data/pdf/{代码}_2026H1.pdf` ✅ 同规划
2. **解析**：pymupdf 抽正文 + pdfplumber 抽表格 → 结构化 JSON ✅ 同规划
3. **切分**：按章节（经营讨论、财务报表、附注）语义切分，表格整体成块 ✅ 同规划
4. **索引**：~~bge-small-zh~~ → **Qwen3-Embedding-0.6B**（本地，1024 维，查询侧
   非对称走 `prompt_name="query"`）+ jieba/BM25，双路索引
   *改因：bge-small-zh 上限 512 token，而本语料有 113 个表格块 >2000 字（最长
   13758），截断即丢正文。0.6B 原生窗口 32768，本语料最长 8913 token 从未接近。*
5. **检索**：混合召回 → ~~可选重排~~ → **RRF 融合 + 规则软加权** → Top-K
   *改因：需求明确排除 Reranker 与 Query Rewrite；RRF 无参、可复现、可解释。
   另有一条**结构化表组路由**（免打分，见 §12）。*
6. **生成**：OpenAI 兼容 API（DeepSeek），带引用来源回答 ✅ 同规划
7. **评测**：自建问答集 + 检索命中率 / 答案忠实度 ✅ 同规划，见 §12

## 8. 进度

- [x] 项目初始化、目录骨架
- [x] 环境确认（Python / PyTorch / CUDA）—— 本机无 NVIDIA 卡，一律 CPU，不再尝试 CUDA
- [x] 公司名单 `data/company_list.csv`（17 家，要求 ≥10）
- [x] 复查 4 家 C 级公司的 2026 半年报披露情况 —— **4 家全部已披露**
- [x] 公告清单 `data/manifest.csv`（17/17 命中，PDF 直链取自官方接口字段）
- [x] 批量下载 17 份 PDF（17/17 成功，24.35 MB；WAF 已绕过，见 `docs/sse_waf_access.md`）
- [x] 解析（pymupdf 正文 + pdfplumber 表格，3362 页 → 305 万字）
- [x] 切分（**4657 chunks** → `data/chunks/` + `data/metadata.parquet`）
- [x] 索引：BM25（jieba）+ FAISS/Qwen3-Embedding-0.6B 双路 → 正文索引 `index/`
- [x] 检索层：两路召回 → RRF 融合 → 规则软加权 → Top-6
- [x] 跨公司条件筛选增强（结构化表组，见 §12.4）
- [x] 演示页 `app.py`（Streamlit，监听 0.0.0.0，启动不重算任何东西）
- [x] 检索评测：内部 20 题 + 3 陷阱题，三路对照与消融（见 §12）
- [x] 正式评测集 10 题（3 道跨公司题）与逐题记录脚本（见 §13）
- [x] 表头增强索引 `index/header/`（`batches_done 1165/1165`、`rows_written 4657`、
      `encode_sec 44512` ≈12.4 h，一次跑完）；`verify_retrieval.py --index index/header` 全绿
- [x] 索引 A/B：同一份 20 题同开关，三路对照 + 逐题比 `救回/弄坏`（见 §12.3）
      —— 混合路 hit@10 **0.300 → 0.700**、MRR@10 0.172 → 0.332，**三路均弄坏 0 题**
- [x] 正式 10 题检索结果逐题落盘（`evaluation/results/`，含每题的 BM25/向量/混合召回）
- [x] 生成层跑通（DeepSeek `deepseek-chat`）：正式 10 题回答 10/10，引用越界 0 次
- [x] 正式 10 题的「是否答对 / 错误原因」逐题判定（**答对 7 题**，2 道陷阱 2/2 正确拒答）
      → `evaluation/results/formal10_per_question.md` + `review_formal10.json`
- [x] 课程要求的 10 题（`evaluation/questions.xlsx`，编号 **R01–R10**）：8 单公司 +
      2 跨公司，17 家公司各出现一次；标准答案由脚本机械核对（32 个数字 token 逐个
      在 gold 原文里查到，查不到即拒绝写文件）；附检索自检 7/10（见 §13.7）
- [x] R01–R10 跑通生成层并逐题判定：**答对 7/10**（单公司 7/8、跨公司 0/2），
      3 道错题全在召回侧 → `evaluation/results.xlsx` + `evaluation/errors.xlsx`（见 §13.7.2）
- [x] 结论 PDF（6 页，`evaluation/R01-R10_评测结论.pdf`）：按题型给「答得好 / 翻车 / 为什么」
      —— 核心是一条无例外的规律：**gold 进最终证据的 7 题全对、gold 掉出的 3 题全错**
- [x] GitHub 上传版整理：`.gitignore` 按「可由公开数据 + 代码 + `.env` 重新生成的一律
      不入库」排除模型 / PDF / 索引 / `.env`；密钥扫描结论：上传集内**无 API Key、密码或 Token**；
      上传版使用说明见 **§0**（仓库内 78 个文件 / 1.8 MB）
- [ ] 演示页截图（`screenshots/`，需人工截，见 §13.8）

## 9. 公告清单（manifest）

`data/manifest.csv` 由 `src/ingest/fetch_announcements.py` 从上交所官方接口生成：

```bash
python src/ingest/fetch_announcements.py     # 生成 manifest
python scripts/verify_manifest.py            # 核验（含随机抽样回源比对）
```

数据源与要点：

- **公告查询**：`query.sse.com.cn/security/stock/queryCompanyBulletin.do`
  - 必须带 `Referer: https://www.sse.com.cn/`，必须给 `securityType`
  - `reportType2=DQBG` 圈定「定期报告」——拿到半年报的关键参数
  - 记录里的 `BULLETIN_TYPE` 直接标注「半年报」，比标题匹配可靠
- **公司全称**：`query.sse.com.cn/commonQuery.do` 的 `FULL_NAME` 字段
  - 文件名用**官方全称**，不用人工填写 —— 人工名会过时。例如 600055
    在 2017-05-24 已由「华润万东医疗装备股份有限公司」更名为
    「北京万东医疗科技股份有限公司」

筛选口径（对应作业要求 1–4）：只取标题以「2026年半年度报告[全文]」结尾的记录，
摘要 / 英文版 / 修订版 / 补充公告均因标题形态不同而自然落选。

> ⚠️ 已发现的坑：圣湘生物的报告标题是「…2026年半年度报告**全文**」，
> 若正则只锚定「报告$」会把它误判为缺失。已修正。

`data/missing_reports.csv` 目前**只有表头**（17 家全部找到）。

## 10. 下载 PDF

```bash
python src/ingest/download_reports.py              # 全量下载
python src/ingest/download_reports.py --limit 2    # 先试 2 家
```

6 线程 / 读取超时 60s / 失败重试 3 次 / 支持断点续传（HTTP Range）/
已存在且校验通过的 PDF 自动跳过 / 单个失败不中断整批。
下载后校验 PDF 文件头 + 大小 + pymupdf 可解析，通过才落盘并回写 manifest
（`download_status=success`，失败则 `failed` 并记 `error`）。

**难点在 WAF**：`static.sse.com.cn` 的 PDF 直链有阿里云 WAF 的 `acw_sc__v2`
JS 挑战，直接请求返回 **200 但内容是 HTML**（很容易被误判成下载成功）。
需先用 `scripts/waf/sse_cookie_solver.js`（Node + jsdom）算出 cookie 再下载。
完整现象、失败的尝试、以及"冷连接随机卡死"的坑，见
[`docs/sse_waf_access.md`](docs/sse_waf_access.md)。

> WAF 求解器需要 Node.js 与 jsdom：`cd scripts/waf && npm install`

## 11. 解析与切分

```bash
python src/indexing/parse_reports.py              # 全量（6 进程，约 133 秒）
python src/indexing/parse_reports.py --limit 1 --workers 1 --verbose   # 单份调试
python scripts/verify_chunks.py                   # 核验 + 随机抽查
```

**产出**：`data/chunks/{代码}_2026H1.jsonl`（每份一个）+ `data/metadata.parquet`。

| 指标 | 值 |
|---|---|
| 报告 | 17/17 成功，0 失败 |
| PDF 总页数 | 3362 |
| chunk 总数 | **4657**（表格 3389 / 正文 1268） |
| 字符总数 | 305 万 |
| 正文 chunk | 中位 **902** 字，67% 落在 500~1000 |
| 耗时 | 133 秒（6 进程） |

每个 chunk 的字段：`chunk_id`、`stock_code`、`company`、`report`、`section`、
`page_start`、`page_end`、`content`、`content_type`（另加 `subsection`、
`char_count` 辅助字段）。`content_type` 取 `text` / `table`。

### 关键实现选择

- **页眉页脚靠频次识别**：不写死正则，而是统计「出现在 ≥30% 页面上的行」。
  各家公司页眉格式不一，频次法一次覆盖全部 17 份。
- **目录页整页跳过**：目录会逐行列出「第一节」「第二节」…，每行都像章节标题。
  判据是「一页出现 ≥3 个第X节」（正文里每节独占页首，不会连排）。
- **表格跨页合并**：合并资产负债表这类长表会断在两页上（如 688617 的
  35 行 + 48 行）。相邻页、列数相同、同章节即判定为续表，合并后
  `page_start`/`page_end` 仍如实记录跨页范围 —— 表格既不拆碎，页码也不丢。
- **表题并入表格块**：附注的排版是「`40、应交税费` / `(1) 明细情况` / 表格」。
  向上回溯把这一簇短块收进表格块当前置说明，否则会残留一堆 7 字孤儿块
  （初版 190 个碎块，现降至 82 个，且剩下的都是「整节内容就是几张表」的
  合理节首）。
- **正文与表格按 y 坐标交错排序**：初版把正文和表格分两个循环产出，
  导致同一页内的章节归属错乱（页内所有正文被标成该页**最后**的章节）。

### 已知残留（如实记录，未做规则清理）

- **0.25% 的正文块混入孤立数字行**（5/1268 块，如 `121,594,548.48` 单独成行）：
  pdfplumber 漏检的小表格，其数字列掉进了正文流。修复规则有误删正文的风险，
  收益（0.25%）不抵风险，保留并记录。
- **113 个表格块 >2000 字**（最长 13758 字）：要求「表格不得拆碎」的直接代价。
  后续做 embedding 时需注意——`bge-small-zh` 上限 512 token，超长表格块要么
  按行分组并重复表头（组内仍可理解），要么单独走关键词检索。
  *（最终改用 Qwen3-Embedding-0.6B，原生窗口 32768，这条顾虑随之消失；见 §7）*

## 12. 检索层与评测（实测数字）

### 12.1 评测集

`evaluation/datasets/qa.jsonl`：**20 题 + 3 题拒答陷阱**，四类各 5 题 ——
单点数值 / 同比与原因 / 跨公司对比 / 条件筛选。每题带 `gold_chunk_ids`（**逐条
读过原文核对**）、`gold_answer`、`note`（注明页码与核对命令）。

**正式结论只引 10 题那一份**：`evaluation/datasets/qa_formal10.jsonl`（10 题，
含 3 道跨公司题 + 2 道拒答陷阱，五个题型全覆盖）。它是从 `qa.jsonl` 按 qid 筛出来的，
题干与 gold 逐字相同；其余题继续做内部回归。选题理由与逐题考点见
[`evaluation/datasets/README.md`](evaluation/datasets/README.md)。

字段口径写在 `scripts/eval_retrieval.py` 的 docstring 里，四处关键决定：

- 命中判定把「内容重复被合并掉的块」（`Hit.dup_chunk_ids`）算作覆盖——语料里
  39 对 chunk 正文两两相同（同一张表跨页切了两块），不该算漏召回
- 拒答陷阱不进检索指标（没有 gold，命中与否无意义）
- gold 条数 > 评测深度时 recall 上不去是**结构上限**，不是召回失败（脚本会点名）
- 软加权三路都开（它是管线的一部分，关掉跑出的基线不对应线上行为）

### 12.2 三路基线（正文索引，20 题，深度 Top-10）

| 路 | hit@5 | hit@10 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | 0.050 | 0.150 | 0.050 | 0.100 | 0.066 |
| 向量 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 混合 | 0.000 | 0.050 | 0.000 | 0.050 | 0.005 |

**混合低于单 BM25**，这条数字单独拿出来看会得出「拆掉融合」的错误结论。逐题看
（`scripts/compare_eval.py`）才知道真相：RRF 把 BM25 的 3 个命中稀释成了 1 个，
而向量路 20 条全是噪声 —— 该修的是向量路本身，不是融合。**汇总会骗人，逐题不会。**

向量路 0.000 的成因已定位（不是模型不行）：它能**正确锁定公司**（Q02 返回的全是
688016 的块、Q03 全是 688617、Q07 全是 688289），但公司在 4657 条里只占 1/17，
锁定公司后仍要在一家公司内部 240~380 个块里挑对那一块，而库里根本没有这条信息 ——
「主要会计数据」表正文里**不含公司名**，17 家公司的同名表几乎同构。同一批查询换个
说法（只留内容词）gold 就进 Top-9，说明模型与索引都没问题，是**输入缺了判别特征**。

### 12.3 修法 A：索引文本前置元数据表头（**已完成，A/B 已实测**）

`build_index.py --header` 把 `公司 代码 报告 章节 小节。` 拼到每条 chunk 正文前再
编码/分词，产物写 **`index/header/`**（不覆盖现有索引：文本一变批计划指纹就变，
`open_embeddings()` 会把已有的 `embeddings.npy` 直接 unlink —— 那一份是 4 小时算的）。

**索引已建完**：`index/header/faiss/build_info.json` 记 `batches_done 1165/1165`、
`rows_written 4657`、`encode_sec 44512`（≈12.4 小时，一次连续跑完，
`resumed_from_batch 0`）。`python scripts/verify_retrieval.py --index index/header`
全绿，含**「沿长度谱均匀抽 4 条重编码，最低余弦 = 1.000000」**（证明行号没串位）、
`allow_mask(['688271'])` 命中 378 块、融合与加权不变量全过。

**A/B 口径**：同一份 20 题、同样开关（结构化路由**开**、软加权**开**、深度 Top-10、
`k_bm25 = k_vector = 20`），只换索引：

| 路径 | hit@5 | hit@10 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 正文 | 0.250 | 0.400 | 0.122 | 0.203 | 0.233 |
| BM25 表头 | **0.500** | **0.550** | **0.272** | **0.303** | **0.306** |
| 向量 正文 | 0.200 | 0.250 | 0.072 | 0.103 | 0.167 |
| 向量 表头 | **0.300** | **0.700** | 0.147 | **0.403** | **0.237** |
| 混合 正文 | 0.200 | 0.300 | 0.072 | 0.153 | 0.172 |
| 混合 表头 | **0.500** | **0.700** | **0.272** | **0.403** | **0.332** |

逐题比对（`scripts/compare_eval.py`，两份结果都在 `evaluation/results/`）：
**BM25 救回 3 题（Q08 / Q10 / Q15）、向量救回 9 题
（Q03/Q05/Q07/Q08/Q09/Q12/Q13/Q14/Q15）、混合救回 8 题
（Q05/Q06/Q07/Q08/Q09/Q10/Q14/Q15）——三路都是「弄坏 0 题」。**
一条没弄坏是这次改动最值得记的性质：表头只增加判别特征，不改变语料本身。

> ⚠️ **口径变更说明**：本节早先那张 `0.050 / 0.150 → 0.300 / 0.300` 的表是
> **BM25 单路 + 结构化增强关**的基线（今天复跑仍得 0.050 / 0.150，可复现），
> 与上表口径不同，**不要互相引**。上表是当前全开关口径，与 §12.4 一致。

#### ⚠️ 旗舰题 Q01 没能救回来 —— 这条必须说实话

Q01（联影医疗营收，gold `688271-0009`）在**两套索引 × 三路共 6 种配置下全错**，
且把向量召回深度从 20 提到 40（`--k-vector 40`）之后**仍然不进 Top-10**。

表头的效果是把它的全库排名从「毫无希望」抬到「就差一点」：

| 旗舰题 gold `688271-0009` | 全库排名 |
|---|---|
| BM25 正文 → 表头 | #795 → #132 |
| 向量 正文 → 表头 | #620 → **#31** |

但 #31 依然落在 Top-20 召回深度**之外**；即使把深度放到 40 让它进了候选池，
**RRF 融合后的 Top-10 截断仍把它挡在外面**。也就是说：症结已经从「召不回」变成
**「融合截断」**，而「调大深度就能救」这个曾经的猜测**已被实测证伪**
（不是没试，是试了没用）。

这里更正一条早先写在本节的错误说法：*「真正越过第六条阈值的是向量路（对抗集实测
cos 0.3433 → 0.6659，现第 6 名 0.6605）」*。那个 0.6659 是**单条对抗探针的余弦估计**，
不是真实检索的结果；按真实评测，Q01 的 gold 在表头向量下排 **#31**，
在 Top-20 深度外，从未进过第 6 名。**以真实排名为准，不引用那条估计。**

### 12.4 跨公司条件筛选增强：结构化表组路由（**不是通用准确率**）

> ⚠️ **先说定位**。这是为「17 家公司里……的有哪几家」这**一类**问题加的确定性取表，
> 触发条件严格（要一个集合 + 问题里没点名公司）。下面出现的 **17/17** 是
> 「表组覆盖了几家公司的表」，是**覆盖率**，**不是本系统的 RAG 准确率**，
> 不能外推到其他题型 —— 另外 15 题一条未变就是它不外溢的证据。
> 它也不保证答对：表组只把该看的表摆上桌，读表算数是模型的事。

`src/retrieval/structured.py`，**零重算**。「17 家公司里……的有哪几家」要的是
**全部公司的同一张表**，不是「最相似的 6 条」—— 相似度检索在这里是结构性错配，
这也是 5 道条件筛选题三路 hit@10 全为 0 的原因。

确定性来自语料的一条硬结构：`subsection` 含「主要会计数据」的块共 **34 张表，
恰好覆盖 17 家公司各 2 张**（会计数据表 + 主要财务指标表；标题有 `六、`/`七、`/
`六、 ` 三种写法，所以用「包含」匹配）。表族靠**行标签投票**区分，不靠关键词：
块是按页切的，数据表块尾会捎带下一小节的标题「(二) 主要财务指标」，所以「财务指标」
四个字在 34 张表里**全都有**，单判它会 34/34 全错（踩过，自检里已断言 17/17）。

正文索引、BM25 单路、条件筛选 5 题：

| 条件筛选 | hit@5 | hit@10 | recall@10 | MRR@10 | 证据覆盖 |
|---|---|---|---|---|---|
| 关（消融） | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 开 | **0.800** | **1.000** | 0.412 | **0.669** | **17.0 / 17 家公司** |

全部 20 题：hit@10 0.150 → 0.400，MRR@10 0.066 → 0.233。**其余 15 题一条未变**
（同一份 20 题逐题比对，无救回也无弄坏）—— 因为触发条件是「要一个集合且没点名
公司」，点名了就退回相似度路。

**两条必须说清的口径**：
1. 条件筛选题的 gold 就是这 34 张表，结构化路由取的也是这一组，所以
   「hit@10 从 0 到 1」在定义上必然发生 —— **它不构成效果证据**。不循环的度量是
   「证据里到齐了几家公司的表」：**17/17**。少一家就漏一个正确答案。
2. 这类问题的证据数**不再是 Top-6**，而是「17 家公司各 1 张 + 6 条相似度补充」=23 条。
   需求里的 Top-6 描述的是相似度路径；6 条装不下 17 家公司，这是刻意放宽，页面上会
   显式标出「本次走结构化表组路由」并给出覆盖公司数。

### 12.5 复现

```bash
# 检索评测（三路逐题落盘，产物含每题的 BM25 / 向量 / 混合召回结果）
python scripts/eval_retrieval.py --routes bm25,vector,hybrid      # 正文索引
python scripts/eval_retrieval.py --index index/header             # 表头增强索引
python scripts/eval_retrieval.py --no-structured                  # 消融：关条件筛选增强
python scripts/eval_retrieval.py --dataset evaluation/datasets/qa_formal10.jsonl
python scripts/compare_eval.py A.json B.json                      # 逐题比两份结果

# 生成评测（逐题记录 召回/证据/回答/标准答案/是否答对/错误原因）
python scripts/eval_generation.py --no-llm                        # 无 key：先出召回与证据
python scripts/eval_generation.py                                 # 配好 key：跑完整链
python scripts/eval_generation.py --report --review evaluation/results/review_formal10.json

# 自检
python scripts/verify_retrieval.py --index index/header           # 行序/两路一致/归一化
python src/retrieval/structured.py                                # 表族自检（断言 17/17）
python scripts/grep_chunks.py -k 主要会计数据 -s 第二节 --full     # 核对 gold
```

结果落在 `evaluation/results/*.json`（文件名带索引变体标签），每份都记了当时的
`index_variant` / `structured` / `weights` 开关 —— 否则几份结果放一起会认不出谁是谁。

**用哪套索引是配置项，不是改代码**：默认仍是正文索引（`index/`），跑表头索引加
`--index index/header` 或把 `INDEX_DIR=./index/header` 写进 `.env` 即可
（演示页同理：`INDEX_DIR=./index/header streamlit run app.py`）。两套索引的
`chunk_id` 行序**完全相同**（都出自同一个 `load_chunks()`），所以同一套题与 gold
可以直接对比。按 §12.3 的 A/B 结果，**表头索引在三路上都更好且没弄坏任何一题**，
建议展示与截图时用它；这里没有改默认值，是因为 `build_index.py --header` 是按
「`index/` 相对 + `header/` 子目录」定位产物的，改默认会让它写到 `index/header/header/`。
（顺带修掉一个会误报的隐患：`store.describe()` 原先按「与配置默认相等 ⇒ 正文」推
索引名，一旦你把 `INDEX_DIR` 指到 `index/header`，它会把表头索引显示成「正文」——
现在改成看目录实际名字，配成什么就报什么。）

### 12.6 断点续算：算过的 embedding 一条都不重算

0.6B 模型在 CPU 上编码 4657 条是按小时计的（正文索引那次实测 `encode_sec` 480 秒
只是**补算**的部分，全量要 4 小时上下），中断一次就前功尽弃是不可接受的。所以
编码按 token 长度降序分批，**每批算完立刻写盘 + 落状态**：

- 向量写进 `embeddings.npy`（`np.memmap`），按**行号**回写 —— 编码次序 ≠ 行序
- 进度写进 `embeddings.state.json`：`batches_done` / `rows_written` / `order_fp` / `real_tokens`
- 重启时读状态，`batches_done` 之前的一律跳过

**实测证据**（`index/faiss/build_info.json`，正文索引那一次）：

| 字段 | 值 |
|---|---|
| `resumed_from_batch` | 943 |
| `encoded_batches_this_run` | 222 |
| `encode_sec`（本次实际计算） | 480.2 |
| `wall_sec` | 495.1 |

即 1165 批里，前 **943 批是上一轮算好的、本次一条没重算**，只补了后 222 批。

**安全阀**：状态里存了 `order_fp = sha1(编码顺序)` 与 `shape / dim / total`。索引
文本一变（例如 `--header` 给正文前置了表头），指纹就对不上，`open_embeddings()`
会**删掉旧 memmap 从头算** —— 这是刻意设计：新旧向量来自不同输入口径，混在同一个
索引里是错的。这也是 `--header` 必须写到 `index/header/`、不能就地覆盖的原因
（正文索引那份是 4 小时算出来的）。

---

## 13. 结论

### 13.1 一句话结论

**「索引文本前置元数据表头」这一个改动，把 20 题内部集的混合召回 hit@10 从
0.300 抬到 0.700（MRR@10 0.172 → 0.332），三路救回 20 题次、弄坏 0 题次；
正式 10 题里 8 道有标答的题，混合路 hit@10 从 1/8 提到 4/8。**
跨公司条件筛选增强单独看是**覆盖率 17/17**（该题型专用，不是通用准确率）。
旗舰题 Q01 仍未召回 —— 症结已从「召不回」移到「融合截断」，且**调深度救不回来
（已实测证伪）**。

端到端（检索 → Top-6 证据 → DeepSeek → 带引用回答）在正式 10 题上跑通：
**回答 10/10、判定答对 7/10、2 道拒答陷阱 2/2 正确拒答、引用越界 0 次**；
3 道未答全的题**全部**是 gold 块没被召回导致的（模型明确说「证据里没有」，
没有一处编造）。**注意这两组数字是不同口径**：检索侧的 hit@10 4/8 是「gold 块有没有
进 Top-10」，生成侧的 7/10 是「答案对不对」——后者可以高于前者，因为同一个事实
在报告里常有多处表述（Q01 就是典型：gold 块没召回，答案仍对）。

### 13.2 正式 10 题逐题检索结果

口径：结构化路由开、软加权开、深度 Top-10、`k_bm25 = k_vector = 20`。
`✓` = 该路 Top-10 里出现了 gold 块。

| qid | 类型 | 正文·混合 | 表头·混合 | 说明 |
|---|---|---|---|---|
| Q01 | 单点数值 | ✗ | ✗ | ★旗舰题。向量全库 #620→#31，仍在 Top-10 外；深度提到 40 也救不回 |
| Q04 | 单点数值 | ✗ | ✗ | 万元单位陷阱。表头向量**原始**排名 #10，**被软加权挤出 Top-10**（见 §13.5-4） |
| Q06 | 同比与原因 | ✗ | ✓ #3 | 数字在表块、原因在正文块，需跨块 |
| Q08 | 同比与原因 | ✗ | ✓ #2 | 原因段标题与内容不符（标题写「非经常性损益」） |
| Q11 | 跨公司对比 | ✗ | ✗ | 两家 gold 分别 #661→#94、#1669→#34，均未进 Top-10 |
| Q13 | 跨公司对比 | ✗ | ✗ | 表头**向量单路** ✓ #8，但融合进 BM25 后被挤出 Top-10（同 Q01 的截断机制） |
| Q15 | 跨公司对比 | ✗ | ✓ #2 | 两家都必须取「财务指标表」而非会计数据表 |
| Q16 | 条件筛选 | ✓ | ✓ | 走**结构化表组**（17/17 覆盖），**不代表相似度召回能力** |
| T01 | 拒答陷阱 | — | — | 无 gold。以生成层「是否拒答」判定（**待配 key**） |
| T03 | 拒答陷阱 | — | — | 同上（该行被 PDF 抽取丢掉，全库 0 命中） |

**8 道有标答题的三路口径（表头索引）**：BM25 hit@5/hit@10 = 0.500/0.500、
向量 = 0.125/0.500、混合 = 0.500/0.500；正文索引下混合只有 0.125/0.125。
**注意**：表头索引这 4 次命中里，**Q16 来自结构化表组增强**（确定性取表，
不是相似度检索）—— 纯相似度召回是 **3/8**。这个拆分必须说，否则会把
「一个题型的专项增强」算进「通用召回能力」。

### 13.3 生成层：正式 10 题逐题判定

口径：表头增强索引、`deepseek-chat`、temperature 0.1、Top-6 证据（Q16 走结构化表组 23 条）。
判定由人读过回答后写进 `evaluation/results/review_formal10.json` —— **脚本不算对错**：
「58,802.29 元」和「58,802.29 万元」字符全同而量级差一万倍，自动比对判不出来。
完整逐题记录（问题 / 召回 / 最终证据 / 回答 / 标准答案 / 判定 / 错误原因）见
`evaluation/results/formal10_per_question.md`。

| qid | 类型 | 证据含 gold | 引用命中 | 判定 | 要点 |
|---|---|---|---|---|---|
| Q01 | 单点数值 | 0/1 | ✗ | ✅ | 答对 70.52 亿 / +17.22%；附带缺陷是把 84.06 亿标成「合并口径」，实为**母公司**口径 |
| Q04 | 单点数值 | 0/1 | ✗ | ✅ | **万元陷阱没踩**：588,022,932.18 元与 58,802.29 万元两种表述都给对 |
| Q06 | 同比与原因 | 1/2 | ✓ | ✅ | 净额 / 同比 / 原因三项全中 |
| Q08 | 同比与原因 | 1/2 | ✓ | ❌ | 比例与构成算得全对，但公司对「为什么上升」的表述没召回；模型明说证据里没有解释性表述 |
| Q11 | 跨公司对比 | 0/2 | ✗ | ❌ | 结论对、联影数字对；心脉取了母公司口径（gold 是合并口径） |
| Q13 | 跨公司对比 | 0/2 | ✗ | ❌ | 结论对、赛诺 +263.59% 对；三友的 -24.18% 没召回 |
| Q15 | 跨公司对比 | 1/2 | ✓ | ✅ | 惠泰 12.89% > 心脉 10.04%，两家都对 |
| Q16 | 条件筛选 | 9/9 | ✓ | ✅ | 9 家全中；并正确处理了天智航「不适用」与爱博诺德缺失两个边界 |
| T01 | 拒答陷阱 | — | — | ✅ | 正确拒答：说明只有半年报、没有三季报数据，未用上半年推算 |
| T03 | 拒答陷阱 | — | — | ✅ | 正确拒答：说明该口径金额不存在并指出是利润表被截断；未拿扣非净利润冒充 |

**10 题里 0 处凭空数字、0 次引用越界**（`invalid_citations` 全空）。3 道 ❌ 的共同原因
只有一个：**需要的那一块没被召回**，而模型对此一律如实说明 —— 这正是「宁缺不编」
的设计意图，也说明**下一步该提升的是召回，不是换更大的模型**。

### 13.4 三条必须与数字同时出现的口径

1. **17/17 是覆盖率，不是准确率。** 它只说明「34 张主要会计数据/财务指标表
   恰好覆盖 17 家公司各 2 张」，且只对「要一个集合且没点名公司」的题型生效
   （§12.4）。另外 15 题一条未变，正是它不外溢的证据。
2. **表头索引的收益是「判别特征」带来的，不是「更多文本」。** 前置的
   `公司/代码/报告/章节/小节` 让原本同构、都不含公司名的会计数据表能被区分开；
   代价是索引文本变长（编码 12.4 小时），且**换文本就必须整体重算**（§12.6 安全阀）。
3. **两类数字口径不同，不要混引。** 「hit@10 = 4/8」是**检索侧**（gold 块有没有进
   Top-10）；「答对 7/10」是**生成侧**（答案对不对，由人判）。生成侧可以高于检索侧，
   因为同一事实在报告里常有多处表述（Q01 就是：gold 块没召回，答案照样对）。
   两组都不推算、不外推到语料以外的公司或年份。

### 13.5 已知不足（如实记录）

1. **Q01 / Q04 / Q11 在表头索引下仍召不回。** Q01 与 Q13 的失败机理是
   **RRF 融合后的 Top-10 截断**（候选池里有，融合排序排出去了），不是召回深度不够。
   下一步该做的是融合/去偏，不是继续加深召回。
2. **3 道答错的题（Q08 / Q11 / Q13）全是召回缺口导致的，不是模型能力问题** ——
   需要的那一块没进证据，模型就明确说「证据里没有」。**要提升的是召回**：
   把这三题缺的 gold 块（`688277-0009` / `688016-0008` / `688085-0008`）召回来，
   答案大概率就对了。这一条是可验证的下一步，不是猜测。
3. **生成只跑了一轮**（单次采样，temperature 0.1）。同一题的表述会变（实测 Q01
   两轮引用编号不同：`[1,3,6]` vs `[1,3]`），所以「7/10」应读作**一次采样的结果**，
   不是稳定上界；要报稳定值得多轮取多数。
4. **软加权消融（20 题内部集，另一组实测）**：加权对 BM25 有帮助
   （hit@10 0.400 → 0.550），**对向量路有害**（0.800 → 0.700，Q04 的 gold 就是
   被它从 Top-10 挤出去的），对混合路 hit@10 有害（0.700 → 0.650）但 MRR@10
   有益（0.332 → 0.371）。n=20 时 1 题 = 0.05，**这些差异在噪声量级**，
   因此**保持默认不变**，仅登记为一个可调杠杆，不作为结论。
5. **截图待补**：演示页截图需人工在浏览器里截（本机未装 playwright/selenium/imgkit，
   不为截图引入新依赖）。页面已在 `0.0.0.0:8501` 监听，按 §13.8 打开即可截。

### 13.6 课程要求对照

| 要求 | 状态 | 证据 |
|---|---|---|
| 10 家以上公司 | ✅ **17 家** | `data/company_list.csv`、§2 |
| 半年报全文 | ✅ 3362 页 / 305 万字 / **4657 chunk** | §11 |
| 文字 + 表格 | ✅ text 1268 + table 3389（`content_type`） | §11 |
| 公司 / 章节 / 页码 | ✅ 每条 chunk 带 `company/section/subsection/page_start/page_end` | `data/metadata.parquet` |
| 向量 + BM25 | ✅ Qwen3-Embedding-0.6B + FAISS ／ jieba + BM25 | §12 |
| 10 道题 | ✅ **两套**：`qa_formal10.jsonl`（Q+T）／ `questions.jsonl`（R01–R10，按主题铺），**两套都跑过生成层** | §12.1、§13.7 |
| 至少 2 道跨公司题 | ✅ 旧套 **3 道**（Q11 / Q13 / Q15）／ 课程套 **2 道**（R09 两家、R10 五家） | §12.1、§13.7 |
| 逐题记录召回 / 正确性 / 错误原因 | ✅ `formal10_per_question.md`（问题/召回/最终证据/回答/标准答案/判定/错误原因）+ `review_formal10.json` | §13.2、§13.3 |
| 代码 | ✅ | 本仓库 |
| 截图 | ⏳ 待人工截（`screenshots/`） | §13.8 |
| 一页结论 | ✅ §13.1–13.4 + `evaluation/results/conclusions.md` | — |

### 13.7 课程要求的 10 题（`evaluation/questions.xlsx`）

课程按主题规定了 10 道题（营业收入 / 净利润 / 毛利率或盈利能力 / 研发投入 / 经营现金流 /
主要业务变化 / 主要产品 / 风险因素 / 两家公司财务指标比较 / 多家公司综合比较），
成题落在 `evaluation/questions.xlsx`，编号 **R01–R10**。

**为什么不与上面那套共用 Q 编号**：`qa.jsonl` / `qa_formal10.jsonl` 里 Q01 是「联影营收」、
Q04 是「拱东营收」，这里是 R01「新华营收」—— 同名不同义在对读两份结果时会真出错。

| | 旧正式 10 题（Q + T） | 课程 10 题（R01–R10） |
|---|---|---|
| 文件 | `evaluation/datasets/qa_formal10.jsonl` | `evaluation/datasets/questions.jsonl` |
| 组成 | 8 有标答 + 2 拒答陷阱 | 8 单公司 + 2 跨公司 |
| 跨公司 | 3 道 | 2 道（R09 两家、R10 五家） |
| 生成层 | 已跑（答对 7/10，§13.3） | 已跑（答对 7/10，§13.7.2） |
| 用途 | 测端到端问答 + 幻觉抵抗 | 按主题覆盖课程要求 |

两套题**不是替代关系**：旧那套带拒答陷阱（T01/T03），是「系统扛不扛幻觉」的证据；
R01–R10 是按课程主题清单铺的面，是「覆盖够不够」的证据。两套都跑过生成层，
**答对数同为 7/10，且 3 道错题同样是召回缺口、都不是模型答错** —— 两次独立出题
得到同一个结论：瓶颈在召回层。没有合并，因为合并会让 10 题的配额被两套各占一半。

**排题刻意的两处设计**：

1. **17 家公司各出现一次**（8 道单公司题用 8 家，2 道跨公司题用另外 9 家）。这是排题时
   挑的，不是碰巧 —— 覆盖全部语料公司，任何一家公司的表读错都会在某题上暴露。
2. **标准答案由脚本机械核对，不靠人写**：`scripts/build_questions_xlsx.py` 把
   `expected_answer` 里的数字 token 抽出来（**32 个**），逐个去 gold 块正文里查
   （去千分位逗号后逐字比对）；查不到即 `[FAIL]` 并**拒绝写文件**。这条拦住的典型错误
   不是整段编造，而是 17×6≈100 个数字**记串**。

**三类值没写进 `expected_answer`**（都不是披露值，只在备注栏标明）：R03 的「上升
0.27 个百分点」是 71.36−71.09 的**推导**；R10 的「58,802.29 万元 = 588,022,900 元」
是**换算**（拱东那家单位是万元，全库唯一）；R10 的**排序**是推导。

#### 13.7.1 检索自检（sheet3）：10 题里 7 题 gold 全进 Top-10

题面定完后拿表头索引 + 混合召回跑了一遍，**只记录，不改题** —— 题目只由「半年报里
有没有明确答案」决定；按系统现在能不能召回挑题，评测集就变成给自己挑好做的题了。

| 结果 | 题 |
|---|---|
| gold 在 **rank 1** | R01 R02 R03 R04 R06 R08 |
| gold 在 Top-10（rank 4） | R07 |
| **未全中** | R05 R09 R10 |

三道未全中分**两种机理**（分路名次另跑一遍测，刻意传 `NO_WEIGHTS` 取**原始排名** ——
开着软加权测，加权本身就会改名次，分不清是融合截断还是加权截断）：

- **R05 / R09 ＝ 融合截断**（与 §12.3 旗舰题 Q01 同一机理）：R05 的 gold 在**向量路
  全库 #10**、已在 k=20 候选池内，是 RRF 融合后的排序把它挤出去；R09 两家 gold 都在
  池内（BM25 #16 / #12），融合后只保住万东一家。**调大深度救不回来**。
- **R10 ＝ 候选池深度不够**：5 张表只有 2 张进池，另 3 张池外（688289-0008、
  688085-0008 的 BM25 全库 0 命中）。5 家 × 同一指标要同时挤进 20 条候选，是相似度
  检索的结构性短板 —— 也正是 §12.4 那条增强存在的理由，但它只对**不点名公司**的
  集合题生效，R10 点名了 5 家，走不到。

复现：

```bash
python scripts/build_questions_xlsx.py               # 含检索自检
python scripts/build_questions_xlsx.py --no-retrieval # 只核对答案、不载模型
# R01–R10 也能直接喂现有评测脚本（字段与 qa.jsonl 一致）：
python scripts/eval_retrieval.py --dataset evaluation/datasets/questions.jsonl --routes bm25,vector
```

> **名字口径**：题面一律用 `data/manifest.csv` 的**证券简称**。排题时踩到两处 ——
> 688351 的简称是「微电生理」（别名表里没有「微创电生理」，用了会**认不出公司**，
> 而公司识别是软加权 `company_one` 的触发条件）；688050 的简称是「爱博医疗」
> （「爱博诺德」是全称的一部分）。

#### 13.7.2 生成层：答对 7/10（`results.xlsx` / `errors.xlsx`）

```bash
# 1) 跑完整链（表头索引，10 题约 1 分钟，含首载 0.6B 向量模型）
python scripts/eval_generation.py --dataset evaluation/datasets/questions.jsonl \
    --index ./index/header --out evaluation/results/generation_questions_R01-R10.json
# 2) 人判后出两份课程要求的表
python scripts/build_results_xlsx.py
```

产物：

| 文件 | 内容 |
|---|---|
| `evaluation/results.xlsx` | **逐题 11 栏**：question_id / question / BM25召回结果 / 向量召回结果 / 最终使用chunk / 引用页码 / 系统回答 / 标准答案 / 是否答对 / 错误类型 / 错误原因。附加两页：`逐题证据明细`（每题每条证据的页码/章节/召回路径/是否 gold，共 60 条）、`口径与汇总` |
| `evaluation/errors.xlsx` | 错题台账（3 行，每行多出「缺哪块 gold」「gold 覆盖几分之几」）+ E1–E10 各类型数量（**含 0 的类型也列出**，便于看全貌） |
| `evaluation/results/review_questions_R01-R10.json` | 逐题判定（人判，本表的输入） |
| `evaluation/results/generation_questions_R01-R10.json` | 原始记录（含全部证据原文与回答原文） |
| `evaluation/R01-R10_评测结论.pdf` | **结论 PDF（6 页）**：哪类题答得好 / 哪类题翻车 / 为什么。含逐题 gold 名次表、两种失败机理、改进方向 |

结论 PDF 由 `python scripts/build_conclusion_pdf.py` 生成（reportlab + 系统自带微软雅黑，
写临时文件 → 用 pypdf 抽文本层回读断言 → 覆盖）。它不复述流程，只讲结论与机理 ——
流程在本文档，原始回答在 `results.xlsx`。

**汇总**（表头索引 `index/header`，混合召回 Top-6 证据，`deepseek-chat` temp 0.1）：

| 指标 | 值 |
|---|---|
| 总准确率 | **7/10 = 70.0%** |
| 单公司题（R01–R08） | **7/8 = 87.5%** |
| 跨公司题（R09–R10） | **0/2 = 0.0%** |

| 错误类型 | 数量 | 题 |
|---|---|---|
| E6 跨公司召回不足 | 2 | R09、R10 |
| E10 其他 | 1 | R05 |
| E1/E2/E3/E4/E5/E7/E8/E9 | **各 0** | — |

三件事值得单独记：

1. **3 道错题全部是召回缺口，不是模型答错。** R05 的净额与上年同期数**完全正确**（模型
   从另一块取到了同一个数），缺的只是公司披露的同比 +373.30%；R09 万东一侧全对、安图
   一侧没数据；R10 只拿到 5 家里的 2 家。三题都属「该进证据的块没进来」，E1–E5、E7–E9
   八类**一类没沾上**：没认错公司、没引错页码、没切块/表格解析缺陷、没算错数、没编造引用编号。
   R10 的拒答是**正确行为**（明确列出缺的是哪三家），但按「题目问到的都要答到」的口径记答错。
2. **0 处凭空数字。** 把 10 篇回答里的数字 token 逐个拿去证据原文里查：除题面自带的股票
   代码、以及 3 处**自算差值**（R03 的 0.27 个百分点、R05 的 126,378,789.27 与
   73,988,414.16 元，均为正确减法）外，**全部逐字出自证据**。
3. **R05 是一个判定边界，已写进 `errors.xlsx` 的原因栏**：它给出了净额 + 自算的增减额，
   只是没给同比比例。按宽松口径（「同比变化」接受绝对增减额）可判对，本表按严格口径
   （与 §13.3 同一口径）判错并注明 —— 换成宽松口径，单公司就是 8/8、总准确率 80%。

> **口径提醒**：这是一次采样的结果（temp 0.1，同一题换一轮表述会变），
> 与 §13.3 的「7/10」是**两次独立运行**、题面也不同，不能相加也不该平均。

### 13.8 收尾：只剩截图

```bash
# 页面已在跑（表头增强索引）。若已关掉，重开：
cd "D:/复旦/学习资料/Agent/上交所医疗器械财报RAG"
INDEX_DIR=./index/header .venv/Scripts/python.exe -m streamlit run app.py
# 浏览器打开 http://localhost:8501，先点一个示例预热（首次走向量路载模型约 30 秒），
# 然后截 4 张图（问题输入 / 最终回答 / 检索结果 / 引用出处）存 screenshots/
```

重新出评测表（无需改代码）：

```bash
python scripts/eval_generation.py --report --review evaluation/results/review_formal10.json
```
