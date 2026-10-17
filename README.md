# 科研文献智能助手（Research Assistant）

基于 LangGraph 的 **Agentic RAG** 应用：Agent 自主决定"直接回答 / 检索知识库 / 总结论文 / 调用计算器"，检索采用 **BM25 + 向量双路召回 + RRF 融合 + Cross-Encoder 重排** 的混合管线，回答附带文献来源追溯（文件名+页码+章节），并配套 **Recall@K / MRR / 引用正确率** 离线评测体系。

> 📖 Agent 零基础？先读 [AGENT_GUIDE.md](./AGENT_GUIDE.md) —— 从领域地图到本项目逐模块的通俗讲解。

## 功能

- 📄 **章节感知知识库**：PDF 逐页解析（字号+正则识别章节标题）→ 章节内递归切分 → Embedding → FAISS + BM25 双路索引落盘
- 🔀 **混合检索与重排**：BM25 关键词路 + FAISS 向量路双召回，RRF 融合，BGE Cross-Encoder 二次精排 Top-K
- 🤖 **Agent 自主路由**：常识直答 / 文献检索 / 论文总结 / 指标计算四路动态选择（非写死流程）
- 💬 **多轮会话**：LangGraph checkpointer 维护上下文，支持追问
- 🔍 **来源可追溯**：回答标注 [n] 编号 + 文件名 + 页码 + 章节，界面展示完整检索与计算轨迹
- 📊 **离线评测体系**：18 题金标集，Recall@K / MRR / 引用正确率，三配置对照实验 + badcase 分析

## 实测评测数据（18 题金标集，Recall@4 / MRR）

| 检索配置 | Recall@4 | MRR | 平均延迟 | term 类 Recall |
|---|---|---|---|---|
| 纯向量检索（基线） | 77.78% | 0.722 | ~260ms | 80.0% |
| BM25+向量+RRF | 83.33% | 0.796 | ~360ms | **100%** |
| **混合检索+Cross-Encoder 重排（上线）** | **88.89%** | **0.806** | ~1.7s | **100%** |
| 端到端引用正确率（完整管线，10 题） | 80.00% | — | — | — |

**读数结论**：混合检索把 Recall@4 从 77.78% 提升到 88.89%（+11.1pp）；其中 BM25 贡献了术语类问题 80%→100% 的查全（专业术语字面命中是它的强项），Cross-Encoder 重排进一步修正排序（MRR 0.722→0.806）并找回 1 道指标题。badcase 剩余 2 道（学习率、全量 AUC），归因为短数字上下文在切分与 Embedding 中的表征弱，可考虑表格抽取与数字扩展增强。

> 复现实验：`.venv\Scripts\python.exe scripts\run_evaluation.py`（加 `--e2e` 跑引用正确率）
> 历史迭代记录在 `data/eval/results_*.json`，调参后重跑即可累积对照证据。

## 技术栈

| 层 | 选型 |
|---|------|
| 编排框架 | LangGraph（`create_agent`，ReAct 循环 + MemorySaver） |
| 对话模型 | 智谱 GLM-4-Flash（OpenAI 兼容模式，可插拔） |
| Embedding | 智谱 embedding-3（2048 维） |
| 稀疏检索 | BM25（rank_bm25 + jieba 中文分词） |
| 向量检索 | FAISS（faiss-cpu，本地持久化） |
| 重排 | BAAI/bge-reranker-base（Cross-Encoder，CPU 可跑） |
| PDF 解析 | PyMuPDF（逐页 + 字号结构，章节感知切分） |
| 界面 | Streamlit |

## 快速开始

```powershell
# 1. 配置密钥：编辑 .env，填入智谱 API Key
ZHIPU_API_KEY="你的key"

# 2. 安装依赖
uv sync

# 3. （可选）生成 3 篇测试用模拟文献
.venv\Scripts\python.exe scripts\make_sample_pdfs.py

# 4. 构建双路索引（章节感知切分 + FAISS + BM25）
.venv\Scripts\python.exe scripts\rebuild_index.py

# 5. 启动
.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 http://localhost:8501 → 左侧上传 PDF → 点"重建知识库" → 开始提问。

> 首次提问会加载重排模型（约 1.1GB，自动从 hf-mirror 下载，之后常驻内存）。

## 项目结构

```
research-assistant/
├── src/
│   ├── config.py           # 配置中心：模型/路径/切分/召回/重排参数
│   ├── document_loader.py  # 章节感知解析与递归切分（source/page/section/paper_title 元数据）
│   ├── vector_store.py     # FAISS 向量索引构建与检索
│   ├── retriever.py        # 混合检索管线：BM25+向量双召回→RRF融合→Cross-Encoder重排
│   ├── tools.py            # Agent 工具：search_literature / summarize_paper / calculate
│   ├── agent.py            # Agent 组装：模型+工具+记忆+路由规则
│   └── evaluation.py       # 评测：Recall@K / MRR / 引用正确率
├── app.py                  # Streamlit 界面
├── scripts/
│   ├── make_sample_pdfs.py   # 生成测试文献（含章节结构、跨页样例）
│   ├── rebuild_index.py      # 一键重建双路索引
│   └── run_evaluation.py     # 三配置对照实验 + badcase 清单
└── data/
    ├── pdfs/                # 文献目录
    ├── vector_index/        # FAISS 索引 + BM25 语料
    └── eval/                # 金标评测集 + 历次实验结果
```

## 核心机制速览

**混合检索管线**（区别于 naive 单路向量检索）：
```
用户 query
  ├─ 路 A Dense：query 向量化 → FAISS Top-8
  ├─ 路 B Sparse：jieba 分词 → BM25 Top-8
  ├─ RRF 融合：score(d) = Σ 1/(60 + rank_i(d))，排名级融合免调参
  └─ Cross-Encoder 重排：BGE 逐对精读 (query, 片段) → 最终 Top-4
```

**Agentic 路由**（四个工具由模型按问题类型自主选择）：
```
用户提问 → Agent 判断问题类型
   ├─ 通用常识 → 直接回答
   ├─ 涉及文献内容 → search_literature（混合检索，带来源）
   ├─ 总结某篇论文 → summarize_paper（按章节组织全文并 LLM 总结）
   └─ 精确计算 → calculate（numexpr 安全求值，仅计算检索获得的数字）
→ 综合工具结果生成回答（强制标注来源编号+文件名）
```

**多轮记忆**：`thread_id` + `MemorySaver` checkpointer，同一会话自动携带历史。

## 已知局限（诚实声明，面试可讲）

- 剩余 2 道 badcase（学习率/全量 AUC）：短数字型指标在 Embedding 空间表征弱，改进方向是表格结构化抽取 + 查询扩展
- Cross-Encoder 重排在 CPU 上增加约 1.4s 延迟（GPU 可降至 ~100ms）；配置项可一键降级
- MemorySaver 存内存，重启即失忆；生产应换 SqliteSaver（接口兼容）
- PDF 解析基于文本层字号结构，扫描件（图片型 PDF）需先 OCR
- GLM-4-Flash 复杂路由偶有保守，已通过命令式 system prompt 与工具顺序规则缓解；预算允许可换 glm-4-plus

## 学习建议

1. 通读 `AGENT_GUIDE.md` 建立概念地图（含混合检索与评测的通俗解释）
2. 跑通界面，用"该观察什么"清单玩一遍
3. 跑一次 `run_evaluation.py` 看三配置差异，改 TOP_K/CHUNK_SIZE 后重跑对比
4. 每改一处记录结论——这些对照数据就是面试素材

## 致谢

项目结构参考 [langchain-ai/langgraph-101](https://github.com/langchain-ai/langgraph-101) 官方教程的设计模式（create_agent + tools + checkpointer）。

> AI生成
