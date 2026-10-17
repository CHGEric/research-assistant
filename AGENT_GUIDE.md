# Agent 小白认知指南 —— 五分钟建立这个领域的"地图"

> 这份指南写给第一次接触 Agent 的你。先建立宏观认识，再看本项目怎么落地。
> 阅读顺序：先通读本文 → 再跑一遍项目 → 再回头逐文件精读代码注释。

---

## 1. Agent 到底是什么？

一句话：**Agent = 会自己动手的 LLM**。

| 类型 | 比喻 | 工作方式 |
|------|------|----------|
| 普通 Chatbot | 一个只会背书的客服 | 你问一句，它凭"记忆"答一句，答不了就瞎编 |
| + 工具的 LLM | 给它接上电话、计算器、图书馆 | 遇到不会的，可以先查再答 |
| **Agent** | **一个能自己安排干活的实习生** | 给它一个目标，它自己决定：先查什么、再算什么、什么时候能交差 |

判断标准：**"决定做什么"这件事由谁来做。**
- 普通程序：开发者用 if-else 写死每一步流程
- Agent：**模型自己根据情况决定下一步**（这就是"自主性"）

## 2. Agent 领域全景图（认门牌）

技术演进是一条清晰的升级路线，每一层都站在上一层之上：

```
LLM（大语言模型）              ← 会说话的"大脑"
  └─ Chat / Prompt 工程        ← 教大脑好好说话
      └─ RAG（检索增强生成）    ← 给大脑配一个"开卷资料库"（本项目模块②）
          └─ Tool Calling      ← 给大脑接上"手"（本项目模块①）
              └─ Agent         ← 大脑+手+决策循环（本项目模块③）
                  └─ Multi-Agent   ← 多个 Agent 分工协作（LangGraph 101 的 201 部分）
                      └─ DeepAgent ← 带规划、文件系统、技能库的"超级实习生"
```

**本项目站在第 4 层（Agent）**，同时用到了 RAG 和 Tool Calling 作为零件。这就是"Agentic RAG"这个词的由来：用 Agent 的决策能力来驱动 RAG 检索，而不是无脑每次都检索。

## 3. 五个核心概念（用大白话）

### 3.1 ReAct 循环 —— Agent 的工作节奏

Agent 干活遵循一个循环：**思考 → 行动 → 观察 → 再思考**。

举例，你问"三篇文献的平均准确率是多少"：
```
思考1：我需要先知道各文献的准确率 → 行动1：调用 search_literature("准确率")
观察1：拿到检索结果（91.2%、94.8%……）
思考2：需要精确计算平均值 → 行动2：调用 calculate("(91.2+94.8)/2")
观察2：结果 93.0
思考3：信息够了，可以回答了 → 输出最终答案
```
这个循环不是我们写的，是 LangGraph 的 `create_agent` 内部用**状态图**实现的——所以这个框架叫 Lang**Graph**。

### 3.2 Tool Calling —— 大脑指挥手的协议

关键分工：
- **模型只"下单"**：输出一段结构化指令（"我要用 search_literature，参数是 xxx"）
- **框架来"跑腿"**：LangGraph 拦截指令 → 真正执行 Python 函数 → 把结果塞回给模型

工具怎么注册？`@tool` 装饰器把函数的**名字、docstring、参数类型**打包成"说明书"给模型看。所以 **docstring 写得越清楚，模型选工具越准**——这不是注释，是给模型看的说明书！

### 3.3 RAG —— 给模型"开卷考试"

LLM 的两个先天缺陷：训练截止后的知识它不知道；你私人的资料它更不知道。
RAG（Retrieval-Augmented Generation）的解法很朴素——**先查资料，再回答**：

```
你的 PDF ──解析──> 知识块 ──Embedding──> 向量 ──存入──> FAISS 索引      （建库，一次性）
用户问题 ──Embedding──> 向量 ──最近邻搜索──> Top-K 个相关块 ──拼进提示词──> LLM 回答（每次提问）
```

### 3.4 Embedding —— 给文字标"语义坐标"

Embedding 模型把一段文本变成一个 2048 维向量（本项目用智谱 embedding-3）。
**语义相近的文本，向量距离就近**。效果：
- 搜"模型效果不好"，能召回写着"性能下降"的段落（关键词匹配做不到）
- "小明今年 8 岁"和"小明是小学生"距离很近，虽然一个字都不重复

### 3.5 FAISS —— 毫秒级的"最近邻搜索"

建好馆之后，每次提问都要在几千个向量里找"离问题最近的 K 个"。暴力遍历太慢，FAISS 用近似算法把这事做到毫秒级。本项目用它的 CPU 版（`faiss-cpu`），学习够用。

### 3.6 状态管理 —— Agent 的"工作记忆"

LLM 本身是金鱼记忆：每次调用只看到本轮消息。多轮对话的实现靠 **checkpointer**：
- 每轮结束，把完整消息历史存档（`MemorySaver` 存内存，生产用 SQLite/Postgres）
- **thread_id**（对话档案编号）是取回记忆的钥匙
- 同一个 thread_id 再来，历史自动带上 → 于是 Agent"记得"你上一句说了什么

本项目 `agent.py` 里 `ask(agent, question, thread_id)` 就是这个机制。

### 3.7 混合检索与重排 —— v2 升级的核心（对应简历亮点） 这里需要编一个故事

**为什么要两路检索？** 向量检索是"语义控"，BM25 是"字面控"，各有盲区：

| 问法 | 向量检索 | BM25 | 谁赢 |
|------|---------|------|------|
| "如何缓解冷启动问题"（自然语言） | ✅ 语义相近就能召回 | ❌ 字面几乎不重叠 | 向量 |
| "负采样比例 1:5"（术语/数字） | ⚠️ 数字在向量空间里长得都差不多 | ✅ 字面命中直接召回 | BM25 |

**RRF 融合**：两路的"分数量纲"完全不同没法直接加，RRF 的聪明之处是只用**排名**：
`score(d) = Σ 1/(60 + rank_i(d))`。两路都召回的块自动累加两次分数 → "双路都认为相关"的块浮到最前面。免调参，工程上极其稳健。

**Cross-Encoder 重排**：向量检索是"两个人各自写自我介绍再比对"（Bi-Encoder，快但粗）；Cross-Encoder 是"两个人坐下来逐句交流"（query 和文档拼在一起过模型，慢但准）。所以工程套路永远是：**先粗召回（快），后精重排（准）**。本项目用 BGE-reranker（中文优化版），CPU 上多花约 1.4 秒。

### 3.8 评测 —— 怎么证明"变好了"而不是"感觉变好了"

三个指标一句话解释：
- **Recall@K**：N 个问题里，正确答案出现在前 K 个检索结果里的比例（查全了吗）
- **MRR**：正确答案排第 1 得 1 分、第 2 得 0.5 分……未出现得 0（排序好吗）
- **引用正确率**：端到端问 Agent，回答能不能追溯到正确的文献（最终交付质量）

本项目实测（18 题金标集）：纯向量 77.78% → 加 BM25/RRF 83.33% → 加重排 88.89%，
每个组件的贡献都能拆开解释——这就是面试时"能讲清楚为什么这么设计"的底气。

## 4. 本项目的地图（文件 ↔ 概念对照）

```
research-assistant/
├── src/
│   ├── config.py           # 配置中心：模型/路径/切分/召回/重排参数（改行为先看这）
│   ├── document_loader.py  # 【RAG 第①步】PDF→章节感知→知识块（source/page/section 元数据）
│   ├── vector_store.py     # 【RAG 第②步】知识块→向量→FAISS 建库/检索（Dense 路）
│   ├── retriever.py        # 【RAG 第③步】BM25 稀疏路 + RRF 融合 + Cross-Encoder 重排
│   ├── tools.py            # 【双手】search_literature / summarize_paper / calculate 三个工具
│   ├── agent.py            # 【总装】create_agent 组装大脑+双手+记忆+规则
│   └── evaluation.py       # 【质检】Recall@K / MRR / 引用正确率
├── app.py                  # 【门面】Streamlit 界面：上传/问答/来源追溯/管线状态
├── scripts/
│   ├── make_sample_pdfs.py   # 生成测试文献（含章节结构与跨页样例）
│   ├── rebuild_index.py      # 一键重建双路索引
│   └── run_evaluation.py     # 三配置对照实验 + badcase 清单
└── data/
    ├── pdfs/               # 你的文献放这里
    ├── vector_index/       # FAISS 索引 + BM25 语料（重启不用重建）
    └── eval/               # 金标评测集 + 历次实验结果
```

一次提问的完整数据流：

```
你在界面输入问题
  → app.py 用 thread_id 调用 agent.py 的 ask()
  → Agent 思考：常识？文献细节？求总结？求计算？
  → 文献细节 → tools.py 的 search_literature
      → retriever.py：FAISS Top-8 + BM25 Top-8 → RRF 融合 → BGE 重排 → Top-4
        （每个片段带 文件名+页码+章节 元数据）
  → 求总结 → summarize_paper（按章节组织全文 → LLM 结构化总结）
  → 求计算 → calculate（numexpr 安全求值，只算检索拿到的数字）
  → 模型综合工具结果，生成带 [1][2] 来源编号的回答
  → app.py 渲染回答 + “检索与计算记录”折叠面板（可追溯）
```
research-assistant/
├── src/
│   ├── config.py           # 配置中心：模型/路径/切分参数（改行为先看这）
│   ├── document_loader.py  # 【RAG 第①步】PDF→文本→知识块（保留页码元数据）
│   ├── vector_store.py     # 【RAG 第②步】知识块→向量→FAISS 建库/检索（Dense 路）
│   ├── retriever.py        # 【RAG 第③步】BM25 稀疏路 + RRF 融合 + Cross-Encoder 重排
│   ├── tools.py            # 【双手】search_literature / summarize_paper / calculate 三个工具
│   ├── agent.py            # 【总装】create_agent 组装大脑+双手+记忆+规则
│   └── evaluation.py       # 【质检】Recall@K / MRR / 引用正确率
├── app.py                  # 【门面】Streamlit 界面：上传/问答/来源追溯/管线状态
├── scripts/
│   ├── make_sample_pdfs.py   # 生成测试文献（含章节结构与跨页样例）
│   ├── rebuild_index.py      # 一键重建双路索引
│   └── run_evaluation.py     # 三配置对照实验 + badcase 清单
└── data/
    ├── pdfs/               # 你的文献放这里
    ├── vector_index/       # FAISS 索引 + BM25 语料（重启不用重建）
    └── eval/               # 金标评测集 + 历次实验结果
```

一次提问的完整数据流：

```
你在界面输入问题
  → app.py 用 thread_id 调用 agent.py 的 ask()
  → Agent 思考：这是常识还是文献问题？
  → 需要检索 → tools.py 的 search_literature
      → retriever.py：FAISS Top-8 + BM25 Top-8 → RRF 融合 → BGE 重排 → Top-4
        （每个片段带 文件名+页码+章节 元数据）
  → 求总结 → summarize_paper（按章节组织全文 → LLM 结构化总结）
  → 需要计算 → tools.py 的 calculate（numexpr 安全求值，只算检索拿到的数字）
  → 模型综合工具结果，生成带 [1][2] 来源编号的回答
  → app.py 渲染回答 + "检索与计算记录"折叠面板（可追溯）
```

## 5. 怎么跑 & 每一步该观察什么

```powershell
cd F:\TeleAgent_workspace\research-assistant
uv sync                                       # 装依赖（首次）
.venv\Scripts\python.exe scripts\make_sample_pdfs.py   # 生成测试文献（已有可跳过）
.venv\Scripts\python.exe scripts\rebuild_index.py     # 构建双路索引（章节感知+FAISS+BM25）
.venv\Scripts\python.exe -m streamlit run app.py       # 启动界面
```

浏览器打开 http://localhost:8501 后，按这个顺序**边玩边对照概念**：

| 操作 | 你在验证什么 |
|------|--------------|
| 问"什么是过拟合" | **不该**触发检索（常识问题直接回答） |
| 问"GraphSAGE 的负采样比例是多少" | 触发混合检索：BM25 字面命中"负采样"，看轨迹面板里的片段带章节信息 |
| 问"这几篇文献用了什么模型" | **应该**触发 search_literature，展开轨迹面板看检索片段 |
| 问"总结一下 resnet 那篇" | **应该**触发 summarize_paper（模糊匹配文件名），总结按章节组织 |
| 问"准确率平均值是多少" | **应该**触发 calculate，看模型把算式传给计算器（先检索拿数字再算） |
| 接着问"第二篇的 F1 呢" | 多轮记忆：它知道"第二篇"指哪篇 |
| 跑 `scripts\run_evaluation.py` | 三配置对照实验：看混合检索+重排比纯向量好多少 |
| 问一个知识库里没有的文献问题 | 检索空手而归 → 它应该说"未找到"而不是编造 |

## 6. 推荐的动手改造练习（由浅入深）

学 Agent 最好的方式不是读，是**改**。每次只改一个变量，跑 `run_evaluation.py` 对比指标变化：

1. **调 TOP_K**（config.py）：4 改成 1，看 Recall 怎么掉 → 理解"检索是质量天花板"
2. **调 CHUNK_SIZE**：800 改成 100，看检索结果多破碎 → 理解切分的意义
3. **关掉重排**（RERANK_ENABLED=False）：看 Recall@4 与延迟如何变化 → 理解重排的价值
4. **加一个新工具**（tools.py）：仿照 calculate 写一个 `get_today`，看模型什么时候用它
5. **改坏 docstring**：把 search_literature 的说明改成模糊的"搜索"，看路由准确率下降 → 证明说明书的价值
6. **换模型**（config.py）：`glm-4-flash` → `glm-4-plus`，对比回答质量与端到端引用正确率
7. **进阶**：把 MemorySaver 换成 SqliteSaver（langgraph 自带），实现重启后记忆不丢

## 7. 面试可能深挖的问题（提前备好）

- **为什么用 Agentic RAG 而不是每次都检索？** → 常识问题检索浪费 token 和延迟；Agent 判断要不要查，兼顾成本与质量
- **为什么要 BM25 + 向量双路？** → 向量擅长语义、BM25 擅长术语字面命中；实测术语类问题 Recall 80%→100%，总 Recall +5.6pp
- **RRF 为什么不用加权分数融合？** → 两路分数量纲不同需归一调参；RRF 只用排名，免调参且稳健（60 是社区惯例平滑常数）
- **Cross-Encoder 和向量检索的本质区别？** → Bi-Encoder 各自编码再比距离（快而粗）；Cross-Encoder 拼接过模型逐 token 交互（慢而准）；所以先粗召回后精重排
- **chunk 大小怎么权衡？** → 太大：检索粒度粗、噪声多；太小：上下文破碎。800 字符+100 重叠是中文论文的经验起点，最终靠实验调
- **评测集怎么构建、指标怎么算？** → 基于文献人工标注 18 题（问题+金标出处+关键词）；Recall@K 看查全、MRR 看排序、引用正确率看端到端可追溯；badcase 剩 2 道短数字题，归因为数字表征弱，改进方向是表格抽取+查询扩展
- **为什么温度设 0.1？** → 科研问答要可复现，创造性是副作用
- **numexpr 而不是 eval？** → 安全性：eval 能执行任意代码，numexpr 只算数学
- **MemorySaver 的局限？** → 存内存、重启即丢；生产换 SqliteSaver/PostgresSaver，接口不变
- **检索不到时模型为什么会编造？怎么防？** → LLM 的训练倾向是"给出答案"；在 system prompt 里强制"检索不到就明说"，并给它说"不知道"的合法出口
- **小模型路由不稳怎么治？** → 实测 GLM-4-Flash 曾不检索、直接拿工具说明里的示例数字瞎算；解法：命令式 system prompt（"检索先行"）+ 工具说明去掉具体数字示例 + 明确工具顺序规则——这本身就是一段很好的面试叙事

## 8. 这个领域继续往哪走？

- **LangGraph 101 的 201 部分**：多 Agent 协作（supervisor + 子 Agent）、Human-in-the-Loop、DeepAgents——你手上这个项目是通往那边的地基
- **MCP（Model Context Protocol）**：工具接入的"USB 接口"标准，正在成为行业事实标准
- **评估（Evals）**：怎么科学地度量"我的 Agent 变好了"——面试高阶加分项

> 记住这张地图的用途：以后遇到任何新名词（工作流、记忆、护栏、子Agent……），
> 先问它属于哪一层、解决什么问题，就不会迷路。

> AI生成