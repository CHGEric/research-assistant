"""
工具模块 —— Agent 的"双手"（v2：三个工具 + 混合检索管线）

【通俗理解】大脑（LLM）再聪明也只是语言模型，它不能读硬盘、不能查数据库、
  也不能做精确算术。Tools 就是给大脑接上的"外挂器官"：

      search_literature -> 一双能查图书馆的手（v2 起走 BM25+向量+RRF+重排混合管线）
      summarize_paper   -> 一双能通读整篇论文并写读书笔记的手
      calculate         -> 一只按计算器的手

关键机制（Tool Calling 的原理）：
  1. 每个工具用 @tool 装饰器注册，LangChain 会把【函数名 + docstring + 参数
     类型】翻译成一份"说明书"交给 LLM；
  2. LLM 读到用户问题后，自行判断：这个问题的答案是"我知道的常识"还是
     "必须查资料"？要查的话该用哪个工具、传什么参数？
  3. 它不执行工具，只输出一段结构化指令（工具名+参数）；
  4. LangGraph 框架截获指令，真正执行函数，把结果喂回给 LLM；
  5. LLM 拿到结果，组织成自然语言回答。

  ——这套"模型决策 + 框架执行"的分工，就是 Agentic RAG 的核心：
    是否检索、检索什么，由模型根据问题动态决定，而不是写死的 if-else。
"""

import numexpr
from langchain_core.tools import tool

from src.config import TOP_K, get_chat_model

# ---------------------------------------------------------------------------
# 工具 1：文献检索（混合管线：BM25 + 向量 双路召回 -> RRF 融合 -> 重排）
# ---------------------------------------------------------------------------


@tool
def search_literature(query: str) -> str:
    """在科研文献知识库中检索与问题相关的原文片段。当问题涉及用户上传的论文内容、实验细节、方法、指标或结论时使用。返回内容包含原文片段及其来源（文件名、页码、所属章节）。"""
    from src.retriever import hybrid_search

    docs = hybrid_search(query, top_k=TOP_K)
    if not docs:
        return "知识库为空或未构建。请提示用户先在左侧上传 PDF 文献并构建知识库。"

    # 把检索结果拼成"带来源编号"的文本块，供 LLM 引用
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知文件")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        snippet = doc.page_content.strip()
        parts.append(f"[{i}] 来源：{source} 第{page}页（{section}）\n{snippet}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 工具 2：论文摘要与章节总结
# ---------------------------------------------------------------------------


@tool
def summarize_paper(paper_name: str) -> str:
    """对指定论文生成摘要与章节要点总结。paper_name 传论文文件名（如 sample_01_resnet_variants.pdf）、文件名的关键词（如 resnet、vit）或论文标题关键词。当用户要求"总结/概括/介绍某篇论文"或"这篇论文讲了什么"时使用。"""
    from src.retriever import load_bm25

    loaded = load_bm25()
    if loaded is None:
        return "知识库为空或未构建。请提示用户先上传 PDF 文献并构建知识库。"
    _, docs = loaded

    # 三级模糊匹配：文件名包含 -> 论文标题包含 -> 大小写不敏感文件名
    matched = [d for d in docs if paper_name in d.metadata.get("source", "")]
    if not matched:
        matched = [d for d in docs if paper_name in d.metadata.get("paper_title", "")]
    if not matched:
        matched = [
            d for d in docs if paper_name.lower() in d.metadata.get("source", "").lower()
        ]
    if not matched:
        sources = sorted({d.metadata.get("source", "") for d in docs})
        return f"未找到匹配 '{paper_name}' 的论文。知识库现有文献：{', '.join(sources)}"

    # 按页码排序后按章节分组（保持阅读顺序），拼出带章节结构的全文
    matched.sort(key=lambda d: (d.metadata.get("page", 0), d.metadata.get("section", "")))
    paper_title = matched[0].metadata.get("paper_title") or matched[0].metadata["source"]
    source_file = matched[0].metadata["source"]

    sections: dict[str, list[str]] = {}  # Python 3.7+ dict 保序 -> 章节保持出现顺序
    for d in matched:
        sections.setdefault(d.metadata.get("section", "正文"), []).append(d.page_content)

    organized = "\n\n".join(
        f"【{sec}】\n{''.join(texts)}" for sec, texts in sections.items()
    )

    # 交给 LLM 做摘要与要点提取（工具内部的一次"子任务"，对用户透明）
    model = get_chat_model()
    prompt = (
        "请基于以下论文内容生成结构化总结，要求：\n"
        "1. 一句话概括论文核心工作；\n"
        "2. 按章节列出要点（保留关键指标数字与术语）；\n"
        "3. 末尾注明论文标题与文件名。\n\n"
        f"论文标题：{paper_title}\n文件名：{source_file}\n\n论文内容：\n{organized}"
    )
    result = model.invoke(prompt)
    content = getattr(result, "content", "")
    if isinstance(content, list):
        content = "".join(str(x) for x in content)
    return f"论文《{paper_title}》（{source_file}）的总结：\n\n{content}"


# ---------------------------------------------------------------------------
# 工具 3：指标计算
# ---------------------------------------------------------------------------


@tool
def calculate(expression: str) -> str:
    """执行数学计算，支持四则运算、平均数、平方等。用于汇总从文献检索中获得的指标（如求多个实验结果的平均值、提升幅度）。expression 是数学表达式字符串，其中的数字必须来自检索结果或对话上文，禁止使用本说明中的任何示例数字。"""
    try:
        # numexpr 是安全表达式求值库，只支持数学运算，不会执行任意 Python 代码
        result = numexpr.evaluate(expression).item()
        return f"计算结果：{expression} = {result}"
    except Exception as e:  # noqa: BLE001  工具层要把错误文本化回传给 LLM
        return f"计算失败：{e}。请检查表达式格式，只支持数字与数学运算符。"


# 工具清单：交给 Agent 的"能力目录"
TOOLS = [search_literature, summarize_paper, calculate]
