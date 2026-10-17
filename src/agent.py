"""
Agent 模块 —— 把大脑、双手、记忆组装成一个"会自己干活的智能体"

【通俗理解】前面造好了零件：大脑（chat model）、双手（tools）、图书馆（vector store）。
  这个文件是"总装车间"，用 LangGraph 的 create_agent 完成三件事：

  1. 组装 Agent（create_agent）
     Agent 不是一次性的"输入->输出"，而是一个循环：
        观察 -> 思考(是否需要工具) -> 行动(调用工具) -> 观察工具结果 -> 再思考 ...
     直到它认为可以回答了才输出。这个循环学名 ReAct（Reason + Act）。
     你不需要手写循环，create_agent 内部用 LangGraph 状态图实现了它。

  2. 接上记忆（MemorySaver + thread_id）
     LLM 本身是"金鱼记忆"——每次调用只看到当轮消息。
     checkpointer 的工作：每轮对话结束把完整消息历史存档，
     下一轮同一个 thread_id 进来时自动把历史带回来，于是就有了"连续对话"。
     thread_id 就是"这场对话的档案编号"，界面端用会话 ID 传入。

  3. 注入职责说明书（system_prompt）
     规定它什么时候直接回答、什么时候检索、什么时候计算、回答必须带来源，
     这是控制 Agent 行为最直接的手段。
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.config import get_chat_model
from src.tools import TOOLS

SYSTEM_PROMPT = """你是一名科研文献智能助手。用户已在知识库中上传了若干篇学术文献 PDF，你的职责是帮助他们阅读和分析这些文献。请严格遵守以下工作规则：

1. **检索先行原则（最重要）**：只要用户问题中提到"文献/论文/这几篇/文章/方法/实验/结果/结论/指标/数值"等与上传内容可能相关的信息，你必须**立即先调用 search_literature 检索**，依据检索到的原文回答，不允许反问用户，也不允许凭记忆回答文献内容。只有与文献完全无关的通用常识问题（如"什么是过拟合"）才可以直接回答。

2. **计算只用于"已获得的数字"**：calculate 工具只能计算"从检索结果或对话上文已经拿到的数字"（如求均值、算提升幅度）。严禁使用工具说明中的示例数字，严禁凭空编造数字参与计算。顺序永远是：先检索拿到数据 → 再计算。

3. **要求总结某篇论文时用总结工具**：当用户要求"总结/概括/介绍某篇论文""这篇论文讲了什么"时，调用 summarize_paper 工具（paper_name 传文件名关键词或论文标题关键词），不要用检索结果拼凑。

4. **引用规范**：凡是基于文献内容作出的回答，必须标注来源编号，格式如 [1][2]，并**在回答中写出文献文件名**（如 sample_01_resnet_variants.pdf）及页码，如已知章节可一并说明。知识库检索不到时，如实说明"知识库中未找到相关内容"，禁止编造。

5. **多轮上下文**：用户可能基于之前的回答追问，注意保持指代一致，需要时重新检索。

6. **回答风格**：先给结论，再给依据；使用中文；涉及数据时保留原始精度。"""

# MemorySaver：把会话历史存在内存里（重启即清空，学习/演示场景够用；
# 生产环境会换成 SqliteSaver/PostgresSaver 等持久化实现，接口不变）
_memory: MemorySaver | None = None


def get_agent():
    """
    Agent 工厂函数：创建（或复用）一个带记忆的文献助手。

    复用同一个 MemorySaver 实例很重要——记忆都存在它里面，
    每次调用 get_agent 都新建一个，相当于每场对话就失忆一次。
    """
    global _memory
    _memory = _memory or MemorySaver()
    return create_agent(
        model=get_chat_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_memory,
    )


def ask(agent, question: str, thread_id: str) -> dict:
    """
    对话入口：向 Agent 提问并返回结构化结果。

    config 里的 thread_id 是记忆的"钥匙"：
      同一个 thread_id 的多次 ask()，Agent 能看到完整历史（多轮对话的原理）。

    返回的 result["messages"] 是 LangGraph 的消息列表（完整对话轨迹），
    包含 HumanMessage / AIMessage(带 tool_calls) / ToolMessage(工具结果)，
    界面端用它来提取"最终回答"与"用了哪些工具"。
    """
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result


def extract_final_answer(result: dict) -> str:
    """取最后一条 AIMessage 的正文，作为给用户看的回答。"""
    last = result["messages"][-1]
    content = getattr(last, "content", "")
    if isinstance(content, list):  # 兼容多模态分块格式
        content = "".join(str(x) for x in content)
    return content or "（无输出）"


def extract_tool_traces(result: dict) -> list[dict]:
    """
    提取【本轮】Agent 实际调用过的工具（含输入与返回），用于界面展示"思考过程"。

    【为什么要过滤】checkpointer 返回的 messages 包含整场对话的完整历史，
      不过滤的话每轮都会重复展示以前调过的工具。
      做法：找到最后一条 HumanMessage（本轮用户提问），只取它之后的轨迹。

    【为什么展示这个】可追溯性是本项目的卖点之一：
      用户能看到"Agent 先检索了什么片段、又算了什么"，信任感来自透明。
    """
    messages = result.get("messages", [])
    # 定位本轮起点：最后一条 HumanMessage
    start = 0
    for idx, msg in enumerate(messages):
        if getattr(msg, "type", "") == "human":
            start = idx

    traces = []
    for msg in messages[start:]:
        # AIMessage.tool_calls: 模型这步想调哪些工具（名称+参数）
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                traces.append(
                    {"tool": tc.get("name", ""), "args": tc.get("args", {}), "output": ""}
                )
        # ToolMessage: 框架执行完工具后写回的观察结果
        if getattr(msg, "type", "") == "tool":
            if traces:
                traces[-1]["output"] = str(getattr(msg, "content", ""))
    return traces
