"""
Streamlit 交互界面 —— 项目的"门面"

【通俗理解】Streamlit 是 Python 生态里最快搭网页界面的方案：
  写普通 Python 脚本，浏览器自动渲染成 Web 应用，不用学 HTML/JS/前端框架。

本项目界面四大功能：
  1. 左侧栏：上传 PDF -> 保存到 data/pdfs -> 一键构建知识库（解析+切分+向量化）
  2. 主区域：多轮问答聊天（历史、追问）
  3. 回答附带"知识库检索记录"折叠面板 -> 可追溯
  4. 知识库状态展示（文献数 / 知识块数）

【Streamlit 最重要的心智模型：脚本级重跑】
  用户每做一次交互（发消息/点按钮），整个脚本就从头到尾重新执行一遍。
  这就是为什么所有"需要跨轮保留"的东西（agent 实例、聊天记录）都要放进
  st.session_state——它是 Streamlit 提供的"重跑中幸存下来的保险箱"。
  thread_id 正是利用这一点实现多轮记忆与界面的对齐。
"""

import shutil
import uuid
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ---------- 页面配置（必须是本脚本第一个 st 调用）----------
st.set_page_config(page_title="科研文献智能助手", page_icon="📚", layout="wide")

# ---------- 初始化：只做一次的准备工作 ----------
from src.config import (  # noqa: E402
    PDF_DIR,
    RERANK_ENABLED,
    RERANK_MODEL,
    TOP_K,
    get_embeddings,
)
from src.document_loader import load_pdf  # noqa: E402
from src.retriever import build_bm25_index, load_bm25  # noqa: E402
from src.vector_store import build_and_save_index, load_index  # noqa: E402
from src.agent import get_agent, ask, extract_final_answer, extract_tool_traces  # noqa: E402


@st.cache_resource(show_spinner=False)
def _cached_agent():
    """cache_resource：整个应用只创建一次 Agent（重跑脚本时复用，记忆不丢失）。"""
    return get_agent()


@st.cache_resource(show_spinner=False)
def _cached_reranker():
    """重排模型 1.1GB，跨会话只加载一次；首次运行会自动从镜像站下载。"""
    if not RERANK_ENABLED:
        return None
    from src.retriever import get_reranker

    return get_reranker()


def _init_state():
    """session_state 初始化：每个浏览器会话独立的"保险箱"。"""
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))  # 本场对话的记忆档案编号
    st.session_state.setdefault("messages", [])  # 界面聊天记录 [{role, content}]
    st.session_state.setdefault("kb_stats", None)  # 知识库统计信息


def _count_pdfs() -> int:
    return len(list(PDF_DIR.glob("*.pdf")))


def _rebuild_knowledge_base():
    """全量重建知识库：解析所有 PDF -> 章节感知切分 -> FAISS 向量索引 + BM25 语料。"""
    if _count_pdfs() == 0:
        st.sidebar.warning("请先上传至少一个 PDF")
        return
    with st.spinner("正在解析文献并构建双路索引（FAISS + BM25）……"):
        chunks = []
        for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
            chunks.extend(load_pdf(pdf_file))  # v2：load_pdf 直接返回章节感知的知识块
        n = build_and_save_index(chunks, get_embeddings())
        build_bm25_index(chunks)  # BM25 语料与 FAISS 同步落盘，检索时双路并行
    st.session_state.kb_stats = {"pdfs": _count_pdfs(), "chunks": n}
    st.sidebar.success(f"知识库构建完成：{_count_pdfs()} 篇文献 / {n} 个知识块")


def _save_uploads(uploads: list[UploadedFile]) -> int:
    """把用户上传的文件写入 data/pdfs（同名覆盖）。"""
    for up in uploads:
        dest = PDF_DIR / up.name
        with open(dest, "wb") as f:
            shutil.copyfileobj(up.file, f)
    return len(uploads)


# ---------- 界面布局 ----------
_init_state()

st.title("📚 科研文献智能助手")
st.caption("基于 LangGraph 的 Agentic RAG：Agent 自主决定何时检索文献、何时计算指标，回答附带来源追溯。")

with st.sidebar:
    st.header("知识库管理")

    uploads = st.file_uploader(
        "上传 PDF 文献",
        type=["pdf"],
        accept_multiple_files=True,
        help="支持同时上传多篇；文件将保存到 data/pdfs",
    )
    if uploads and st.button("保存并重建知识库", type="primary"):
        n = _save_uploads(uploads)
        st.sidebar.info(f"已保存 {n} 个文件，开始构建……")
        _rebuild_knowledge_base()

    if st.button("重建知识库（不新增文件）"):
        _rebuild_knowledge_base()

    st.divider()
    kb = load_index(get_embeddings())
    bm25_ready = load_bm25() is not None
    if st.session_state.kb_stats:
        stats = st.session_state.kb_stats
    elif kb:  # 磁盘上已有索引：显示真实入库块数（ntotal 是 FAISS 索引的向量总数）
        stats = {"pdfs": _count_pdfs(), "chunks": kb.index.ntotal}
    else:
        stats = {"pdfs": _count_pdfs(), "chunks": "未构建"}
    st.metric("知识库文献数", stats["pdfs"])
    st.metric("知识块数量", stats["chunks"])
    st.caption("双路索引存于 data/vector_index/，重启程序后无需重建。")

    st.divider()
    st.subheader("检索管线")
    st.markdown(
        f"- 双路召回：FAISS 向量 + BM25 关键词（RRF 融合）\n"
        f"- 精排：`{RERANK_MODEL.split('/')[-1]}`" if RERANK_ENABLED else "- 精排：已关闭",
        )
    if RERANK_ENABLED:
        if _cached_reranker() is not None:
            st.success("重排模型已加载")
        else:
            st.warning("重排模型未加载")
    st.caption(f"每路召回 Top-8，融合重排后取 Top-{TOP_K} 片段。")

# ---------- 聊天区 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("traces"):
            with st.expander("🔍 本次知识库检索与计算记录（可追溯）"):
                for t in msg["traces"]:
                    st.markdown(f"**工具**：`{t['tool']}`")
                    st.caption(f"输入：{t['args']}")
                    st.text(t["output"][:1200] + ("……" if len(t["output"]) > 1200 else ""))
                    st.divider()

question = st.chat_input("向文献助手提问，例如：这几篇文献分别用了什么模型？平均准确率是多少？")

if question:
    if not kb:
        st.warning("知识库尚未构建，请先在左侧上传文献并构建。当前仅可回答通用问题。")

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中……"):
            try:
                result = ask(_cached_agent(), question, st.session_state.thread_id)
                answer = extract_final_answer(result)
                traces = [t for t in extract_tool_traces(result) if t["tool"]]
            except Exception as e:  # noqa: BLE001  界面层兜底：任何错误都转成可读提示
                answer, traces = f"出错了：{e}", []
        st.markdown(answer)

        if traces:
            with st.expander("🔍 本次知识库检索与计算记录（可追溯）"):
                for t in traces:
                    st.markdown(f"**工具**：`{t['tool']}`")
                    st.caption(f"输入：{t['args']}")
                    st.text(t["output"][:1200] + ("……" if len(t["output"]) > 1200 else ""))
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "traces": traces}
    )
