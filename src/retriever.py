"""
混合检索与重排模块 —— 简历核心亮点的落地

【为什么要"混合"检索】
  单一向量检索（Dense）靠语义相似，但有个软肋：专业术语、缩写、指标名
  （如 "BPR 排序损失"、"0.905"、"RandAugment"）在 Embedding 空间里未必靠前，
  而这些恰恰是 BM25 这类"关键词检索"（Sparse）的强项——字面命中就召回。
  反过来，"如何缓解冷启动"这类自然语言问法，向量检索远强于 BM25。
  → 两路各取所长，用 RRF 融合，再用 Cross-Encoder 精筛。

管线（一次 search 的完整路径）：
  用户 query
    ├─ 路A Dense：query 向量化 → FAISS 取 Top-RECALL_N
    ├─ 路B Sparse：jieba 分词 → BM25 取 Top-RECALL_N
    ├─ RRF 融合：score(d) = Σ 1/(60 + rank_i(d))，两路并集去重排序
    └─ Cross-Encoder 重排：模型逐对精读 (query, 片段) 打分 → 取最终 Top-K

【RRF 为什么好用】两路分数量纲完全不同（余弦距离 vs BM25 分），直接加权
  需要调参归一；RRF 只用"排名"不用"分数"，天然免调参，工程上极其稳健。

【Cross-Encoder 与向量检索的本质区别】
  向量检索：query 和文档"各自"编码再比距离（Bi-Encoder，快但粗）
  Cross-Encoder：query 和文档"拼一起"过模型，逐 token 交互（慢但准）
  → 所以工程上永远先粗召回、后精重排，兼顾速度与精度。
"""

import pickle
from pathlib import Path

from langchain_core.documents import Document

from src.config import (
    INDEX_DIR,
    RECALL_N,
    RERANK_ENABLED,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
    get_embeddings,
)

BM25_FILE = "bm25_data.pkl"  # 与 FAISS 索引同目录存放

# ---------------------------------------------------------------------------
# BM25：稀疏关键词检索路
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """jieba 中文分词 + 过滤标点/空白。BM25 的最小检索单位是"词"。"""
    import jieba

    tokens = jieba.lcut(text.lower())
    return [t for t in tokens if t.strip() and any(ch.isalnum() for ch in t)]


def build_bm25_index(chunks: list[Document], index_dir=INDEX_DIR) -> int:
    """
    为全部知识块构建 BM25 语料并持久化。

    BM25 本身是"查询时现算"的模型，没有可保存的权重——真正要存的是
    语料（分词结果 + 原文 + 元数据），加载后重建 BM25Okapi 即可（重建是毫秒级）。
    """
    from rank_bm25 import BM25Okapi

    if not chunks:
        raise ValueError("没有可用的文本块，请先解析 PDF")

    data = {
        "tokens": [_tokenize(d.page_content) for d in chunks],
        "texts": [d.page_content for d in chunks],
        "metadatas": [d.metadata for d in chunks],
    }
    # 语料构建一次（触发内部统计量计算），确保落盘前可用
    BM25Okapi(data["tokens"])

    path = Path(index_dir) / BM25_FILE
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return len(chunks)


def load_bm25(index_dir=INDEX_DIR):
    """加载 BM25 语料并重建检索器；不存在则返回 None。"""
    from rank_bm25 import BM25Okapi

    path = Path(index_dir) / BM25_FILE
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    docs = [
        Document(page_content=t, metadata=m) for t, m in zip(data["texts"], data["metadatas"])
    ]
    return BM25Okapi(data["tokens"]), docs


def bm25_search(bm25, docs: list[Document], query: str, n: int = RECALL_N) -> list[Document]:
    """BM25 检索：返回得分最高的 n 个知识块。"""
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [docs[i] for i in ranked if scores[i] > 0]


# ---------------------------------------------------------------------------
# RRF：双路融合
# ---------------------------------------------------------------------------

def rrf_fuse(result_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """
    Reciprocal Rank Fusion：按"排名"融合多路结果。

    score(d) = Σ_routes 1 / (k + rank(d))，rank 从 1 计。
    同一个块被两路同时召回会累加两次"倒数排名分"——
    即"两路都认为相关"的块自动排到最前面，这正是我们想要的信号。
    """
    scores: dict[int, float] = {}
    docs_by_key: dict[int, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = hash(doc.page_content + str(sorted(doc.metadata.items())))
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [docs_by_key[key] for key in ranked_keys]


# ---------------------------------------------------------------------------
# Cross-Encoder 重排
# ---------------------------------------------------------------------------

_reranker = None  # 进程内单例：模型 1.1GB，加载一次全程复用


def get_reranker():
    """
    加载 Cross-Encoder 重排模型（BGE 中文优化版）。

    单例的原因：模型加载要数秒 + 上 GB 内存，每次检索都重建会拖垮性能。
    Streamlit 端用 st.cache_resource 包一层可实现跨会话复用。
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker


def rerank(query: str, candidates: list[Document], top_k: int = TOP_K) -> list[Document]:
    """用 Cross-Encoder 对候选逐一精读打分，返回分数最高的 top_k 块。"""
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, d.page_content) for d in candidates]
    scores = model.predict(pairs)  # 返回与 pairs 等长的相关性分数
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 混合检索总入口
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    use_rerank: bool | None = None,
    use_sparse: bool = True,
) -> list[Document]:
    """
    混合检索总入口：双路召回 → RRF 融合 →（可选）Cross-Encoder 重排。

    参数：
      use_rerank  None=按配置文件；True/False 强制开/关（评测对照实验用）
      use_sparse  False 可退回纯向量检索（即 v1 基线，用于对照）
    任一路索引缺失时自动降级，保证检索永远可用（工程容错）。
    """
    if use_rerank is None:
        use_rerank = RERANK_ENABLED

    from src.vector_store import load_index

    vectorstore = load_index(get_embeddings())
    if vectorstore is None:
        return []

    # 路 A：Dense 向量召回
    dense = vectorstore.similarity_search(query, k=RECALL_N)

    # 路 B：Sparse BM25 召回（缺索引则自动降级为纯向量）
    sparse: list[Document] = []
    if use_sparse:
        loaded = load_bm25()
        if loaded is not None:
            bm25, docs = loaded
            sparse = bm25_search(bm25, docs, query, n=RECALL_N)

    fused = rrf_fuse([dense, sparse])

    # 精筛：reranker 关闭时直接取融合排名前 top_k（纯 RRF 配置）
    if not use_rerank or not fused:
        return fused[:top_k]
    try:
        return rerank(query, fused, top_k=top_k)
    except Exception as e:  # noqa: BLE001  重排失败（如模型未下载）不阻断检索
        print(f"[retriever] 重排降级为 RRF 排序：{e}")
        return fused[:top_k]
