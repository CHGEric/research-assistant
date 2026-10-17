"""
向量库模块 —— RAG 流水线的第二站（Embedding + FAISS）

【通俗理解】这是 Agent 的"私人图书馆"，干三件事：

  1. 建馆（build_index）：把每个知识块用 Embedding 模型变成向量，
     存进 FAISS——一个专门做"向量近似检索"的库，亿级向量也能毫秒级查。
  2. 开馆（load_index）：索引存在硬盘上，程序重启不用重新建馆。
  3. 查询（search）：用户的问题也被转成向量，FAISS 找出"距离最近"的 K 个块。
     这就是语义检索：搜"过拟合怎么解决"，能召回写着"正则化/早停/数据增强"的段落。

【为什么叫"向量"检索】
  Embedding 把文本映射到几千维空间中的一个点，语义相近的点彼此靠近。
  检索 = 在这个空间里找用户问题点的"最近邻居"（Top-K）。
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import INDEX_DIR, TOP_K


def build_and_save_index(chunks, embeddings, index_dir=INDEX_DIR) -> int:
    """
    用知识块建立 FAISS 索引并落盘。

    FAISS.from_documents 内部做了两件事：
      a. 调用 embeddings 把每个 chunk 的文本转成向量
      b. 把向量 + 原文 + metadata 一起存入 FAISS 索引
    返回 chunk 数量，供界面显示"知识库里有多少知识块"。
    """
    if not chunks:
        raise ValueError("没有可用的文本块，请先上传 PDF")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir))
    return len(chunks)


def load_index(embeddings, index_dir=INDEX_DIR):
    """
    从硬盘加载已存在的索引；不存在则返回 None。

    allow_dangerous_deserialization=True：
    FAISS 索引文件用 pickle 序列化，LangChain 要求显式确认（因为 pickle
    理论上可执行任意代码）。这里的文件是我们自己刚生成的，属于受控来源。
    """
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search(vectorstore, query: str, k: int = TOP_K):
    """
    语义检索：返回与 query 最相关的 k 个知识块。

    similarity_search 默认按"向量距离"升序返回，第一个结果最相关。
    每个 Document 自带 metadata，所以检索结果天然携带"文件名+页码"。
    """
    return vectorstore.similarity_search(query, k=k)
