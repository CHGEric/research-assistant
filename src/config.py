"""
配置中心 —— 项目的"总开关"

【通俗理解】把所有可调参数集中在一个地方，就像家里的配电箱：
  改模型、改路径、改检索数量，都只动这一个文件，其他模块只管"读配置"。

这个文件定义了：
  1. API 地址与密钥（智谱，OpenAI 兼容模式）
  2. 用哪个对话模型 / Embedding 模型
  3. 数据与索引的存放路径
  4. 文本切分与检索参数（RAG 的核心旋钮）
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---- 项目根目录与 .env 加载 ------------------------------------------------
# Path(__file__) 是本文件，向上两级就是项目根目录（src/config.py -> 项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ---- API 配置（智谱，OpenAI 兼容模式）--------------------------------------
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 对话模型：glm-4-flash 永久免费，适合学习与 MVP
# 面向生产的可插拔设计：换模型只改这一个字符串，例如 "glm-4-plus"
CHAT_MODEL = "glm-4-flash"

# Embedding 模型：把文本变成"语义坐标"的模型（详见 AGENT_GUIDE.md）
# 智谱 embedding-3 默认输出 2048 维向量，走 OpenAI 兼容接口
EMBEDDING_MODEL = "embedding-3"

# ---- 路径 -------------------------------------------------------------------
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"          # 用户上传的文献
INDEX_DIR = PROJECT_ROOT / "data" / "vector_index"  # FAISS 向量索引

# 目录不存在则自动创建（exist_ok=True 表示已存在也不报错）
PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---- RAG 参数（最常调优的三个旋钮）------------------------------------------
# CHUNK_SIZE    每个文本块的字符上限。太大→检索不精准；太小→上下文破碎
# CHUNK_OVERLAP 相邻块的重叠字符数。防止一句话的关键信息被切断在两个块里
# TOP_K         每次检索取回最相似的 K 个块。本科研问答 K=4~6 常见
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 1

# ---- 混合检索与重排参数 ------------------------------------------------------
# RECALL_N      双路召回阶段每路各取 N 个候选（向量一路 + BM25 一路）
#               候选池要"宁多勿漏"，最终精筛交给 Reranker
# RRF_K         Reciprocal Rank Fusion 的平滑常数（社区惯例取 60）：
#               score(d) = Σ 1/(RRF_K + rank_i(d))，rank 从 1 计
# RERANK_MODEL  Cross-Encoder 重排模型（BGE 中文优化版，本地 CPU 可跑）
# RERANK_ENABLED 关掉可退回纯 RRF，用于评测对照与降级容错
RECALL_N = 8
RRF_K = 60
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_ENABLED = True

# HuggingFace 国内镜像：模型下载走 hf-mirror，避免直连超时
# setdefault：用户自己设了 HF_ENDPOINT 时不覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ---- 模型工厂函数 -----------------------------------------------------------
def get_chat_model():
    """
    创建对话模型实例（ChatOpenAI 兼容层）。

    【通俗理解】相当于给 Agent 配一个"大脑"。
    返回的对象支持 invoke（一问一答）、stream（流式输出）、tool calling（调用工具）。
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        f"openai:{CHAT_MODEL}",
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
        temperature=0.1,  # 越接近 0 回答越稳定，科研问答要的是"可复现"
    )


def get_embeddings():
    """
    创建 Embedding 实例。

    【通俗理解】Embedding 模型的工作只有一件事：
      把一段文字变成一个高维向量（一串数字），语义越接近的文本，向量距离越近。
    这样"检索"就从"关键词匹配"升级成了"语义匹配"——
      搜"模型效果不好"，也能召回写着"性能下降/欠拟合"的段落。
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=ZHIPU_BASE_URL,
        api_key=ZHIPU_API_KEY,
    )
