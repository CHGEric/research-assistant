"""
索引重建脚本 —— 一键构建双路索引（FAISS 向量库 + BM25 语料库）

运行：.venv\\Scripts\\python.exe scripts\\rebuild_index.py

什么时候需要跑：
  - 新增/删除/替换了 data/pdfs 下的文献
  - 调整了切分参数（config.py 的 CHUNK_SIZE / CHUNK_OVERLAP）
  - 更换了 Embedding 模型
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PDF_DIR, get_embeddings  # noqa: E402
from src.document_loader import load_pdf_dir  # noqa: E402
from src.retriever import build_bm25_index  # noqa: E402
from src.vector_store import build_and_save_index  # noqa: E402


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"data/pdfs 下没有 PDF 文件：{PDF_DIR}")
        sys.exit(1)

    print(f"发现 {len(pdf_files)} 篇文献，开始章节感知解析……")
    chunks = load_pdf_dir(PDF_DIR)
    print(f"解析完成：共 {len(chunks)} 个知识块")

    sections = {}
    titles = {}
    for c in chunks:
        sections[c.metadata["source"]] = sections.get(c.metadata["source"], set())
        sections[c.metadata["source"]].add(c.metadata.get("section", ""))
        titles[c.metadata["source"]] = c.metadata.get("paper_title", "")
    for src in sorted(sections):
        secs = [s for s in sections[src] if s]
        print(f"  - {src}（{titles[src] or src}）：{len(secs)} 个章节 -> {', '.join(sorted(secs))}")

    print("构建 FAISS 向量索引（调用 Embedding API）……")
    n = build_and_save_index(chunks, get_embeddings())
    print(f"FAISS 入库 {n} 块")

    print("构建 BM25 语料库（jieba 分词）……")
    build_bm25_index(chunks)
    print("BM25 语料落盘完成")

    print("\n双路索引构建完成：向量索引与关键词索引均已就绪。")


if __name__ == "__main__":
    main()
