"""
对照实验脚本 —— 三种检索配置同题竞技

运行：.venv\\Scripts\\python.exe scripts\\run_evaluation.py
可选：--e2e 额外跑端到端引用正确率（会真实调用 LLM，较慢）

实验设计（控制变量）：
  配置1 纯向量检索      —— 基线（v1 项目的检索方式）
  配置2 BM25+向量+RRF   —— 加稀疏路，不加精排
  配置3 混合+Cross-Encoder 重排 —— 完整管线（v2 上线配置）

同一金标集（data/eval/eval_set.json，18 题）跑三种配置，
输出 Recall@K / MRR / 平均延迟对比表 + badcase 清单，
结果落盘 data/eval/results_时间戳.json —— 每次调参后重跑即可累积迭代证据。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PDF_DIR, TOP_K, get_embeddings  # noqa: E402
from src.evaluation import (  # noqa: E402
    eval_e2e_citation,
    eval_retrieval,
    load_eval_set,
    save_results,
)
from src.retriever import hybrid_search  # noqa: E402
from src.vector_store import load_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true", help="额外跑端到端引用正确率")
    parser.add_argument("--e2e-limit", type=int, default=10)
    args = parser.parse_args()

    eval_set = load_eval_set()

    # 金标文献缺失的题自动跳过：跨语言子集基于 2 篇版权论文（不随仓库分发），
    # 克隆后仅放样例文献时跳过对应题，保证可复现口径与 README 一致。
    available = {p.name for p in PDF_DIR.glob("*.pdf")}
    dropped = [it for it in eval_set if it["gold_source"] not in available]
    if dropped:
        print(f"跳过 {len(dropped)} 题：金标文献不在 data/pdfs（如未放置的版权论文）")
        eval_set = [it for it in eval_set if it["gold_source"] in available]
    print(f"加载金标评测集：{len(eval_set)} 题\n")

    vectorstore = load_index(get_embeddings())
    if vectorstore is None:
        print("错误：向量索引不存在，请先构建知识库")
        sys.exit(1)

    def dense_only(q: str):
        """配置1：纯向量检索（基线）。"""
        return vectorstore.similarity_search(q, k=TOP_K)

    def hybrid_rrf_only(q: str):
        """配置2：BM25+向量双路召回 + RRF 融合，不做重排。"""
        return hybrid_search(q, top_k=TOP_K, use_rerank=False)

    def hybrid_full(q: str):
        """配置3：完整管线（双路 + RRF + Cross-Encoder 重排）。"""
        return hybrid_search(q, top_k=TOP_K, use_rerank=True)

    configs = [
        ("纯向量检索(基线)", dense_only),
        ("BM25+向量+RRF", hybrid_rrf_only),
        ("混合检索+Cross-Encoder重排", hybrid_full),
    ]

    results = {}
    for name, fn in configs:
        print(f"评测中：{name} ...")
        metrics = eval_retrieval(fn, eval_set, k=TOP_K)
        results[name] = metrics
        print(
            f"  Recall@{TOP_K}={metrics['recall_at_k']:.2%}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"平均延迟={metrics['latency_ms']:.0f}ms"
        )
        for t, v in metrics["recall_by_type"].items():
            print(f"    - {t} 类 Recall: {v:.2%}")
        print()

    # badcase 分析：以最优配置（完整管线）的未命中题为准
    full_metrics = results[configs[-1][0]]
    if full_metrics["misses"]:
        print("=== badcase 清单（完整管线未召回，建议人工归因）===")
        for m in full_metrics["misses"]:
            print(f"  [{m['type']}] {m['question']}")
    else:
        print("完整管线无未命中题目。")

    if args.e2e:
        print("\n跑端到端引用正确率（真实调用 Agent，约需数分钟）...")
        from src.agent import get_agent

        agent = get_agent()
        e2e = eval_e2e_citation(agent, eval_set, limit=args.e2e_limit)
        results["端到端引用正确率(完整管线)"] = e2e
        print(f"  引用正确率：{e2e['citation_accuracy']:.2%}（{e2e['n']} 题）")

    path = save_results({"top_k": TOP_K, "configs": results})
    print(f"\n结果已保存：{path}")


if __name__ == "__main__":
    main()
