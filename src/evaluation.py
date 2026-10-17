"""
评测模块 —— 量化回答"我的检索到底好不好"

【通俗理解】没有评测的 RAG 优化就是"凭感觉调参"。
  这个模块把"检索质量"变成三个可计算的数字：

  Recall@K  —— 问 100 个问题，金标答案出现在 Top-K 结果里的比例（查全能力）
  MRR       —— 金标答案排在第 1 位得 1 分、第 2 位得 0.5 分……未出现得 0（排序质量）
  引用正确率 —— 端到端跑 Agent，回答中是否引用了正确的文献（最终交付质量）

评测思想（和机器学习实验一脉相承）：
  1. 人工标注一个"金标集"：每道题标注正确出处（source + 关键词）
  2. 用同一组题跑不同配置，控制变量对照
  3. 未命中的题就是 badcase，逐个人工分析"为什么没召回"
"""

import json
import time
from pathlib import Path
from statistics import mean

from src.config import PROJECT_ROOT, TOP_K

DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.json"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "data" / "eval"


def load_eval_set(path=DEFAULT_EVAL_PATH) -> list[dict]:
    """加载金标评测集（data/eval/eval_set.json）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def is_hit(doc, item: dict) -> bool:
    """
    单个检索结果是否命中金标：来源文件匹配 且 正文含任一关键词。

    【为什么双重条件】只查文件名会误判"召回了对的论文但错的章节"；
      加关键词校验确保召回的正是包含答案的那个知识块。
    """
    if doc.metadata.get("source") != item["gold_source"]:
        return False
    return any(kw in doc.page_content for kw in item["gold_keywords"])


def eval_retrieval(search_fn, eval_set: list[dict], k: int = TOP_K) -> dict:
    """
    评测一个检索配置：search_fn(query) -> list[Document]（已按相关性排序）。

    返回 Recall@K / MRR / 平均延迟 / 未命中题目（badcase 原料）。
    """
    recalls, mrrs, latencies, misses = [], [], [], []
    per_type: dict[str, list[float]] = {}  # 分题型统计 Recall，定位弱项
    per_item: list[dict] = []  # 每题命中明细（子集分析用）

    for item in eval_set:
        t0 = time.perf_counter()
        docs = search_fn(item["question"])
        latencies.append(time.perf_counter() - t0)

        ranks = [i + 1 for i, d in enumerate(docs[:k]) if is_hit(d, item)]
        recall = 1.0 if ranks else 0.0
        mrr = 1.0 / ranks[0] if ranks else 0.0
        recalls.append(recall)
        mrrs.append(mrr)
        per_type.setdefault(item["type"], []).append(recall)
        if not ranks:
            misses.append({"question": item["question"], "type": item["type"]})
        per_item.append({
            "id": item.get("id"),
            "type": item["type"],
            "hit": bool(ranks),
            "rank": ranks[0] if ranks else None,
        })

    return {
        "n": len(eval_set),
        "k": k,
        "recall_at_k": round(mean(recalls), 4),
        "mrr": round(mean(mrrs), 4),
        "latency_ms": round(mean(latencies) * 1000, 1),
        "recall_by_type": {t: round(mean(v), 4) for t, v in per_type.items()},
        "per_item": per_item,
        "misses": misses,
    }


def eval_e2e_citation(agent, eval_set: list[dict], limit: int = 10) -> dict:
    """
    端到端引用正确率：真实跑 Agent，检查回答是否可追溯到正确文献。

    判定（三选一即算引用正确）：
      a. 回答文本中出现金标文件名或论文标题
      b. 回答中的 [n] 编号能映射到本轮检索结果中的金标来源
         （Agent 的检索工具输出带"[n] 来源：xxx.pdf"，回答标注 [n] 即可追溯）
    每题使用独立 thread_id，避免多轮上下文互相污染评测。
    只跑前 limit 题——每题要调一次 LLM，全跑成本高。
    """
    import re

    from src.agent import ask, extract_final_answer, extract_tool_traces

    cite_pattern = re.compile(r"\[(\d+)\] 来源：(\S+?\.pdf)")

    correct, details = 0, []
    for item in eval_set[:limit]:
        result = ask(agent, item["question"], thread_id=f"eval-e2e-{item['id']}")
        answer = extract_final_answer(result)

        # 从本轮检索工具输出中建立 编号 -> 文件名 的映射
        cite_map: dict[str, str] = {}
        for t in extract_tool_traces(result):
            if t["tool"] == "search_literature":
                for num, src in cite_pattern.findall(t["output"]):
                    cite_map[num] = src

        cited_sources = {cite_map.get(n) for n in re.findall(r"\[(\d+)\]", answer)}
        cited_sources.discard(None)
        ok = (
            item["gold_source"] in answer
            or item["gold_title"] in answer
            or item["gold_source"] in cited_sources
        )
        correct += int(ok)
        details.append({"question": item["question"], "cited_correctly": ok})
    return {
        "n": min(limit, len(eval_set)),
        "citation_accuracy": round(correct / max(1, min(limit, len(eval_set))), 4),
        "details": details,
    }


def save_results(results: dict, result_dir: Path = DEFAULT_RESULT_DIR) -> Path:
    """把对照实验结果落盘（JSON），文件名带时间戳，保留历史迭代记录。"""
    result_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = result_dir / f"results_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path
