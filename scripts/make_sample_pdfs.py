"""
生成 3 篇模拟科研文献 PDF（虚构内容，仅供测试知识库、混合检索与评测）

【v2 变更】章节标题独立成行（13pt）且与正文（11pt）字号区分，
  用于测试"章节感知切分"能否正确识别章节边界；
  sample_01 额外带第 2 页（相关工作/讨论），测试跨页章节延续逻辑。

运行：uv run python scripts/make_sample_pdfs.py
"""

import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PDF_DIR  # noqa: E402

# 论文结构：file -> pages -> [(章节标题, 章节正文), ...]
#   章节标题会以 13pt 独立成行；正文以 11pt 渲染
PAPERS = [
    {
        "file": "sample_01_resnet_variants.pdf",
        "title": "基于残差网络变体的图像分类研究",
        "pages": [
            [  # 第 1 页
                ("摘要", "本文提出一种改进的残差网络结构 ResNet-V，通过引入注意力模块提升图像分类性能。在 CIFAR-10 数据集上取得 91.2% 的分类准确率，相比基线 ResNet-18 的 89.5% 提升 1.7 个百分点。"),
                ("方法", "我们以 ResNet-18 为骨干网络，在残差块之间插入通道注意力模块（SE Block），并采用余弦退火学习率调度策略。训练使用 SGD 优化器，初始学习率 0.1，batch size 为 128，共训练 200 个 epoch。"),
                ("实验结果", "在 CIFAR-10 测试集上，ResNet-V 达到准确率 91.2%，宏平均 F1 分数 0.905，推理速度 132 张/秒。消融实验显示注意力模块贡献了 1.1 个百分点的提升，余弦退火调度贡献 0.6 个百分点。"),
            ],
            [  # 第 2 页：测试跨页章节
                ("相关工作", "残差网络通过跳跃连接缓解深层网络的梯度消失问题。近期研究表明，注意力机制（如 SENet、CBAM）能以极小的计算开销增强通道特征表达。本文将 SE Block 引入残差分支属于该方向的延续性工作。"),
                ("讨论", "注意力机制与学习率调度的组合能有效提升残差网络的分类性能，且推理开销增加不到 3%。未来工作将探索在更大规模数据集（ImageNet）上的可扩展性。"),
            ],
        ],
    },
    {
        "file": "sample_02_vit_small.pdf",
        "title": "轻量级 Vision Transformer 的迁移学习研究",
        "pages": [
            [
                ("摘要", "本文研究轻量级 ViT-Small 模型在小样本数据集上的迁移学习表现。通过 ImageNet 预训练权重微调，在 Oxford Flowers 数据集上达到 94.8% 的准确率。"),
                ("方法", "采用 ViT-Small（22M 参数）作为骨干，冻结前 6 层 Transformer 编码器，仅微调后 6 层与分类头。使用 AdamW 优化器，学习率 3e-4，warmup 5 个 epoch，数据增强采用 RandAugment。"),
                ("实验结果", "ViT-Small 在 Oxford Flowers 上取得准确率 94.8%，F1 分数 0.941，优于同规模的 MobileNetV3 的 92.3%。训练仅耗时 35 分钟（单张 V100 GPU）。"),
                ("结论", "预训练 ViT 在小样本迁移任务上仍有竞争力，冻结早期层可减少 40% 训练显存占用。"),
            ],
        ],
    },
    {
        "file": "sample_03_gnn_recommendation.pdf",
        "title": "基于图神经网络的推荐系统冷启动研究",
        "pages": [
            [
                ("摘要", "针对推荐系统的冷启动问题，本文提出基于 GraphSAGE 的用户-商品二部图建模方法。在 MovieLens-1M 数据集上，新用户首周推荐的点击率提升 18.6%。"),
                ("方法", "构建用户-商品交互二部图，使用 GraphSAGE 聚合二阶邻居信息，融合用户侧特征（注册渠道、设备类型）与商品侧特征（类目、价格分桶）。训练损失为 BPR 排序损失，负采样比例 1:5。"),
                ("实验结果", "冷启动用户场景下，模型 AUC 达到 0.861，召回率@20 为 0.397，点击率相比基线协同过滤（CF）提升 18.6%。全量用户场景 AUC 为 0.883。"),
                ("结论", "图结构信息能显著缓解冷启动数据稀疏问题，且推理延迟控制在 15ms 以内，满足在线服务要求。"),
            ],
        ],
    },
]

TITLE_SIZE = 16      # 论文大标题
SECTION_SIZE = 13    # 章节标题（章节感知的识别目标）
BODY_SIZE = 11       # 正文


def _wrap(text: str, width: int = 40) -> list[str]:
    """按显示宽度折行（中文按字符计）。"""
    return [text[i : i + width] for i in range(0, len(text), width)]


def make_pdf(path: Path, title: str, pages: list[list[tuple[str, str]]]):
    doc = pymupdf.open()
    for page_idx, page_sections in enumerate(pages):
        page = doc.new_page()  # A4
        y = 72
        if page_idx == 0:  # 首页才渲染论文大标题
            page.insert_text((72, y), title, fontsize=TITLE_SIZE, fontname="china-s")
            y += 28
        for section_title, body in page_sections:
            page.insert_text((72, y), section_title, fontsize=SECTION_SIZE, fontname="china-s")
            y += 20
            for line in _wrap(body):
                page.insert_text((72, y), line, fontsize=BODY_SIZE, fontname="china-s")
                y += 16
            y += 8
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    for paper in PAPERS:
        make_pdf(PDF_DIR / paper["file"], paper["title"], paper["pages"])
    print(f"已生成 {len(PAPERS)} 篇模拟文献（含章节结构）至 {PDF_DIR}")
